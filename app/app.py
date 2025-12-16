import sys
import os
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Environment + Memory knobs
# -----------------------------
IS_RENDER = (
    os.environ.get("RENDER", "").lower() == "true"
    or os.environ.get("RENDER_SERVICE_ID") is not None
)

# Keep conservative for Render free tier
MAX_ROWS_SCORE = 10000 if IS_RENDER else 200000        # max rows to score
MAX_ROWS_STATE = 2000 if IS_RENDER else 20000          # max rows stored in session_state
MAX_ROWS_SHAP = 100 if IS_RENDER else 200              # shap cap
MAX_ROWS_DOWNLOAD_RENDER = 10000                       # max rows allowed to download on Render

# -----------------------------
# Paths (ABSOLUTE, Render-safe)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # project root

# ensure `src` import works
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCHEMA_PATH = BASE_DIR / "reports" / "schema.json"
SHAP_BG_PATH = BASE_DIR / "data" / "processed" / "shap_background.csv"
DRIFT_BASELINE_PATH = BASE_DIR / "data" / "processed" / "train_baseline_sample.csv"

# -----------------------------
# Imports from your src/
# -----------------------------
from src.explain import make_explainer, shap_values_for, top_k_reasons
from src.drift import drift_report
from src.feedback import init_db, write_feedback, DB_PATH

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="FinTech Fraud Detector", layout="wide")
st.title("FinTech Fraud Detector")
st.write("Upload transactions CSV. If your file includes `Class`, the app will also show evaluation metrics.")

if IS_RENDER:
    st.caption("Running on Render: using memory-safe limits (row caps + smaller session storage).")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def safe_read_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Reduce RAM significantly (safe for numeric features)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype("float32")
            elif pd.api.types.is_integer_dtype(df[c]):
                df[c] = df[c].astype("int32")
    return df

pipe = load_model()

tab_score, tab_health, tab_feedback = st.tabs(["Score", "Model Health (Drift)", "Feedback"])

# -----------------------------
# SCORE TAB
# -----------------------------
with tab_score:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader_score")

    if uploaded is None:
        st.info("Upload a CSV in this tab first. Then Drift + Feedback tabs will work automatically.")
    else:
        try:
            df_in = safe_read_csv(uploaded)

            # Optional labels
            y_true = None
            if "Class" in df_in.columns:
                y_true = df_in["Class"].astype("int8").values
                df = df_in.drop(columns=["Class"])
                st.info("Detected `Class` column. Evaluation metrics will be shown.")
            else:
                df = df_in

            # Schema validation + enforce order
            if SCHEMA_PATH.exists():
                with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                expected_cols = schema.get("columns", [])

                if expected_cols:
                    missing = [c for c in expected_cols if c not in df.columns]
                    extra = [c for c in df.columns if c not in expected_cols]

                    if missing:
                        st.error(f"Missing columns (first 15): {missing[:15]}")
                        st.stop()

                    if extra:
                        st.warning(f"Extra columns ignored (first 15): {extra[:15]}")

                    df = df[expected_cols]  # enforce exact column order

            # Hard cap rows BEFORE heavy ops (prevents Render OOM)
            if len(df) > MAX_ROWS_SCORE:
                st.warning(
                    f"Large file ({len(df)} rows). "
                    f"Scoring first {MAX_ROWS_SCORE} rows to avoid memory crash."
                )
                df = df.head(MAX_ROWS_SCORE)
                if y_true is not None:
                    y_true = y_true[:MAX_ROWS_SCORE]

            # Downcast numerics to reduce RAM
            df = downcast_numeric(df)

            threshold = st.slider("Fraud threshold", 0.0, 1.0, 0.50, 0.01, key="threshold")

            # Predict probabilities
            proba = pipe.predict_proba(df)[:, 1].astype("float32")

            # Build LIGHTWEIGHT results (avoid df.copy())
            results = pd.DataFrame(index=df.index)
            results["fraud_probability"] = proba
            results["fraud_pred"] = (proba >= threshold).astype("int8")
            results["risk_band"] = pd.cut(
                proba,
                bins=[-0.01, 0.2, 0.7, 1.01],
                labels=["LOW", "MEDIUM", "HIGH"],
            ).astype("category")

            # Add labels back if present (lightweight)
            if y_true is not None:
                results["Class"] = y_true

                tp = int(((y_true == 1) & (results["fraud_pred"].values == 1)).sum())
                fp = int(((y_true == 0) & (results["fraud_pred"].values == 1)).sum())
                fn = int(((y_true == 1) & (results["fraud_pred"].values == 0)).sum())
                tn = int(((y_true == 0) & (results["fraud_pred"].values == 0)).sum())

                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0

                st.subheader("Evaluation (because `Class` exists in upload)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Precision", f"{precision:.3f}")
                c2.metric("Recall", f"{recall:.3f}")
                c3.metric("TP / FP", f"{tp} / {fp}")
                c4.metric("FN / TN", f"{fn} / {tn}")

            # Store ONLY small sample in session_state (critical for Render)
            st.session_state["live_df_for_drift"] = df.head(MAX_ROWS_STATE).copy()
            st.session_state["latest_results"] = results.head(MAX_ROWS_STATE).copy()
            st.session_state["has_scored"] = True

            # SHAP explainability (optional)
            st.subheader("Explainability (SHAP)")
            if IS_RENDER:
                st.caption("SHAP is limited on Render to prevent memory crashes.")

            show_explain = st.checkbox(
                "Compute SHAP explanations (Top-3 reasons)",
                value=False,
                key="shap_toggle"
            )

            if show_explain:
                if not SHAP_BG_PATH.exists():
                    st.warning(f"Missing SHAP background file: `{SHAP_BG_PATH}`. Re-run `python src/train.py`.")
                else:
                    bg = pd.read_csv(SHAP_BG_PATH)
                    bg = bg[[c for c in df.columns if c in bg.columns]]

                    explainer, scaler = make_explainer(pipe, bg)

                    n_explain = st.slider(
                        "Rows to explain (for speed)",
                        10,
                        MAX_ROWS_SHAP,
                        min(50, MAX_ROWS_SHAP),
                        10,
                        key="n_explain"
                    )
                    explain_df = df.head(n_explain).copy()

                    sv = shap_values_for(pipe, explainer, scaler, explain_df)
                    feature_names = list(explain_df.columns)

                    reasons = []
                    for i in range(min(n_explain, sv.shape[0])):
                        top3 = top_k_reasons(feature_names, sv[i], k=3)
                        reasons.append(", ".join([f"{f} ({v:+.3f})" for f, v in top3]))

                    results.loc[results.index[:len(reasons)], "top_reasons"] = reasons

                    st.dataframe(
                        results[["fraud_probability", "risk_band", "top_reasons"]].head(n_explain),
                        use_container_width=True,
                    )

            # Preview (Render-safe: only 50 rows, join is lighter than concat)
            st.subheader("Scored Results (preview)")
            preview = results.head(50).join(df.head(50), how="left")
            st.dataframe(preview, use_container_width=True)

            st.subheader("Risk Band Counts")
            st.bar_chart(results["risk_band"].value_counts())

            # -----------------------------
            # Download (Render-safe)
            # -----------------------------
            st.subheader("Download")

            download_mode = st.radio(
                "Choose download format",
                ["Results only (recommended)", "Results + input features (may be larger)"],
                index=0 if IS_RENDER else 1,
                key="download_mode",
            )

            if download_mode == "Results only (recommended)":
                out_df = results.copy()
            else:
                # df is already capped by MAX_ROWS_SCORE
                out_df = pd.concat(
                    [df.reset_index(drop=True), results.reset_index(drop=True)],
                    axis=1
                )

            if IS_RENDER and len(out_df) > MAX_ROWS_DOWNLOAD_RENDER:
                st.warning(f"Render memory-safe mode: downloading first {MAX_ROWS_DOWNLOAD_RENDER} rows only.")
                out_df = out_df.head(MAX_ROWS_DOWNLOAD_RENDER)

            st.download_button(
                "Download Scored CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_transactions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error("Scoring failed. See details below (this was causing 502).")
            st.exception(e)
            st.stop()

# -----------------------------
# DRIFT TAB
# -----------------------------
with tab_health:
    st.subheader("Drift Monitoring (PSI)")
    st.write("Compares uploaded data distribution vs a training baseline sample.")

    if not DRIFT_BASELINE_PATH.exists():
        st.warning(f"Missing drift baseline file: `{DRIFT_BASELINE_PATH}`. Re-run `python src/train.py`.")
    elif "live_df_for_drift" not in st.session_state:
        st.warning("No live data found yet. Go to Score tab, upload a CSV, then return here.")
    else:
        try:
            live_df = st.session_state["live_df_for_drift"]
            train_base = pd.read_csv(DRIFT_BASELINE_PATH)

            common_cols = [c for c in train_base.columns if c in live_df.columns]
            train_base = train_base[common_cols]
            live_df = live_df[common_cols]

            rep = drift_report(train_base, live_df, top_n=15)

            def status_tag(v):
                if v < 0.1:
                    return "🟢 OK"
                if v < 0.2:
                    return "🟡 Moderate"
                return "🔴 High"

            rep2 = rep.copy()
            rep2["status"] = rep2["psi"].apply(status_tag)
            st.dataframe(rep2, use_container_width=True)

        except Exception as e:
            st.error("Drift computation failed.")
            st.exception(e)
            st.stop()

# -----------------------------
# FEEDBACK TAB
# -----------------------------
with tab_feedback:
    st.subheader("Analyst Feedback Loop (SQLite)")
    st.write("Confirm fraud / mark false positives. On Render the DB is stored in a writable temp path.")
    st.caption(f"DB path: {DB_PATH}")

    try:
        init_db()
    except Exception as e:
        st.error(f"Feedback DB init failed: {e}")
        st.stop()

    if "latest_results" not in st.session_state:
        st.warning("No scored results found yet. Go to Score tab, upload a CSV, then return here.")
        st.stop()

    results_fb = st.session_state["latest_results"]
    st.dataframe(results_fb[["fraud_probability", "risk_band"]].head(20), use_container_width=True)

    max_idx = int(len(results_fb) - 1)
    row_id = st.number_input(
        "Pick a row index to review",
        min_value=0,
        max_value=max_idx,
        value=0,
        step=1,
        key="row_id"
    )

    sel = results_fb.iloc[int(row_id)]
    st.write(
        {
            "row_index": int(row_id),
            "fraud_probability": float(sel["fraud_probability"]),
            "risk_band": str(sel["risk_band"]),
        }
    )

    c1, c2 = st.columns(2)

    if c1.button("✅ Confirm Fraud", key="btn_confirm"):
        try:
            write_feedback(int(row_id), float(sel["fraud_probability"]), str(sel["risk_band"]), "CONFIRM_FRAUD")
            st.success("Saved feedback: CONFIRM_FRAUD")
        except Exception as e:
            st.error("Failed to save feedback.")
            st.exception(e)

    if c2.button("❌ False Positive", key="btn_fp"):
        try:
            write_feedback(int(row_id), float(sel["fraud_probability"]), str(sel["risk_band"]), "FALSE_POSITIVE")
            st.success("Saved feedback: FALSE_POSITIVE")
        except Exception as e:
            st.error("Failed to save feedback.")
            st.exception(e)
