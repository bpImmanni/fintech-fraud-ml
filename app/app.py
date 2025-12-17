import sys
import os
from pathlib import Path
import json
import hashlib
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

MAX_ROWS_SCORE = 10_000 if IS_RENDER else 200_000       # max rows to score per run
MAX_ROWS_STATE = 2_000 if IS_RENDER else 20_000         # max rows stored in session_state
MAX_ROWS_SHAP = 100 if IS_RENDER else 200               # SHAP cap (optional)

# -----------------------------
# Paths (absolute, Render-safe)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # project root
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCHEMA_PATH = BASE_DIR / "reports" / "schema.json"
SHAP_BG_PATH = BASE_DIR / "data" / "processed" / "shap_background.csv"
DRIFT_BASELINE_PATH = BASE_DIR / "data" / "processed" / "train_baseline_sample.csv"

# -----------------------------
# Imports from src/
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
    st.caption("Running on Render: scoring is manual + download is on-demand to prevent memory restarts.")

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

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype("float32")
            elif pd.api.types.is_integer_dtype(df[c]):
                df[c] = df[c].astype("int32")
    return df

def file_fingerprint(uploaded) -> str:
    """
    Stable fingerprint without reading full file into memory.
    Uses name + size only (good enough to detect a new upload).
    """
    name = getattr(uploaded, "name", "file")
    size = getattr(uploaded, "size", 0)
    return f"{name}::{size}"

pipe = load_model()

tab_score, tab_health, tab_feedback = st.tabs(["Score", "Model Health (Drift)", "Feedback"])

# Initialize session keys
st.session_state.setdefault("last_file_fp", None)
st.session_state.setdefault("has_scored", False)
st.session_state.setdefault("latest_results", None)
st.session_state.setdefault("live_df_for_drift", None)
st.session_state.setdefault("download_bytes", None)

# -----------------------------
# SCORE TAB
# -----------------------------
with tab_score:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader_score")

    if uploaded is None:
        st.info("Upload a CSV, then click **Run Scoring**.")
    else:
        fp = file_fingerprint(uploaded)

        # If user uploaded a different file, reset cached scoring artifacts
        if st.session_state["last_file_fp"] != fp:
            st.session_state["last_file_fp"] = fp
            st.session_state["has_scored"] = False
            st.session_state["latest_results"] = None
            st.session_state["live_df_for_drift"] = None
            st.session_state["download_bytes"] = None
            st.toast("New file detected. Please click Run Scoring.", icon="🆕")

        threshold = st.slider("Fraud threshold", 0.0, 1.0, 0.50, 0.01, key="threshold")

        run = st.button("▶️ Run Scoring", type="primary")

        if run:
            try:
                df_in = safe_read_csv(uploaded)

                # Optional labels
                y_true = None
                if "Class" in df_in.columns:
                    y_true = df_in["Class"].astype("int8").values
                    df = df_in.drop(columns=["Class"])
                    st.info("Detected `Class`. Evaluation metrics will be shown.")
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

                        df = df[expected_cols]

                # Cap rows before heavy work
                if len(df) > MAX_ROWS_SCORE:
                    st.warning(f"Large file ({len(df)} rows). Scoring first {MAX_ROWS_SCORE} rows.")
                    df = df.head(MAX_ROWS_SCORE)
                    if y_true is not None:
                        y_true = y_true[:MAX_ROWS_SCORE]

                df = downcast_numeric(df)

                proba = pipe.predict_proba(df)[:, 1].astype("float32")

                results = pd.DataFrame(index=df.index)
                results["fraud_probability"] = proba
                results["fraud_pred"] = (proba >= threshold).astype("int8")
                results["risk_band"] = pd.cut(
                    proba,
                    bins=[-0.01, 0.2, 0.7, 1.01],
                    labels=["LOW", "MEDIUM", "HIGH"],
                ).astype("category")

                # Metrics if labels exist
                if y_true is not None:
                    results["Class"] = y_true
                    tp = int(((y_true == 1) & (results["fraud_pred"].values == 1)).sum())
                    fp_ = int(((y_true == 0) & (results["fraud_pred"].values == 1)).sum())
                    fn = int(((y_true == 1) & (results["fraud_pred"].values == 0)).sum())
                    tn = int(((y_true == 0) & (results["fraud_pred"].values == 0)).sum())
                    precision = tp / (tp + fp_) if (tp + fp_) else 0.0
                    recall = tp / (tp + fn) if (tp + fn) else 0.0

                    st.subheader("Evaluation")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Precision", f"{precision:.3f}")
                    c2.metric("Recall", f"{recall:.3f}")
                    c3.metric("TP / FP", f"{tp} / {fp_}")
                    c4.metric("FN / TN", f"{fn} / {tn}")

                # Store SMALL samples only (prevents Render restarts on clicks)
                st.session_state["live_df_for_drift"] = df.head(MAX_ROWS_STATE).copy()
                st.session_state["latest_results"] = results.head(MAX_ROWS_STATE).copy()
                st.session_state["has_scored"] = True
                st.session_state["download_bytes"] = None  # reset

                st.success("Scoring completed ✅ (stored a small sample for Drift + Feedback)")

            except Exception as e:
                st.error("Scoring failed (this prevents 502 by showing the real error).")
                st.exception(e)
                st.stop()

        # Show preview if already scored (NO recompute)
        if st.session_state["has_scored"] and st.session_state["latest_results"] is not None:
            results_preview = st.session_state["latest_results"]
            st.subheader("Scored Results (preview)")
            st.dataframe(results_preview.head(50), use_container_width=True)

            st.subheader("Risk Band Counts")
            st.bar_chart(results_preview["risk_band"].value_counts())

            # On-demand download bytes to avoid memory spikes on every rerun
            if st.button("📦 Prepare download file (on-demand)"):
                try:
                    # Re-read and score again ONLY for download export (still capped)
                    uploaded.seek(0)
                    df_in = safe_read_csv(uploaded)

                    y_true = None
                    if "Class" in df_in.columns:
                        y_true = df_in["Class"].astype("int8").values
                        df = df_in.drop(columns=["Class"])
                    else:
                        df = df_in

                    if SCHEMA_PATH.exists():
                        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                            schema = json.load(f)
                        expected_cols = schema.get("columns", [])
                        if expected_cols:
                            df = df[expected_cols]

                    if len(df) > MAX_ROWS_SCORE:
                        df = df.head(MAX_ROWS_SCORE)
                        if y_true is not None:
                            y_true = y_true[:MAX_ROWS_SCORE]

                    df = downcast_numeric(df)
                    proba = pipe.predict_proba(df)[:, 1].astype("float32")

                    results = pd.DataFrame(index=df.index)
                    results["fraud_probability"] = proba
                    results["fraud_pred"] = (proba >= st.session_state["threshold"]).astype("int8")
                    results["risk_band"] = pd.cut(
                        proba, bins=[-0.01, 0.2, 0.7, 1.01], labels=["LOW", "MEDIUM", "HIGH"]
                    ).astype("category")

                    if y_true is not None:
                        results["Class"] = y_true

                    scored_out = pd.concat([df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
                    st.session_state["download_bytes"] = scored_out.to_csv(index=False).encode("utf-8")
                    st.success("Download file prepared ✅")
                except Exception as e:
                    st.error("Failed to prepare download.")
                    st.exception(e)

            if st.session_state["download_bytes"] is not None:
                st.download_button(
                    "⬇️ Download Scored CSV",
                    data=st.session_state["download_bytes"],
                    file_name="scored_transactions.csv",
                    mime="text/csv",
                )

            # SHAP optional (keep off by default on Render)
            st.subheader("Explainability (SHAP)")
            show_explain = st.checkbox("Compute SHAP explanations (Top-3 reasons)", value=False, key="shap_toggle")
            if show_explain:
                if not SHAP_BG_PATH.exists():
                    st.warning(f"Missing SHAP background: `{SHAP_BG_PATH}` (commit it or re-run training).")
                else:
                    try:
                        uploaded.seek(0)
                        df_in = safe_read_csv(uploaded)
                        if "Class" in df_in.columns:
                            df = df_in.drop(columns=["Class"])
                        else:
                            df = df_in

                        if SCHEMA_PATH.exists():
                            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                                schema = json.load(f)
                            expected_cols = schema.get("columns", [])
                            if expected_cols:
                                df = df[expected_cols]

                        df = df.head(min(MAX_ROWS_SHAP, len(df)))
                        df = downcast_numeric(df)

                        bg = pd.read_csv(SHAP_BG_PATH)
                        bg = bg[[c for c in df.columns if c in bg.columns]]

                        explainer, scaler = make_explainer(pipe, bg)
                        sv = shap_values_for(pipe, explainer, scaler, df)
                        feature_names = list(df.columns)

                        reasons = []
                        for i in range(min(len(df), sv.shape[0])):
                            top3 = top_k_reasons(feature_names, sv[i], k=3)
                            reasons.append(", ".join([f"{f} ({v:+.3f})" for f, v in top3]))

                        tmp = pd.DataFrame({"top_reasons": reasons})
                        st.dataframe(tmp.head(50), use_container_width=True)

                    except Exception as e:
                        st.error("SHAP failed (disabled to prevent 502).")
                        st.exception(e)

# -----------------------------
# DRIFT TAB
# -----------------------------
with tab_health:
    st.subheader("Drift Monitoring (PSI)")
    st.write("Compares uploaded data distribution vs a training baseline sample.")

    if not DRIFT_BASELINE_PATH.exists():
        st.warning(f"Missing drift baseline file: `{DRIFT_BASELINE_PATH}`. Commit it (small) or disable Drift.")
    elif st.session_state.get("live_df_for_drift") is None:
        st.warning("No live data found. Go to Score tab and click Run Scoring.")
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

# -----------------------------
# FEEDBACK TAB
# -----------------------------
with tab_feedback:
    st.subheader("Analyst Feedback Loop (SQLite)")
    st.write("Confirm fraud / mark false positives. On Render, DB must be in a writable path (we use /tmp).")
    st.caption(f"DB path: {DB_PATH}")

    try:
        init_db()
    except Exception as e:
        st.error(f"Feedback DB init failed: {e}")
        st.exception(e)
        st.stop()

    if st.session_state.get("latest_results") is None:
        st.warning("No scored results found. Go to Score tab and click Run Scoring.")
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
        key="row_id",
    )

    sel = results_fb.iloc[int(row_id)]
    st.write(
        {
            "row_index": int(row_id),
            "fraud_probability": float(sel["fraud_probability"]),
            "risk_band": str(sel["risk_band"]),
        }
    )

    # Use a form so the click doesn't trigger extra UI events
    with st.form("feedback_form"):
        c1, c2 = st.columns(2)
        confirm = c1.form_submit_button("✅ Confirm Fraud")
        fp = c2.form_submit_button("❌ False Positive")

        if confirm or fp:
            action = "CONFIRM_FRAUD" if confirm else "FALSE_POSITIVE"
            try:
                write_feedback(int(row_id), float(sel["fraud_probability"]), str(sel["risk_band"]), action)
                st.success(f"Saved feedback: {action}")
            except Exception as e:
                st.error("Failed to save feedback.")
                st.exception(e)
