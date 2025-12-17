import sys
import os
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Paths (Streamlit Cloud safe)
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

# -----------------------------
# Helpers
# -----------------------------
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

pipe = load_model()

tab_score, tab_health, tab_feedback = st.tabs(["Score", "Model Health (Drift)", "Feedback"])

# session defaults
st.session_state.setdefault("latest_results", None)
st.session_state.setdefault("live_df_for_drift", None)

# -----------------------------
# SCORE TAB
# -----------------------------
with tab_score:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader_score")

    if uploaded is None:
        st.info("Upload a CSV to score.")
        st.stop()

    try:
        df_in = safe_read_csv(uploaded)

        # Optional label
        y_true = None
        if "Class" in df_in.columns:
            y_true = df_in["Class"].astype("int8").values
            df = df_in.drop(columns=["Class"])
            st.info("Detected `Class`. Evaluation metrics will be shown.")
        else:
            df = df_in

        # Schema enforce
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

        # safer types
        df = downcast_numeric(df)

        threshold = st.slider("Fraud threshold", 0.0, 1.0, 0.50, 0.01, key="threshold")

        # predict
        proba = pipe.predict_proba(df)[:, 1].astype("float32")

        results = df.copy()
        results["fraud_probability"] = proba
        results["fraud_pred"] = (proba >= threshold).astype("int8")
        results["risk_band"] = pd.cut(
            proba,
            bins=[-0.01, 0.2, 0.7, 1.01],
            labels=["LOW", "MEDIUM", "HIGH"],
        ).astype("category")

        # store for Drift/Feedback
        st.session_state["live_df_for_drift"] = df.copy()
        st.session_state["latest_results"] = results.copy()

        # metrics
        if y_true is not None:
            pred = results["fraud_pred"].values
            tp = int(((y_true == 1) & (pred == 1)).sum())
            fp = int(((y_true == 0) & (pred == 1)).sum())
            fn = int(((y_true == 1) & (pred == 0)).sum())
            tn = int(((y_true == 0) & (pred == 0)).sum())

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0

            st.subheader("Evaluation")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{precision:.3f}")
            c2.metric("Recall", f"{recall:.3f}")
            c3.metric("TP / FP", f"{tp} / {fp}")
            c4.metric("FN / TN", f"{fn} / {tn}")

        st.subheader("Scored Results (preview)")
        st.dataframe(results.head(50), use_container_width=True)

        st.subheader("Risk Band Counts")
        st.bar_chart(results["risk_band"].value_counts())

        st.download_button(
            "Download Scored CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv",
        )

        # SHAP (optional)
        st.subheader("Explainability (SHAP)")
        show_explain = st.checkbox("Compute SHAP explanations (Top-3 reasons)", value=False, key="shap_toggle")

        if show_explain:
            if not SHAP_BG_PATH.exists():
                st.warning(f"Missing SHAP background: `{SHAP_BG_PATH}`. Commit it or regenerate it.")
            else:
                bg = pd.read_csv(SHAP_BG_PATH)
                bg = bg[[c for c in df.columns if c in bg.columns]]

                explainer, scaler = make_explainer(pipe, bg)

                n_explain = st.slider("Rows to explain", 10, min(200, len(df)), 50, 10, key="n_explain")
                explain_df = df.head(n_explain).copy()

                sv = shap_values_for(pipe, explainer, scaler, explain_df)
                feature_names = list(explain_df.columns)

                reasons = []
                for i in range(min(n_explain, sv.shape[0])):
                    top3 = top_k_reasons(feature_names, sv[i], k=3)
                    reasons.append(", ".join([f"{f} ({v:+.3f})" for f, v in top3]))

                out = results.head(n_explain).copy()
                out["top_reasons"] = reasons
                st.dataframe(out[["fraud_probability", "risk_band", "top_reasons"]], use_container_width=True)

    except Exception as e:
        st.error("Scoring failed. Here is the real error:")
        st.exception(e)
        st.stop()

# -----------------------------
# DRIFT TAB
# -----------------------------
with tab_health:
    st.subheader("Drift Monitoring (PSI)")
    st.write("Compares uploaded data distribution vs a training baseline sample.")

    if not DRIFT_BASELINE_PATH.exists():
        st.warning(
            f"Missing drift baseline file: `{DRIFT_BASELINE_PATH}`.\n\n"
            "Fix: commit `data/processed/train_baseline_sample.csv` (it should be small)."
        )
        st.stop()

    if st.session_state.get("live_df_for_drift") is None:
        st.warning("No live data found. Go to Score tab and upload a CSV first.")
        st.stop()

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
    st.write(
        "Confirm fraud / mark false positives. "
        "On Streamlit Cloud, the filesystem is ephemeral (DB resets on restart)."
    )
    st.caption(f"DB path: {DB_PATH}")

    try:
        init_db()
    except Exception as e:
        st.error("Feedback DB init failed.")
        st.exception(e)
        st.stop()

    if st.session_state.get("latest_results") is None:
        st.warning("No scored results found. Go to Score tab and upload a CSV first.")
        st.stop()

    results_fb = st.session_state["latest_results"]
    st.dataframe(results_fb[["fraud_probability", "risk_band"]].head(20), use_container_width=True)

    max_idx = int(len(results_fb) - 1)
    row_id = st.number_input("Pick a row index to review", 0, max_idx, 0, 1, key="row_id")

    sel = results_fb.iloc[int(row_id)]
    st.write(
        {
            "row_index": int(row_id),
            "fraud_probability": float(sel["fraud_probability"]),
            "risk_band": str(sel["risk_band"]),
        }
    )

    # Use a form so a click doesn't cause weird rerun side effects
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
