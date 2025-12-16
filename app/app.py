import sys
import os

# Add project root to PYTHONPATH so `src` is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.explain import make_explainer, shap_values_for, top_k_reasons
from src.drift import drift_report
from src.feedback import init_db, write_feedback

MODEL_PATH = "models/fraud_model.pkl"
SCHEMA_PATH = "reports/schema.json"
SHAP_BG_PATH = "data/processed/shap_background.csv"
DRIFT_BASELINE_PATH = "data/processed/train_baseline_sample.csv"

st.set_page_config(page_title="FinTech Fraud Detector", layout="wide")
st.title("FinTech Fraud Detector")
st.write("Upload transactions CSV. If your file includes `Class`, the app will also show evaluation metrics.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

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
        df_in = pd.read_csv(uploaded)

        # Optional labels
        y_true = None
        if "Class" in df_in.columns:
            y_true = df_in["Class"].astype(int).values
            df = df_in.drop(columns=["Class"])
            st.info("Detected `Class` column. Evaluation metrics will be shown.")
        else:
            df = df_in.copy()

        # Schema validation + enforce order
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema = json.load(f)
            expected_cols = schema.get("columns", [])

            if expected_cols:
                missing = [c for c in expected_cols if c not in df.columns]
                extra = [c for c in df.columns if c not in expected_cols]

                if missing:
                    st.error(f"Missing columns (showing first 15): {missing[:15]}")
                    st.stop()

                if extra:
                    st.warning(f"Extra columns ignored (showing first 15): {extra[:15]}")

                df = df[expected_cols]  # enforce column order

        threshold = st.slider("Fraud threshold", 0.0, 1.0, 0.50, 0.01, key="threshold")

        proba = pipe.predict_proba(df)[:, 1]

        results = df.copy()
        results["fraud_probability"] = proba
        results["fraud_pred"] = (results["fraud_probability"] >= threshold).astype(int)
        results["risk_band"] = pd.cut(
            results["fraud_probability"],
            bins=[-0.01, 0.2, 0.7, 1.01],
            labels=["LOW", "MEDIUM", "HIGH"]
        )

        # Save session state IMMEDIATELY (fixes your issue)
        st.session_state["live_df_for_drift"] = df.copy()
        st.session_state["latest_results"] = results.copy()
        st.session_state["has_scored"] = True

        # Evaluation metrics if labels exist
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

        # SHAP explainability
        st.subheader("Explainability (SHAP)")
        show_explain = st.checkbox("Compute SHAP explanations (Top-3 reasons)", value=False, key="shap_toggle")

        if show_explain:
            if not os.path.exists(SHAP_BG_PATH):
                st.warning(f"Missing SHAP background file: `{SHAP_BG_PATH}`. Re-run `python src/train.py`.")
            else:
                bg = pd.read_csv(SHAP_BG_PATH)
                bg = bg[[c for c in df.columns if c in bg.columns]]

                explainer, scaler = make_explainer(pipe, bg)

                n_explain = st.slider("Rows to explain (for speed)", 10, 200, 50, 10, key="n_explain")
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
                    use_container_width=True
                )

        st.subheader("Scored Results (preview)")
        st.dataframe(results.head(50), use_container_width=True)

        st.subheader("Risk Band Counts")
        st.bar_chart(results["risk_band"].value_counts())

        st.download_button(
            "Download Scored CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv"
        )


# -----------------------------
# DRIFT TAB
# -----------------------------
with tab_health:
    st.subheader("Drift Monitoring (PSI)")
    st.write("Compares uploaded data distribution vs a training baseline sample.")

    if not os.path.exists(DRIFT_BASELINE_PATH):
        st.warning(f"Missing drift baseline file: `{DRIFT_BASELINE_PATH}`. Re-run `python src/train.py`.")
    elif "live_df_for_drift" not in st.session_state:
        st.warning("No live data found yet. Go to Score tab, upload a CSV, then return here.")
    else:
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


# -----------------------------
# FEEDBACK TAB
# -----------------------------
with tab_feedback:
    st.subheader("Analyst Feedback Loop (SQLite)")
    st.write(
        "Confirm fraud / mark false positives. "
        "On cloud (Render), the DB must be stored in a writable path (usually `/tmp`)."
    )

    # Don't let DB init crash the whole app on Render
    try:
        init_db()
    except Exception as e:
        st.error("Feedback storage is not available on this deployment.")
        st.caption(f"Reason: {e}")
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
        write_feedback(int(row_id), float(sel["fraud_probability"]), str(sel["risk_band"]), "CONFIRM_FRAUD")
        st.success("Saved feedback: CONFIRM_FRAUD")

    if c2.button("❌ False Positive", key="btn_fp"):
        write_feedback(int(row_id), float(sel["fraud_probability"]), str(sel["risk_band"]), "FALSE_POSITIVE")
        st.success("Saved feedback: FALSE_POSITIVE")
