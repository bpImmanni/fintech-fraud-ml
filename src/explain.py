import numpy as np
import pandas as pd
import shap

def make_explainer(pipe, background_df: pd.DataFrame):
    # pipeline: scaler -> model (xgboost)
    model = pipe.named_steps["model"]
    scaler = pipe.named_steps["scaler"]

    X_bg = scaler.transform(background_df)
    explainer = shap.TreeExplainer(model)
    return explainer, scaler

def shap_values_for(pipe, explainer, scaler, X: pd.DataFrame):
    Xs = scaler.transform(X)
    sv = explainer.shap_values(Xs)
    # sv shape: (n_samples, n_features)
    return np.array(sv)

def top_k_reasons(feature_names, shap_row, k=3):
    idx = np.argsort(np.abs(shap_row))[::-1][:k]
    return [(feature_names[i], float(shap_row[i])) for i in idx]
