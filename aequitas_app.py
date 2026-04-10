"""
Aequitas: The Algorithmic Bias & Fairness Auditor
A professional-grade tool for detecting, visualizing, and explaining systemic bias in ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings
import io
import base64
from datetime import datetime
import json

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aequitas | Bias Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=Exo+2:wght@300;400;600;700&display=swap');

/* ── Root & Background ── */
:root {
    --bg-void:    #050810;
    --bg-panel:   #0b0f1a;
    --bg-card:    #0f1521;
    --bg-hover:   #141c2e;
    --cyan:       #00d4ff;
    --cyan-dim:   #0096b3;
    --green:      #00ff88;
    --green-dim:  #00b35f;
    --red:        #ff3366;
    --red-dim:    #b3234a;
    --amber:      #ffaa00;
    --purple:     #8b5cf6;
    --text-prime: #e2e8f4;
    --text-sec:   #7a8ba8;
    --text-dim:   #3d4f6b;
    --border:     #1a2540;
    --border-glow:#00d4ff22;
    --font-mono:  'Share Tech Mono', monospace;
    --font-head:  'Rajdhani', sans-serif;
    --font-body:  'Exo 2', sans-serif;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    background-color: var(--bg-void) !important;
    color: var(--text-prime) !important;
    font-family: var(--font-body) !important;
}

.stApp { background: var(--bg-void) !important; }

/* ── Scanline overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,212,255,0.015) 2px,
        rgba(0,212,255,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Headers ── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: var(--font-head) !important;
    letter-spacing: 0.05em !important;
    color: var(--text-prime) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-body) !important; }

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: var(--text-sec) !important; font-family: var(--font-mono) !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: var(--cyan) !important; font-family: var(--font-head) !important; font-size: 2rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    border-radius: 2px !important;
}
.stButton > button:hover {
    background: var(--cyan) !important;
    color: var(--bg-void) !important;
    box-shadow: 0 0 20px var(--cyan-dim) !important;
}

/* ── Selectbox / Inputs ── */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-prime) !important;
}
.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-prime) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    font-family: var(--font-mono) !important;
    color: var(--cyan) !important;
    font-size: 12px !important;
}
.streamlit-expanderContent {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── DataFrame ── */
.stDataFrame { background: var(--bg-card) !important; }
iframe { background: var(--bg-card) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    color: var(--text-sec) !important;
    background: transparent !important;
    border-radius: 2px 2px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}

/* ── Toggle ── */
.stToggle > label { color: var(--text-sec) !important; font-family: var(--font-mono) !important; }

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--cyan-dim) !important;
    border-radius: 4px !important;
    padding: 8px !important;
}

/* ── Custom Components ── */
.aq-header {
    background: linear-gradient(135deg, #050810 0%, #0b1628 50%, #050810 100%);
    border-bottom: 1px solid var(--border);
    padding: 24px 0 20px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.aq-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--green), var(--cyan), transparent);
}
.aq-title {
    font-family: var(--font-head);
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    background: linear-gradient(135deg, #ffffff 30%, var(--cyan) 70%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
.aq-subtitle {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-sec);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 6px;
}
.aq-badge {
    display: inline-block;
    background: var(--border-glow);
    border: 1px solid var(--cyan-dim);
    color: var(--cyan);
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 0.15em;
    padding: 3px 10px;
    border-radius: 2px;
    margin-right: 8px;
    margin-top: 10px;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
}
.metric-card.ok::before   { background: var(--green); }
.metric-card.warn::before { background: var(--amber); }
.metric-card.fail::before { background: var(--red); }
.metric-card.info::before { background: var(--cyan); }

.metric-label {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-sec);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: var(--font-head);
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-value.ok   { color: var(--green); }
.metric-value.warn { color: var(--amber); }
.metric-value.fail { color: var(--red); }
.metric-value.info { color: var(--cyan); }

.metric-desc {
    font-size: 12px;
    color: var(--text-sec);
    margin-top: 8px;
    font-family: var(--font-body);
}
.flag-banner {
    background: linear-gradient(90deg, rgba(255,51,102,0.15), rgba(255,51,102,0.05));
    border: 1px solid var(--red-dim);
    border-radius: 4px;
    padding: 12px 16px;
    margin: 12px 0;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--red);
    display: flex;
    align-items: center;
    gap: 10px;
}
.pass-banner {
    background: linear-gradient(90deg, rgba(0,255,136,0.1), rgba(0,255,136,0.03));
    border: 1px solid var(--green-dim);
    border-radius: 4px;
    padding: 12px 16px;
    margin: 12px 0;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--green);
}
.info-box {
    background: rgba(0,212,255,0.05);
    border: 1px solid var(--cyan-dim);
    border-radius: 4px;
    padding: 14px 16px;
    margin: 10px 0;
    font-size: 13px;
    color: var(--text-sec);
    line-height: 1.6;
}
.info-box strong { color: var(--cyan); }
.proxy-card {
    background: rgba(255,170,0,0.08);
    border: 1px solid var(--amber);
    border-radius: 4px;
    padding: 14px 16px;
    margin: 10px 0;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--amber);
}
.section-header {
    font-family: var(--font-head);
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--text-prime);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 28px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header .num {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--cyan);
    background: rgba(0,212,255,0.1);
    border: 1px solid var(--cyan-dim);
    padding: 2px 8px;
    border-radius: 2px;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-dot.green { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-dot.red   { background: var(--red);   box-shadow: 0 0 6px var(--red);   }
.status-dot.amber { background: var(--amber);  box-shadow: 0 0 6px var(--amber);  }

/* Sidebar section headers */
.sb-section {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--cyan);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 4px 0;
    border-bottom: 1px solid var(--border);
    margin: 16px 0 10px;
}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0b0f1a",
    plot_bgcolor="#0b0f1a",
    font=dict(family="Share Tech Mono, monospace", color="#7a8ba8", size=11),
    title_font=dict(family="Rajdhani, sans-serif", color="#e2e8f4", size=14),
    xaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540", tickfont=dict(color="#7a8ba8")),
    yaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540", tickfont=dict(color="#7a8ba8")),
    legend=dict(bgcolor="#0f1521", bordercolor="#1a2540", borderwidth=1),
    margin=dict(l=20, r=20, t=50, b=20),
    colorway=["#00d4ff", "#00ff88", "#ff3366", "#ffaa00", "#8b5cf6"],
)

CYAN, GREEN, RED, AMBER, PURPLE = "#00d4ff", "#00ff88", "#ff3366", "#ffaa00", "#8b5cf6"

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def info_expander(label: str, content: str):
    with st.expander(f"ℹ️  {label}"):
        st.markdown(f'<div class="info-box">{content}</div>', unsafe_allow_html=True)

def status_card(label, value, status, desc=""):
    color_cls = {"ok": "ok", "warn": "warn", "fail": "fail", "info": "info"}.get(status, "info")
    st.markdown(f"""
    <div class="metric-card {color_cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_cls}">{value}</div>
        {"<div class='metric-desc'>" + desc + "</div>" if desc else ""}
    </div>""", unsafe_allow_html=True)

def compute_fairness_metrics(df, protected_col, target_col, privileged_val, predictions=None):
    """Compute core fairness metrics."""
    series = predictions if predictions is not None else df[target_col]
    priv_mask = df[protected_col] == privileged_val
    unpriv_mask = ~priv_mask

    priv_sel  = series[priv_mask].mean()
    unpriv_sel = series[unpriv_mask].mean()
    dp_diff   = unpriv_sel - priv_sel
    di_ratio  = unpriv_sel / priv_sel if priv_sel > 0 else 0

    # Equalized Odds — needs ground truth
    tpr_priv = tpr_unpriv = None
    if predictions is not None:
        y_true = df[target_col]
        tp_priv   = ((predictions == 1) & (y_true == 1) & priv_mask).sum()
        fn_priv   = ((predictions == 0) & (y_true == 1) & priv_mask).sum()
        tp_unpriv = ((predictions == 1) & (y_true == 1) & unpriv_mask).sum()
        fn_unpriv = ((predictions == 0) & (y_true == 1) & unpriv_mask).sum()
        tpr_priv   = tp_priv   / (tp_priv   + fn_priv)   if (tp_priv   + fn_priv)   > 0 else 0
        tpr_unpriv = tp_unpriv / (tp_unpriv + fn_unpriv) if (tp_unpriv + fn_unpriv) > 0 else 0

    return dict(
        priv_selection_rate=priv_sel,
        unpriv_selection_rate=unpriv_sel,
        dp_difference=dp_diff,
        di_ratio=di_ratio,
        tpr_privileged=tpr_priv,
        tpr_unprivileged=tpr_unpriv,
    )

def reweight_data(df, protected_col, target_col, privileged_val):
    """Compute sample weights for fairness re-weighting."""
    n = len(df)
    priv_mask  = df[protected_col] == privileged_val
    unpriv_mask = ~priv_mask
    pos_mask   = df[target_col] == 1
    neg_mask   = ~pos_mask

    n_priv     = priv_mask.sum()
    n_unpriv   = unpriv_mask.sum()
    n_pos      = pos_mask.sum()
    n_neg      = neg_mask.sum()

    weights = np.ones(n)
    for i in df.index:
        is_priv = priv_mask[i]
        is_pos  = pos_mask[i]
        n_g     = n_priv if is_priv else n_unpriv
        n_y     = n_pos  if is_pos  else n_neg
        n_gy    = ((priv_mask if is_priv else unpriv_mask) & (pos_mask if is_pos else neg_mask)).sum()
        weights[df.index.get_loc(i)] = (n_g * n_y) / (n * n_gy) if n_gy > 0 else 1
    return weights

def train_model(df, protected_col, target_col, sample_weight=None):
    """Force-encodes ALL non-numeric data to ensure sklearn never sees a string."""
    df_enc = df.copy()
    encoders = {}
    feature_cols = [c for c in df.columns if c != target_col]

    def force_encode(dataframe, column_name):
        col_data = dataframe[column_name]
        # SAFE CHECK: If it's not float or int, it's a string/category for our purposes
        if not pd.api.types.is_numeric_dtype(col_data):
            le = LabelEncoder()
            # Convert to string to handle the <StringDtype> and NaNs
            dataframe[column_name] = le.fit_transform(col_data.astype(str))
            return le
        return None

    # 1. Encode every feature that isn't a number
    for col in feature_cols:
        enc = force_encode(df_enc, col)
        if enc:
            encoders[col] = enc

    # 2. Encode the target if it isn't a number
    enc_target = force_encode(df_enc, target_col)
    if enc_target:
        encoders[target_col] = enc_target

    # 3. Prepare X and y
    # We convert to .values to strip any remaining Pandas-specific dtypes
    X = df_enc[feature_cols].fillna(-1).values.astype(float)
    y = df_enc[target_col].values.astype(float)

    if len(np.unique(y)) < 2:
        return None, encoders, None, feature_cols

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6, n_jobs=-1)
    
    w = sample_weight if sample_weight is not None else None
    model.fit(X, y, sample_weight=w)
    preds = model.predict(X)

    return model, encoders, preds, df_enc[feature_cols]

def compute_shap(model, X_enc, feature_names, max_samples=100):
    """Safe SHAP computation with background sampling."""
    try:
        # 1. Force X_enc to be numeric (float) before sampling
        # This prevents the 'Male' error in the SHAP tab
        X_numeric = X_enc.apply(pd.to_numeric, errors='coerce').fillna(-1)
        
        if len(X_numeric) > max_samples:
            X_sample = shap.sample(X_numeric, max_samples)
        else:
            X_sample = X_numeric
            
        explainer = shap.TreeExplainer(model)
        
        # 2. Convert to values to strip any remaining weird Pandas types
        shap_values = explainer.shap_values(X_sample.values)
        
        if isinstance(shap_values, list):
            return shap_values[1], X_sample
        return shap_values, X_sample
    except Exception as e:
        st.warning(f"SHAP Analysis skipped: {e}")
        return None, None

def correlation_with_protected(df, protected_col, feature_names):
    """Compute Cramér's V between each feature and the protected attribute."""
    df_enc = df.copy()
    corrs = {}
    for col in feature_names:
        if col == protected_col:
            continue
        try:
            ct = pd.crosstab(df_enc[col].astype(str), df_enc[protected_col].astype(str))
            from scipy.stats import chi2_contingency
            chi2, _, _, _ = chi2_contingency(ct)
            n = ct.sum().sum()
            k = min(ct.shape) - 1
            v = np.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0
            corrs[col] = v
        except Exception:
            corrs[col] = 0
    return corrs

def generate_markdown_report(cfg, metrics, metrics_rw, proxies, di_pass):
    """Generate a full Markdown audit report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    priv  = cfg["privileged_val"]
    unpriv_label = "Non-privileged"
    di    = metrics["di_ratio"]
    di_rw = metrics_rw["di_ratio"] if metrics_rw else None

    status = "⛔ FAILED" if not di_pass else "✅ PASSED"
    proxy_section = ""
    if proxies:
        for p in proxies:
            proxy_section += f"- **{p['feature']}** — Correlation with protected attribute: `{p['corr']:.3f}`, SHAP importance rank: `#{p['rank']}`\n"
    else:
        proxy_section = "_No high-risk proxies detected._\n"

    mitigation_section = ""
    if metrics_rw:
        di_delta = metrics_rw["di_ratio"] - metrics["di_ratio"]
        mitigation_section = f"""
## 5. Mitigation Analysis (Re-weighting)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Disparate Impact Ratio | {metrics['di_ratio']:.4f} | {metrics_rw['di_ratio']:.4f} | {di_delta:+.4f} |
| DP Difference | {metrics['dp_difference']:.4f} | {metrics_rw['dp_difference']:.4f} | {metrics_rw['dp_difference'] - metrics['dp_difference']:+.4f} |

**Finding:** Re-weighting {'improved' if di_delta > 0 else 'did not improve'} the Disparate Impact Ratio by `{abs(di_delta):.4f}`.
"""

    return f"""# AEQUITAS ALGORITHMIC BIAS AUDIT REPORT
**Generated:** {now}  
**Dataset:** `{cfg.get('filename', 'Uploaded Dataset')}`  
**Protected Attribute:** `{cfg['protected_col']}`  
**Target Variable:** `{cfg['target_col']}`  
**Privileged Group:** `{priv}`  

---

## Executive Summary

This automated audit assessed `{cfg['protected_col']}` as a protected attribute against the outcome `{cfg['target_col']}`. The overall audit status is **{status}**.

---

## 1. Fairness Metrics

### Demographic Parity
- **Privileged Selection Rate:** `{metrics['priv_selection_rate']:.4f}` ({metrics['priv_selection_rate']*100:.1f}%)
- **Unprivileged Selection Rate:** `{metrics['unpriv_selection_rate']:.4f}` ({metrics['unpriv_selection_rate']*100:.1f}%)
- **Difference:** `{metrics['dp_difference']:+.4f}`

> A difference close to 0 indicates demographic parity.

### Disparate Impact Ratio (80% Rule — EEOC Standard)
- **Ratio:** `{di:.4f}`
- **Threshold:** `0.80`
- **Status:** {status}

> If the ratio is below 0.80, the model is at significant legal risk under employment discrimination law in many jurisdictions.

### Equalized Odds (True Positive Rate)
{f"- **Privileged TPR:** `{metrics['tpr_privileged']:.4f}`" if metrics.get('tpr_privileged') is not None else "_Requires model predictions._"}
{f"- **Unprivileged TPR:** `{metrics['tpr_unprivileged']:.4f}`" if metrics.get('tpr_unprivileged') is not None else ""}

---

## 2. Potential Bias Proxies Detected

{proxy_section}

---

## 3. Legal & Regulatory Implications

### EU AI Act (Regulation (EU) 2024/1689)
- **High-Risk Classification:** AI systems used in hiring, credit, education, and law enforcement are classified as **High-Risk** under Annex III.
- **Article 10:** High-risk AI training data must be examined for possible biases.
- **Article 13:** High-risk AI systems must be transparent and provide sufficient information for effective oversight.
- **Article 15:** High-risk AI systems must achieve appropriate levels of accuracy, robustness, and cybersecurity.

### US Legal Exposure
- **Title VII, Civil Rights Act (1964):** Prohibits employment discrimination based on race, color, religion, sex, or national origin.
- **EEOC Uniform Guidelines (1978):** The 80% (4/5) Rule is the primary legal standard for disparate impact in employment.
- **Fair Housing Act / Equal Credit Opportunity Act:** Apply to housing and credit decisions.

---

## 4. Developer Recommendations

1. **Audit your training data** for historical bias. If protected attributes correlate with outcomes in training data, the model will learn and perpetuate that discrimination.
2. **Remove or transform proxy features** identified above before training. Consider using fairness-aware feature selection.
3. **Apply pre-processing mitigation** (re-weighting, resampling) or **in-processing** (fairness constraints via Fairlearn's `ExponentiatedGradient`).
4. **Implement continuous monitoring** — track fairness metrics in production, not just at training time.
5. **Document your bias mitigation process** to comply with EU AI Act Article 13 transparency requirements.
6. **Involve domain experts** — particularly people from affected groups — in the design and evaluation process.

{mitigation_section}

---

## 6. Methodology

| Component | Method |
|-----------|--------|
| Bias Detection | Demographic Parity, Disparate Impact (EEOC 80% Rule), Equalized Odds |
| Model | Random Forest (scikit-learn, 100 estimators, max_depth=6) |
| Explainability | SHAP TreeExplainer (Shapley Additive Explanations) |
| Proxy Detection | Cramér's V correlation + SHAP feature importance |
| Mitigation | Calmon et al. (2017) re-weighting algorithm |

---

_Report generated by **Aequitas: The Algorithmic Bias & Fairness Auditor**._  
_This report is for informational purposes. Consult qualified legal counsel for compliance advice._
"""

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="aq-header">
    <p class="aq-title">⚖ AEQUITAS</p>
    <p class="aq-subtitle">Algorithmic Bias &amp; Fairness Auditor · v2.0 (Stable)</p>
    <span class="aq-badge">EU AI ACT COMPLIANT</span>
    <span class="aq-badge">EEOC STANDARD</span>
    <span class="aq-badge">SHAP EXPLAINER</span>
</div>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "preds" not in st.session_state:
    st.session_state.preds = None
if "shap_values" not in st.session_state:
    st.session_state.shap_values = None
if "X_enc" not in st.session_state:
    st.session_state.X_enc = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "report_md" not in st.session_state:
    st.session_state.report_md = None

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-section">// SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="status-dot green"></span><span style="font-family:var(--font-mono);font-size:11px;color:#7a8ba8">AUDITOR ONLINE</span>', unsafe_allow_html=True)
    st.markdown(f'<br><span style="font-family:var(--font-mono);font-size:10px;color:#3d4f6b">{datetime.now().strftime("%Y-%m-%d %H:%M")}</span>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">// CONFIGURATION</div>', unsafe_allow_html=True)

    protected_col = None
    target_col    = None
    privileged_val = None
    run_audit      = False

    if st.session_state.df is not None:
        df = st.session_state.df
        cols = df.columns.tolist()

        protected_col = st.selectbox("Protected Attribute", cols,
            help="The column representing a sensitive characteristic (e.g., Race, Gender).")

        remaining = [c for c in cols if c != protected_col]
        target_col = st.selectbox("Target / Outcome Column", remaining,
            help="The column representing the ML decision outcome (e.g., Loan_Approved).")

        if protected_col and protected_col in df.columns:
            unique_vals = df[protected_col].dropna().unique().tolist()
            if len(unique_vals) <= 30:
                privileged_val = st.selectbox("Privileged Group Value",
                    unique_vals,
                    help="Which value in the protected column represents the historically advantaged group?")
            else:
                privileged_val = st.text_input("Privileged Group Value (type it)",
                    help="Enter the exact string value for the privileged group.")

        st.markdown('<div class="sb-section">// ACTIONS</div>', unsafe_allow_html=True)
        run_audit = st.button("▶  RUN FULL AUDIT", use_container_width=True)
    else:
        st.markdown('<div style="font-family:var(--font-mono);font-size:11px;color:#3d4f6b;margin-top:12px">Upload a dataset to begin configuration.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">// LEGAL REFS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:var(--font-mono);font-size:10px;color:#3d4f6b;line-height:1.8">
    EU AI Act 2024/1689<br>
    EEOC Uniform Guidelines<br>
    Title VII Civil Rights Act<br>
    Fair Housing Act<br>
    Equal Credit Opp. Act
    </div>""", unsafe_allow_html=True)


# ─── MAIN TABS ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "|DATA INGESTION|",
    "|FAIRNESS DASHBOARD|",
    "|SHAP EXPLAINER|",
    "|MITIGATION SANDBOX|",
    "|AUDIT REPORT|",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA INGESTION
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header"><span class="num">01</span>DATA INGESTION HUB</div>', unsafe_allow_html=True)

    col_up, col_demo = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drag & drop a CSV or XLSX file",
            type=["csv", "xlsx"],
            help="Upload your dataset. It should contain the outcome column and at least one protected attribute."
        )
        if uploaded:
            try:
                if uploaded.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)
                st.session_state.df = df
                st.session_state.filename = uploaded.name
                # Reset derived state
                for k in ["model","preds","shap_values","X_enc","feature_cols","report_md"]:
                    st.session_state[k] = None
                st.markdown('<div class="pass-banner">✅  Dataset loaded successfully.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="flag-banner">⛔ Error loading file: {e}</div>', unsafe_allow_html=True)

    with col_demo:
        st.markdown('<div style="font-family:var(--font-mono);font-size:11px;color:#3d4f6b;margin-bottom:12px">// NO DATA? USE DEMO</div>', unsafe_allow_html=True)
        if st.button("Load Demo Dataset", use_container_width=True):
            np.random.seed(42)
            n = 500
            gender = np.random.choice(["Male", "Female"], n, p=[0.55, 0.45])
            race   = np.random.choice(["White", "Black", "Hispanic", "Asian"], n, p=[0.5, 0.2, 0.2, 0.1])
            age    = np.random.randint(22, 60, n)
            edu    = np.random.choice(["High School", "Bachelor", "Master", "PhD"], n)
            exp    = np.random.randint(0, 20, n)
            zipcode = np.random.choice(["10001","10002","10003","90210","60601"], n)
            # Biased target
            base_prob = 0.45
            prob = np.where(gender=="Male",   base_prob+0.20, base_prob) * \
                   np.where(race=="White",    1.3, np.where(race=="Asian", 1.1, 0.85)) * \
                   (1 + exp * 0.015)
            prob = np.clip(prob, 0.05, 0.95)
            hired = (np.random.rand(n) < prob).astype(int)
            demo_df = pd.DataFrame({
                "Gender": gender, "Race": race, "Age": age,
                "Education": edu, "Experience_Years": exp,
                "Zip_Code": zipcode, "Hired": hired
            })
            st.session_state.df = demo_df
            st.session_state.filename = "demo_hiring_data.csv"
            for k in ["model","preds","shap_values","X_enc","feature_cols","report_md"]:
                st.session_state[k] = None
            st.markdown('<div class="pass-banner">✅  Demo dataset loaded.</div>', unsafe_allow_html=True)
            st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown('<div class="section-header"><span class="num">02</span>DATA PREVIEW</div>', unsafe_allow_html=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            status_card("ROWS", f"{len(df):,}", "info")
        with col_b:
            status_card("COLUMNS", f"{len(df.columns)}", "info")
        with col_c:
            missing = df.isnull().sum().sum()
            status_card("MISSING VALUES", f"{missing:,}", "warn" if missing > 0 else "ok")
        with col_d:
            cat_cols = df.select_dtypes(include="object").shape[1]
            status_card("CATEGORICAL COLS", f"{cat_cols}", "info")

        st.markdown("**First 5 Rows**")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)

        st.markdown("**Column Statistics**")
        desc = df.describe(include="all").T.reset_index()
        desc.columns = [str(c) for c in desc.columns]
        st.dataframe(desc, use_container_width=True, hide_index=True)

        # Distribution plot
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            st.markdown("**Numeric Feature Distributions**")
            ncols = min(len(num_cols), 4)
            nrows = (len(num_cols) + ncols - 1) // ncols
            fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=num_cols[:ncols*nrows])
            for idx, col in enumerate(num_cols[:ncols*nrows]):
                r, c = divmod(idx, ncols)
                fig.add_trace(go.Histogram(x=df[col], name=col,
                    marker_color=CYAN, opacity=0.7, showlegend=False), row=r+1, col=c+1)
            fig.update_layout(**{**PLOTLY_LAYOUT, "height": 250*nrows, "showlegend": False,
                                  "title_text": "Distribution of Numeric Features"})
            fig.update_xaxes(gridcolor="#1a2540")
            fig.update_yaxes(gridcolor="#1a2540")
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — FAIRNESS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header"><span class="num">03</span>FAIRNESS DASHBOARD</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.markdown('<div class="info-box">⚠ Upload a dataset and run the audit first.</div>', unsafe_allow_html=True)
    elif not run_audit and st.session_state.model is None:
        st.markdown('<div class="info-box">Configure the sidebar and click <strong>RUN FULL AUDIT</strong> to see results.</div>', unsafe_allow_html=True)
    else:
        if run_audit and protected_col and target_col and privileged_val:
            df = st.session_state.df.copy()

            # Validate
            if df[protected_col].nunique() > 50:
                st.markdown('<div class="flag-banner">⛔ Protected attribute has too many unique values. Choose a categorical column.</div>', unsafe_allow_html=True)
                st.stop()
            if df[target_col].nunique() > 20:
                st.markdown('<div class="flag-banner">⛔ Target column has too many unique values. Choose a binary outcome column.</div>', unsafe_allow_html=True)
                st.stop()

            # Binarize target if needed
            unique_t = sorted(df[target_col].dropna().unique())
            if len(unique_t) == 2:
                mapping = {unique_t[0]: 0, unique_t[1]: 1}
                df[target_col] = df[target_col].map(mapping)

            with st.spinner("Training model and computing fairness metrics..."):
                model, encoders, preds, feat_cols = train_model(df, protected_col, target_col)
                st.session_state.model    = model
                st.session_state.preds    = preds
                st.session_state.feature_cols = feat_cols

                # Encode X for SHAP
                df_enc = df.copy()
                for col in feat_cols:
                    if df_enc[col].dtype == object or df_enc[col].dtype.name == "category":
                        le = encoders.get(col, LabelEncoder().fit(df_enc[col].astype(str)))
                        df_enc[col] = le.transform(df_enc[col].astype(str))
                X_enc = df_enc.fillna(-1)
                st.session_state.X_enc = X_enc

                metrics = compute_fairness_metrics(df, protected_col, target_col, privileged_val,
                                                   pd.Series(preds, index=df.index) if preds is not None else None)
                st.session_state.metrics     = metrics
                st.session_state.cfg = dict(
                    protected_col=protected_col, target_col=target_col,
                    privileged_val=privileged_val, filename=st.session_state.get("filename",""))

        if "metrics" not in st.session_state or st.session_state.get("metrics") is None:
            st.markdown('<div class="info-box">Run the audit first.</div>', unsafe_allow_html=True)
        else:
            df       = st.session_state.df
            metrics  = st.session_state.metrics
            cfg      = st.session_state.cfg
            protected_col  = cfg["protected_col"]
            target_col     = cfg["target_col"]
            privileged_val = cfg["privileged_val"]
            unpriv_vals    = df[protected_col].dropna().unique()
            unpriv_label   = [v for v in unpriv_vals if v != privileged_val]
            unpriv_label   = str(unpriv_label[0]) if unpriv_label else "Other"

            di = metrics["di_ratio"]
            di_pass = di >= 0.8

            # ── Metric 1: Demographic Parity ──────────────────────────────
            st.markdown("#### Metric 1 — Demographic Parity")
            info_expander("What is Demographic Parity?", """
            <strong>Demographic Parity</strong> (also called Statistical Parity) requires that the model's positive outcome rate is equal across groups.<br><br>
            <strong>Formula:</strong> <code>DP Difference = P(Ŷ=1 | A=unprivileged) − P(Ŷ=1 | A=privileged)</code><br><br>
            A value of <strong>0</strong> is ideal. Negative values mean the unprivileged group receives fewer positive outcomes.
            """)

            c1, c2, c3 = st.columns(3)
            with c1:
                status_card(f"SELECTION RATE — {privileged_val}",
                    f"{metrics['priv_selection_rate']:.1%}", "info",
                    f"Receives positive outcome {metrics['priv_selection_rate']:.1%} of the time")
            with c2:
                status_card(f"SELECTION RATE — {unpriv_label}",
                    f"{metrics['unpriv_selection_rate']:.1%}",
                    "ok" if abs(metrics['dp_difference']) < 0.05 else "fail",
                    f"Receives positive outcome {metrics['unpriv_selection_rate']:.1%} of the time")
            with c3:
                dp_status = "ok" if abs(metrics['dp_difference']) < 0.05 else ("warn" if abs(metrics['dp_difference']) < 0.1 else "fail")
                status_card("DP DIFFERENCE", f"{metrics['dp_difference']:+.4f}", dp_status,
                    "Closer to 0 = more fair")

            # Bar chart
            fig_bar = go.Figure([
                go.Bar(name=str(privileged_val), x=["Selection Rate"], y=[metrics['priv_selection_rate']],
                       marker_color=CYAN, marker_line_color=CYAN, marker_line_width=1),
                go.Bar(name=unpriv_label, x=["Selection Rate"], y=[metrics['unpriv_selection_rate']],
                       marker_color=RED if abs(metrics['dp_difference'])>0.05 else GREEN,
                       marker_line_color=RED if abs(metrics['dp_difference'])>0.05 else GREEN, marker_line_width=1)
            ])
            fig_bar.update_layout(**{**PLOTLY_LAYOUT, "barmode": "group",
                "title_text": "Selection Rate by Group",
                "yaxis_title": "Selection Rate", "height": 320})
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")

            # ── Metric 2: Disparate Impact ────────────────────────────────
            st.markdown("#### Metric 2 — Disparate Impact Ratio (EEOC 80% Rule)")
            info_expander("What is the 80% Rule?", """
            <strong>Disparate Impact</strong> is the primary legal standard used by the EEOC (Equal Employment Opportunity Commission).<br><br>
            <strong>Formula:</strong> <code>DI Ratio = Selection_Rate(unprivileged) / Selection_Rate(privileged)</code><br><br>
            Under the <strong>EEOC Uniform Guidelines (1978)</strong>, a ratio below <strong>0.80</strong> constitutes legal evidence of adverse impact — the employer must justify the practice as a business necessity.
            """)

            c1, c2 = st.columns([1, 2])
            with c1:
                di_status = "ok" if di >= 0.8 else "fail"
                status_card("DISPARATE IMPACT RATIO", f"{di:.4f}", di_status,
                    "✅ Passes 80% rule" if di >= 0.8 else "⛔ Fails 80% rule — LEGAL RISK")

            with c2:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=di,
                    number=dict(font=dict(color=GREEN if di_pass else RED, family="Rajdhani", size=36)),
                    delta=dict(reference=0.8, increasing=dict(color=GREEN), decreasing=dict(color=RED)),
                    gauge=dict(
                        axis=dict(range=[0, 1.5], tickwidth=1, tickcolor="#1a2540",
                                  tickfont=dict(color="#7a8ba8")),
                        bar=dict(color=GREEN if di_pass else RED),
                        bgcolor="#0b0f1a",
                        borderwidth=1, bordercolor="#1a2540",
                        steps=[
                            dict(range=[0, 0.8], color="rgba(255,51,102,0.15)"),
                            dict(range=[0.8, 1.25], color="rgba(0,255,136,0.08)"),
                            dict(range=[1.25, 1.5], color="rgba(255,170,0,0.1)"),
                        ],
                        threshold=dict(line=dict(color=AMBER, width=3), thickness=0.75, value=0.8)
                    ),
                    title=dict(text="Disparate Impact Ratio", font=dict(color="#7a8ba8", size=12, family="Share Tech Mono"))
                ))
                fig_gauge.update_layout(**{**PLOTLY_LAYOUT, "height": 240, "margin": dict(l=20,r=20,t=40,b=10)})
                st.plotly_chart(fig_gauge, use_container_width=True)

            if not di_pass:
                st.markdown(f'<div class="flag-banner">⛔ LEGAL WARNING: Disparate Impact Ratio = {di:.4f}. This is below the EEOC 80% threshold. This model may constitute illegal discrimination under US employment law and violates EU AI Act Article 10.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pass-banner">✅ COMPLIANT: Disparate Impact Ratio = {di:.4f}. Passes the EEOC 80% rule.</div>', unsafe_allow_html=True)

            st.markdown("---")

            # ── Metric 3: Equalized Odds ──────────────────────────────────
            st.markdown("#### Metric 3 — Equalized Odds (True Positive Rate)")
            info_expander("What is Equalized Odds?", """
            <strong>Equalized Odds</strong> requires that the model has equal True Positive Rates <em>and</em> False Positive Rates across groups.<br><br>
            <strong>True Positive Rate (TPR / Recall):</strong> <code>TPR = TP / (TP + FN)</code><br><br>
            If the TPR is lower for the unprivileged group, qualified members of that group are less likely to receive the positive outcome — a subtle but serious form of discrimination.
            """)

            tpr_p = metrics.get("tpr_privileged")
            tpr_u = metrics.get("tpr_unprivileged")
            if tpr_p is not None and tpr_u is not None:
                c1, c2, c3 = st.columns(3)
                tpr_diff = tpr_u - tpr_p
                with c1:
                    status_card(f"TPR — {privileged_val}", f"{tpr_p:.1%}", "info")
                with c2:
                    status_card(f"TPR — {unpriv_label}", f"{tpr_u:.1%}",
                        "ok" if abs(tpr_diff) < 0.05 else "fail")
                with c3:
                    status_card("TPR DIFFERENCE", f"{tpr_diff:+.4f}",
                        "ok" if abs(tpr_diff) < 0.05 else "fail",
                        "Closer to 0 = more fair")

                # Bias heatmap
                groups  = df[protected_col].dropna().unique()
                heatmap_metrics = []
                for g in groups:
                    mask = df[protected_col] == g
                    preds_s = pd.Series(st.session_state.preds, index=df.index)
                    sel_r = preds_s[mask].mean()
                    y_true = df[target_col]
                    tp = ((preds_s == 1) & (y_true == 1) & mask).sum()
                    fn = ((preds_s == 0) & (y_true == 1) & mask).sum()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fp = ((preds_s == 1) & (y_true == 0) & mask).sum()
                    tn = ((preds_s == 0) & (y_true == 0) & mask).sum()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    heatmap_metrics.append({"Group": str(g), "Selection Rate": sel_r,
                                             "TPR": tpr, "FPR": fpr})

                hm_df = pd.DataFrame(heatmap_metrics).set_index("Group")
                fig_hm = go.Figure(go.Heatmap(
                    z=hm_df.values,
                    x=hm_df.columns.tolist(),
                    y=hm_df.index.tolist(),
                    colorscale=[[0,"#ff3366"],[0.5,"#ffaa00"],[1,"#00ff88"]],
                    text=[[f"{v:.3f}" for v in row] for row in hm_df.values],
                    texttemplate="%{text}",
                    textfont=dict(family="Share Tech Mono", size=14),
                    zmin=0, zmax=1,
                    showscale=True,
                    colorbar=dict(tickfont=dict(color="#7a8ba8"))
                ))
                fig_hm.update_layout(**{**PLOTLY_LAYOUT, "height": 320,
                    "title_text": "Bias Heatmap — Fairness Metrics by Group"})
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.markdown('<div class="info-box">Equalized Odds requires model predictions. Run the audit with a model.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP EXPLAINER
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
        st.markdown('<div class="section-header"><span class="num">04</span>BLACK-BOX EXPLAINER</div>', unsafe_allow_html=True)
        info_expander("What is SHAP?", """
        <strong>SHAP</strong> (SHapley Additive exPlanations) uses game theory to explain the output of any ML model.<br><br>
        Each feature receives a <strong>Shapley value</strong> representing its contribution to the prediction for each individual.
        The global importance is computed as the mean absolute SHAP value across all predictions.<br><br>
        <strong>High SHAP importance</strong> means the model relies heavily on that feature to make decisions.
        """)

        if st.session_state.model is None:
            st.markdown('<div class="info-box">Run the audit first to compute SHAP values.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Computing SHAP explanations (this may take a moment)..."):
                if st.session_state.shap_values is None:
                    sv, X_sample = compute_shap(
                        st.session_state.model,
                        st.session_state.X_enc,
                        st.session_state.feature_cols
                    )
                    st.session_state.shap_values = sv
                    st.session_state.X_sample    = X_sample
                else:
                    sv       = st.session_state.shap_values
                    X_sample = st.session_state.X_sample

            # --- DATA PREP (THE IMMORTAL VERSION) ---
            feat_cols = list(st.session_state.feature_cols)
            
            # Handle SHAP values that might be 3D (multi-class) or 2D (binary)
            if len(sv.shape) == 3:
                # Multi-class: take the mean importance across all classes
                mean_abs = np.abs(sv).mean(axis=(0, 2))
            else:
                # Binary: standard mean absolute SHAP
                mean_abs = np.abs(sv).mean(axis=0)
            
            mean_abs_1d = np.array(mean_abs).flatten()

            # --- DYNAMIC LENGTH MATCHING ---
            num_values = len(mean_abs_1d)
            num_names = len(feat_cols)
            
            if num_names > num_values:
                actual_names = feat_cols[:num_values]
                display_values = mean_abs_1d
            elif num_values > num_names:
                actual_names = feat_cols
                display_values = mean_abs_1d[:num_names]
            else:
                actual_names = feat_cols
                display_values = mean_abs_1d

            # Create the DataFrame
            shap_df = pd.DataFrame({
                "Feature": actual_names, 
                "Mean |SHAP|": display_values
            }).sort_values("Mean |SHAP|", ascending=True)

            # Feature importance bar chart
            colors = [CYAN] * len(shap_df)
            fig_shap = go.Figure(go.Bar(
                x=shap_df["Mean |SHAP|"], y=shap_df["Feature"],
                orientation="h",
                marker_color=colors,
                marker_line_color=CYAN, marker_line_width=0.5,
            ))
            fig_shap.update_layout(**{**PLOTLY_LAYOUT, "height": max(300, 30 * len(actual_names) + 80),
                "title_text": "SHAP Feature Importance (Mean |SHAP Value|)",
                "xaxis_title": "Mean |SHAP Value|", "yaxis_title": ""})
            st.plotly_chart(fig_shap, use_container_width=True)

            # SHAP bee-swarm style (dot plot)
            st.markdown("**SHAP Value Distribution (Beeswarm)**")
            fig_bee = go.Figure()
            
            for i, feat in enumerate(shap_df["Feature"].tolist()):
                # Find the index in original feature list
                feat_idx = feat_cols.index(feat)
                shap_vals = sv[:, feat_idx]
                feat_vals = np.array(X_sample.iloc[:, feat_idx].values)
                
                # FIX: Use np.ptp() instead of .ptp() to avoid AttributeError
                range_vals = np.ptp(feat_vals) if len(feat_vals) > 0 else 0
                denom = range_vals if range_vals > 0 else 1
                norm_vals = (feat_vals - feat_vals.min()) / denom
                
                jitter = np.random.uniform(-0.3, 0.3, len(shap_vals))
                fig_bee.add_trace(go.Scatter(
                    x=shap_vals, y=np.full(len(shap_vals), i) + jitter,
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=norm_vals,
                        colorscale=[[0, "#0096b3"], [0.5, "#8b5cf6"], [1, "#ff3366"]],
                        opacity=0.6,
                    ),
                    name=feat, showlegend=False,
                    hovertemplate=f"<b>{feat}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
                ))
                
            fig_bee.update_layout(**{**PLOTLY_LAYOUT,
                "height": max(300, 30 * len(actual_names) + 80),
                "title_text": "SHAP Beeswarm — Feature Value (blue=low → red=high)",
                "xaxis_title": "SHAP Value (impact on model output)",
                "yaxis": dict(tickvals=list(range(len(shap_df))),
                              ticktext=shap_df["Feature"].tolist(),
                              gridcolor="#1a2540"),
            })
            st.plotly_chart(fig_bee, use_container_width=True)

            # ── Proxy Detector ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-header"><span class="num">05</span>PROXY FEATURE DETECTOR</div>', unsafe_allow_html=True)
            
            df  = st.session_state.df
            cfg = st.session_state.get("cfg", {})
            protected_col  = cfg.get("protected_col")

            if protected_col:
                # Use actual_names to ensure alignment with SHAP results
                corrs = correlation_with_protected(df, protected_col, actual_names)
                ranked_list = shap_df[::-1]["Feature"].tolist()

                CORR_THRESH  = 0.25
                SHAP_RANK_THRESH = max(1, int(len(ranked_list) * 0.4))

                proxies = []
                for feat, corr_v in corrs.items():
                    rank = ranked_list.index(feat) + 1 if feat in ranked_list else 999
                    if corr_v >= CORR_THRESH and rank <= SHAP_RANK_THRESH:
                        proxies.append({"feature": feat, "corr": corr_v, "rank": rank})

                st.session_state.proxies = proxies

                if proxies:
                    for p in proxies:
                        st.markdown(f"""
                        <div class="proxy-card">
                            ⚠ <strong>PROXY DETECTED: {p['feature']}</strong><br>
                            Correlation with <code>{protected_col}</code>: <code>{p['corr']:.3f}</code> &nbsp;|&nbsp;
                            SHAP Rank: <code>#{p['rank']} of {len(ranked_list)}</code><br>
                            <span style="color:#7a8ba8;font-size:11px">This feature may allow the model to discriminate indirectly.</span>
                        </div>""", unsafe_allow_html=True)

                    corr_df = pd.DataFrame([
                        {"Feature": k, "Cramér's V": v} for k, v in corrs.items()
                    ]).sort_values("Cramér's V", ascending=False)

                    fig_corr = go.Figure(go.Bar(
                        x=corr_df["Feature"], y=corr_df["Cramér's V"],
                        marker_color=[AMBER if f in [p["feature"] for p in proxies] else CYAN
                                      for f in corr_df["Feature"]],
                    ))
                    fig_corr.add_hline(y=CORR_THRESH, line_dash="dash", line_color=RED)
                    fig_corr.update_layout(**{**PLOTLY_LAYOUT, "height": 320})
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.markdown('<div class="pass-banner">✅ No high-risk proxy features detected.</div>', unsafe_allow_html=True)
                    st.session_state.proxies = []
# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — MITIGATION SANDBOX
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header"><span class="num">06</span>MITIGATION SANDBOX</div>', unsafe_allow_html=True)
    info_expander("What is Re-weighting?", """
    <strong>Re-weighting</strong> is a pre-processing de-biasing technique that assigns different sample weights to training instances
    to compensate for historical under/over-representation.<br><br>
    Groups that are under-represented in positive outcomes get higher weights, forcing the model to pay more attention to them.<br><br>
    <strong>Formula (Calmon et al., 2017):</strong> <code>w(x) = P(A) · P(Y) / P(A, Y)</code><br><br>
    This approach does not modify the data itself, only the learning process.
    """)

    if st.session_state.model is None:
        st.markdown('<div class="info-box">Run the audit first.</div>', unsafe_allow_html=True)
    else:
        apply_rw = st.toggle("Apply Re-weighting Mitigation", value=False)

        if apply_rw:
            df  = st.session_state.df
            cfg = st.session_state.get("cfg", {})
            protected_col  = cfg.get("protected_col")
            target_col     = cfg.get("target_col")
            privileged_val = cfg.get("privileged_val")

            if protected_col and target_col and privileged_val:
                with st.spinner("Re-weighting and retraining..."):
                    weights = reweight_data(df, protected_col, target_col, privileged_val)
                    _, _, preds_rw, _ = train_model(df, protected_col, target_col, sample_weight=weights)

                metrics_orig = st.session_state.metrics
                metrics_rw   = compute_fairness_metrics(df, protected_col, target_col, privileged_val,
                    pd.Series(preds_rw, index=df.index) if preds_rw is not None else None)
                st.session_state.metrics_rw = metrics_rw

                # Before / After comparison
                st.markdown("### Before vs. After Re-weighting")
                compare_data = {
                    "Metric": [
                        "Privileged Selection Rate",
                        "Unprivileged Selection Rate",
                        "DP Difference",
                        "Disparate Impact Ratio",
                        "TPR — Privileged",
                        "TPR — Unprivileged",
                    ],
                    "Before": [
                        f"{metrics_orig['priv_selection_rate']:.4f}",
                        f"{metrics_orig['unpriv_selection_rate']:.4f}",
                        f"{metrics_orig['dp_difference']:+.4f}",
                        f"{metrics_orig['di_ratio']:.4f}",
                        f"{metrics_orig.get('tpr_privileged',0):.4f}" if metrics_orig.get('tpr_privileged') is not None else "N/A",
                        f"{metrics_orig.get('tpr_unprivileged',0):.4f}" if metrics_orig.get('tpr_unprivileged') is not None else "N/A",
                    ],
                    "After": [
                        f"{metrics_rw['priv_selection_rate']:.4f}",
                        f"{metrics_rw['unpriv_selection_rate']:.4f}",
                        f"{metrics_rw['dp_difference']:+.4f}",
                        f"{metrics_rw['di_ratio']:.4f}",
                        f"{metrics_rw.get('tpr_privileged',0):.4f}" if metrics_rw.get('tpr_privileged') is not None else "N/A",
                        f"{metrics_rw.get('tpr_unprivileged',0):.4f}" if metrics_rw.get('tpr_unprivileged') is not None else "N/A",
                    ],
                }
                cmp_df = pd.DataFrame(compare_data)
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

                # Visual comparison
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(name="Before", x=["Selection Rate (Priv)", "Selection Rate (Unpriv)", "DI Ratio"],
                    y=[metrics_orig['priv_selection_rate'], metrics_orig['unpriv_selection_rate'], metrics_orig['di_ratio']],
                    marker_color=RED, opacity=0.8))
                fig_cmp.add_trace(go.Bar(name="After Re-weighting",
                    x=["Selection Rate (Priv)", "Selection Rate (Unpriv)", "DI Ratio"],
                    y=[metrics_rw['priv_selection_rate'], metrics_rw['unpriv_selection_rate'], metrics_rw['di_ratio']],
                    marker_color=GREEN, opacity=0.8))
                fig_cmp.add_hline(y=0.8, line_dash="dash", line_color=AMBER,
                    annotation_text="80% DI Threshold", annotation_font_color=AMBER)
                fig_cmp.update_layout(**{**PLOTLY_LAYOUT, "barmode": "group", "height": 380,
                    "title_text": "Fairness Metrics: Before vs. After Re-weighting",
                    "yaxis_title": "Value"})
                st.plotly_chart(fig_cmp, use_container_width=True)

                di_delta = metrics_rw['di_ratio'] - metrics_orig['di_ratio']
                if di_delta > 0.01:
                    st.markdown(f'<div class="pass-banner">✅ Re-weighting improved the Disparate Impact Ratio by +{di_delta:.4f}.</div>', unsafe_allow_html=True)
                elif di_delta > 0:
                    st.markdown(f'<div class="pass-banner">✅ Modest improvement: +{di_delta:.4f} in Disparate Impact Ratio.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="flag-banner">⚠ Re-weighting did not significantly improve the Disparate Impact Ratio on this dataset. Consider more advanced techniques.</div>', unsafe_allow_html=True)

                if not (metrics_rw['di_ratio'] >= 0.8):
                    st.markdown("""
                    <div class="info-box"><strong>Further Mitigation Strategies:</strong><br>
                    1. <strong>Fairlearn ExponentiatedGradient</strong> — in-processing fairness constraints during training<br>
                    2. <strong>Adversarial Debiasing</strong> — a neural network architecture that learns to be fair<br>
                    3. <strong>Calibrated Equalized Odds</strong> — post-processing threshold optimization<br>
                    4. <strong>Remove proxy features</strong> identified in the SHAP explainer tab
                    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — AUDIT REPORT
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header"><span class="num">07</span>AUDIT REPORT EXPORT</div>', unsafe_allow_html=True)

    if st.session_state.get("metrics") is None:
        st.markdown('<div class="info-box">Run the audit first to generate the report.</div>', unsafe_allow_html=True)
    else:
        cfg            = st.session_state.get("cfg", {})
        metrics        = st.session_state.metrics
        metrics_rw     = st.session_state.get("metrics_rw")
        proxies        = st.session_state.get("proxies", [])
        di_pass        = metrics["di_ratio"] >= 0.8

        report_md = generate_markdown_report(cfg, metrics, metrics_rw, proxies, di_pass)
        st.session_state.report_md = report_md

        st.markdown("**Report Preview:**")
        st.markdown(report_md)

        st.download_button(
            label="⬇  DOWNLOAD MARKDOWN REPORT",
            data=report_md,
            file_name=f"aequitas_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=False,
        )

        # PDF export via fpdf2
        st.markdown("---")
        if st.button("⬇  EXPORT PDF REPORT", use_container_width=False):
            try:
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 20)
                pdf.set_text_color(0, 212, 255)
                pdf.cell(0, 12, "AEQUITAS: ALGORITHMIC BIAS AUDIT REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(120, 140, 170)
                pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", new_x="LMARGIN", new_y="NEXT", align="C")
                pdf.ln(6)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(50, 50, 60)
                # Write stripped markdown as plain text
                clean = report_md.replace("**", "").replace("`", "").replace("#", "").replace("_", "")
                for line in clean.split("\n"):
                    try:
                        pdf.multi_cell(0, 5, line.encode("latin-1", "replace").decode("latin-1"))
                    except Exception:
                        pass
                buf = io.BytesIO(bytes(pdf.output()))
                st.download_button("📄 Download PDF", data=buf,
                    file_name=f"aequitas_audit_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf")
            except Exception as e:
                st.error(f"PDF generation error: {e}")
