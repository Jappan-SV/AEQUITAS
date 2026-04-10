# ⚖ AEQUITAS — Setup & Run Guide

## Prerequisites

```bash
pip install streamlit fairlearn shap plotly scikit-learn pandas numpy openpyxl fpdf2 scipy
```

## Run the App

```bash
streamlit run aequitas_app.py
```

Then open `http://localhost:8501` in your browser.

---

## Quick Start (Demo Dataset)

1. Open the app
2. Go to **📂 DATA INGESTION** tab
3. Click **"Load Demo Dataset"** — loads a synthetic biased hiring dataset (500 rows)
4. In the **sidebar**:
   - Protected Attribute → `Gender`
   - Target Column → `Hired`
   - Privileged Group → `Male`
5. Click **▶ RUN FULL AUDIT**
6. Explore all tabs

---

## Features Guide

| Tab | Feature |
|-----|---------|
| 📂 Data Ingestion | Upload CSV/XLSX, data preview, distribution plots |
| 📊 Fairness Dashboard | Demographic Parity, Disparate Impact (80% Rule), Equalized Odds, Bias Heatmap |
| 🧠 SHAP Explainer | Feature importance, beeswarm plot, proxy detector |
| 🔧 Mitigation Sandbox | Re-weighting toggle, before/after comparison |
| 📄 Audit Report | Full Markdown + PDF export |

---

## CSV Format Requirements

Your dataset should have:
- **At least one protected attribute column** (e.g., Gender, Race) — categorical
- **A binary outcome column** (e.g., Hired: 0/1 or Yes/No) — binary
- Additional feature columns (numeric or categorical)

### Example minimal CSV:

```csv
Gender,Age,Education,Zip_Code,Hired
Male,34,Bachelor,10001,1
Female,28,Master,10002,0
Male,45,PhD,90210,1
...
```

---

## Legal References

- **EU AI Act** (Regulation EU 2024/1689) — Articles 10, 13, 15
- **EEOC Uniform Guidelines on Employee Selection Procedures (1978)** — 80% Rule
- **Title VII, Civil Rights Act (1964)**
- **Fair Housing Act / Equal Credit Opportunity Act**

---

## Fairness Metrics Reference

| Metric | Formula | Threshold |
|--------|---------|-----------|
| Demographic Parity Difference | P(Ŷ=1\|Unpriv) − P(Ŷ=1\|Priv) | ≈ 0 |
| Disparate Impact Ratio | P(Ŷ=1\|Unpriv) / P(Ŷ=1\|Priv) | ≥ 0.80 (EEOC) |
| Equal TPR Difference | TPR(Unpriv) − TPR(Priv) | ≈ 0 |
| Cramér's V (Proxy) | √(χ²/(n·min(r,c)−1)) | < 0.25 = low risk |
