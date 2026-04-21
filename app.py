import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="PCOS Clinical Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Merriweather:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
.stApp { background-color: #f0f2f5; }

.header-bar {
    background: linear-gradient(90deg, #1a3a6b 0%, #2557a7 100%);
    color: white; padding: 20px 36px;
    margin: -1rem -1rem 28px -1rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 3px 12px rgba(26,58,107,0.3);
}
.header-title { font-family: 'Merriweather', serif; font-size: 1.4rem; font-weight: 700; }
.header-sub { font-size: 0.78rem; opacity: 0.75; margin-top: 4px; letter-spacing: 0.04em; text-transform: uppercase; }
.header-badge {
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.3);
    border-radius: 8px; padding: 8px 18px; font-size: 0.85rem; font-weight: 600;
}

.stats-row { display: flex; gap: 14px; margin-bottom: 22px; }
.stat-card {
    flex: 1; background: white; border-radius: 10px;
    padding: 16px 20px; border-left: 5px solid;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    display: flex; align-items: center; gap: 14px;
}
.stat-card:nth-child(1) { border-color: #2557a7; }
.stat-card:nth-child(2) { border-color: #0d7377; }
.stat-card:nth-child(3) { border-color: #5a67a6; }
.stat-card:nth-child(4) { border-color: #b7451a; }
.stat-icon { font-size: 1.9rem; }
.stat-value { font-size: 1.55rem; font-weight: 700; color: #1a1a2e; line-height: 1.1; }
.stat-label { font-size: 0.73rem; color: #999; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

.stTabs [data-baseweb="tab-list"] {
    background: white !important; border-radius: 10px !important;
    padding: 5px !important; gap: 4px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07) !important;
    margin-bottom: 16px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important; padding: 10px 24px !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    color: #666 !important; background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #2557a7 !important; color: white !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: white; border-radius: 12px;
    padding: 26px 28px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}

label { font-size: 0.84rem !important; font-weight: 600 !important; color: #374151 !important; }
.stNumberInput input {
    border-radius: 8px !important; border: 1.5px solid #d1d9e6 !important;
    background: #fafbfd !important;
}

.stButton > button {
    background: linear-gradient(90deg, #1a3a6b, #2557a7) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; padding: 14px 0 !important;
    font-size: 1rem !important; font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(37,87,167,0.35) !important;
}

.result-positive {
    background: #fff5f5; border: 2px solid #fc8181;
    border-radius: 12px; padding: 30px 28px; text-align: center;
}
.result-negative {
    background: #f0fff4; border: 2px solid #68d391;
    border-radius: 12px; padding: 30px 28px; text-align: center;
}
.result-heading { font-family: 'Merriweather', serif; font-size: 1.5rem; font-weight: 700; margin: 10px 0 6px 0; }
.result-positive .result-heading { color: #c53030; }
.result-negative .result-heading { color: #276749; }
.result-note { font-size: 0.88rem; color: #555; line-height: 1.6; }

.conf-card {
    background: white; border-radius: 12px;
    padding: 24px 26px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); height: 100%;
}
.conf-title {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #2557a7;
    border-bottom: 1px solid #e8edf5; padding-bottom: 10px; margin-bottom: 20px;
}
.conf-row { margin-bottom: 18px; }
.conf-lbl { display: flex; justify-content: space-between; font-size: 0.87rem; font-weight: 600; color: #333; margin-bottom: 7px; }
.conf-track { background: #edf2f7; border-radius: 999px; height: 11px; overflow: hidden; }
.conf-fill-p { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #c53030, #fc8181); }
.conf-fill-n { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #276749, #68d391); }

[data-testid="stMetric"] {
    background: white !important; border-radius: 10px !important;
    padding: 16px 18px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.07) !important;
    border-top: 3px solid #2557a7 !important;
}

.disclaimer {
    background: #fffbeb; border: 1px solid #f6d860;
    border-left: 4px solid #d69e2e; border-radius: 8px;
    padding: 14px 18px; font-size: 0.84rem; color: #744210;
    line-height: 1.6; margin-top: 18px;
}
.footer {
    text-align: center; padding: 22px; color: #bbb;
    font-size: 0.8rem; margin-top: 16px; border-top: 1px solid #e2e8f0;
}
.footer a { color: #2557a7; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("PCOS_extended_dataset.csv")
    df = df.drop(columns=["Sl. No", "Patient File No."], errors="ignore")
    for col in ["II    beta-HCG(mIU/mL)", "AMH(ng/mL)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    X = df.drop("PCOS (Y/N)", axis=1)
    y = df["PCOS (Y/N)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = RandomForestClassifier(n_estimators=100, random_state=42)
    mdl.fit(X_train, y_train)
    acc = accuracy_score(y_test, mdl.predict(X_test))
    return mdl, X.columns.tolist(), X.median().to_dict(), acc, len(df)

model, feature_cols, medians, accuracy, n_patients = load_model()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-bar">
  <div>
    <div class="header-title">🏥 PCOS Clinical Detection System</div>
    <div class="header-sub">Supervised Machine Learning · Random Forest Classifier · Binary Classification</div>
  </div>
  <div class="header-badge">Model Accuracy &nbsp;{accuracy*100:.1f}%</div>
</div>
""", unsafe_allow_html=True)

# ── Stat Cards ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stats-row">
  <div class="stat-card">
    <div class="stat-icon">🎯</div>
    <div><div class="stat-value">{accuracy*100:.1f}%</div><div class="stat-label">Test Accuracy</div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">👥</div>
    <div><div class="stat-value">{n_patients:,}</div><div class="stat-label">Patient Records</div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">📊</div>
    <div><div class="stat-value">42</div><div class="stat-label">Clinical Features</div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">🌲</div>
    <div><div class="stat-value">100</div><div class="stat-label">Decision Trees</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input Tabs ───────────────────────────────────────────────────────────────
st.markdown("##### Enter Patient Clinical Parameters")
t1, t2, t3, t4 = st.tabs([
    "👤  Personal & Physical",
    "🔬  Hormonal Profile",
    "🫀  Ultrasound & Blood",
    "📋  Symptoms & Lifestyle"
])

with t1:
    a, b, c, d = st.columns(4)
    age    = a.number_input("Age (years)",      15, 55, 28)
    weight = b.number_input("Weight (Kg)",      30.0, 150.0, 60.0, 0.5)
    height = c.number_input("Height (Cm)",      130.0, 200.0, 160.0, 0.5)
    bmi    = d.number_input("BMI",              10.0, 55.0, 24.0, 0.1)
    e, f, g, h = st.columns(4)
    waist  = e.number_input("Waist (inch)",     20.0, 60.0, 30.0, 0.5)
    hip    = f.number_input("Hip (inch)",       25.0, 65.0, 37.0, 0.5)
    pulse  = g.number_input("Pulse rate (bpm)", 40, 120, 72)
    rr     = h.number_input("RR (breaths/min)", 10, 40, 18)

with t2:
    a, b, c = st.columns(3)
    fsh    = a.number_input("FSH (mIU/mL)",         0.0, 30.0, 6.5, 0.1)
    lh     = b.number_input("LH (mIU/mL)",          0.0, 40.0, 5.0, 0.1)
    fsh_lh = c.number_input("FSH / LH Ratio",       0.0, 10.0, 1.3, 0.01)
    d, e, f = st.columns(3)
    amh    = d.number_input("AMH (ng/mL)",          0.0, 20.0, 3.5, 0.1)
    tsh    = e.number_input("TSH (mIU/L)",          0.0, 10.0, 2.0, 0.1)
    prl    = f.number_input("Prolactin (ng/mL)",    0.0, 100.0, 15.0, 0.5)
    g, h, i = st.columns(3)
    vit_d  = g.number_input("Vitamin D3 (ng/mL)",   0.0, 100.0, 25.0, 0.5)
    prg    = h.number_input("Progesterone (ng/mL)", 0.0, 20.0, 0.5, 0.1)
    rbs    = i.number_input("RBS (mg/dl)",          50.0, 300.0, 90.0, 1.0)

with t3:
    a, b, c, d = st.columns(4)
    hb           = a.number_input("Hb (g/dl)",           6.0, 18.0, 12.5, 0.1)
    cycle_len    = b.number_input("Cycle length (days)", 20, 60, 28)
    follicle_l   = c.number_input("Follicle No. (L)",    0, 30, 5)
    follicle_r   = d.number_input("Follicle No. (R)",    0, 30, 5)
    e, f, g = st.columns(3)
    avg_f_size_l = e.number_input("Avg. F size L (mm)",  0.0, 30.0, 10.0, 0.5)
    avg_f_size_r = f.number_input("Avg. F size R (mm)",  0.0, 30.0, 10.0, 0.5)
    endometrium  = g.number_input("Endometrium (mm)",    0.0, 20.0, 7.0, 0.1)

with t4:
    a, b, c = st.columns(3)
    cycle_ri     = a.selectbox("Menstrual Cycle",     [2,4], format_func=lambda x: "Regular" if x==2 else "Irregular")
    pregnant     = b.selectbox("Pregnant",            [0,1], format_func=lambda x: "Yes" if x else "No")
    abortions    = c.number_input("No. of Abortions", 0, 10, 0)
    d, e, f = st.columns(3)
    weight_gain  = d.selectbox("Weight Gain",         [0,1], format_func=lambda x: "Yes" if x else "No")
    hair_growth  = e.selectbox("Excess Hair Growth",  [0,1], format_func=lambda x: "Yes" if x else "No")
    skin_dark    = f.selectbox("Skin Darkening",      [0,1], format_func=lambda x: "Yes" if x else "No")
    g, h, i = st.columns(3)
    hair_loss    = g.selectbox("Hair Loss",           [0,1], format_func=lambda x: "Yes" if x else "No")
    pimples      = h.selectbox("Pimples / Acne",      [0,1], format_func=lambda x: "Yes" if x else "No")
    fast_food    = i.selectbox("Fast Food Intake",    [0,1], format_func=lambda x: "Yes" if x else "No")
    _, j, _ = st.columns([1,1,1])
    reg_exercise = j.selectbox("Regular Exercise",    [0,1], format_func=lambda x: "Yes" if x else "No")

# ── Predict Button ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1.5, 2, 1.5])
with btn_col:
    predict = st.button("🔍  Run Diagnostic Prediction", use_container_width=True)

# ── Result ───────────────────────────────────────────────────────────────────
if predict:
    user_values = {
        "Age (yrs)": age, "Weight (Kg)": weight, "Height(Cm) ": height, "BMI": bmi,
        "Pulse rate(bpm) ": pulse, "RR (breaths/min)": rr, "Hb(g/dl)": hb,
        "Cycle(R/I)": cycle_ri, "Cycle length(days)": cycle_len,
        "Pregnant(Y/N)": pregnant, "No. of abortions": abortions,
        "FSH(mIU/mL)": fsh, "LH(mIU/mL)": lh, "FSH/LH": fsh_lh,
        "Hip(inch)": hip, "Waist(inch)": waist,
        "Waist:Hip Ratio": round(waist/hip, 3) if hip > 0 else medians.get("Waist:Hip Ratio", 0.8),
        "TSH (mIU/L)": tsh, "AMH(ng/mL)": amh, "PRL(ng/mL)": prl,
        "Vit D3 (ng/mL)": vit_d, "PRG(ng/mL)": prg, "RBS(mg/dl)": rbs,
        "Weight gain(Y/N)": weight_gain, "hair growth(Y/N)": hair_growth,
        "Skin darkening (Y/N)": skin_dark, "Hair loss(Y/N)": hair_loss,
        "Pimples(Y/N)": pimples, "Fast food (Y/N)": fast_food,
        "Reg.Exercise(Y/N)": reg_exercise,
        "Follicle No. (L)": follicle_l, "Follicle No. (R)": follicle_r,
        "Avg. F size (L) (mm)": avg_f_size_l, "Avg. F size (R) (mm)": avg_f_size_r,
        "Endometrium (mm)": endometrium,
    }
    row = {col: user_values.get(col, medians.get(col, 0)) for col in feature_cols}
    input_df = pd.DataFrame([row])

    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    pcos_pct   = int(proba[1] * 100)
    no_pct     = int(proba[0] * 100)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("##### Diagnostic Result")

    r1, r2 = st.columns([1, 1], gap="large")
    with r1:
        if prediction == 1:
            st.markdown("""
            <div class="result-positive">
              <div style="font-size:2.6rem">⚕️</div>
              <div class="result-heading">PCOS Detected</div>
              <div class="result-note">Clinical parameters indicate a positive PCOS result.<br>
              Gynaecological evaluation is strongly recommended.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-negative">
              <div style="font-size:2.6rem">✅</div>
              <div class="result-heading">No PCOS Detected</div>
              <div class="result-note">Parameters appear within normal clinical range.<br>
              Routine health monitoring is advised.</div>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="conf-card">
          <div class="conf-title">Prediction Confidence Score</div>
          <div class="conf-row">
            <div class="conf-lbl"><span>⚕️ PCOS Positive</span><span>{pcos_pct}%</span></div>
            <div class="conf-track"><div class="conf-fill-p" style="width:{pcos_pct}%"></div></div>
          </div>
          <div class="conf-row">
            <div class="conf-lbl"><span>✅ PCOS Negative</span><span>{no_pct}%</span></div>
            <div class="conf-track"><div class="conf-fill-n" style="width:{no_pct}%"></div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Key Clinical Indicators")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FSH/LH Ratio",    f"{fsh_lh:.2f}",             "Above normal" if fsh_lh > 2 else "Normal range")
    m2.metric("AMH (ng/mL)",     f"{amh:.1f}",                 "Elevated"     if amh > 3.5 else "Normal range")
    m3.metric("Total Follicles", f"{follicle_l+follicle_r}",   "High count"   if (follicle_l+follicle_r) > 12 else "Normal range")
    m4.metric("BMI",             f"{bmi:.1f}",                  "Overweight"  if bmi > 25 else "Normal range")

    st.markdown("""
    <div class="disclaimer">
      <strong>⚠️ Medical Disclaimer:</strong> This system is developed as an academic ML project
      for educational purposes only. The result generated does <strong>not</strong> constitute a
      clinical diagnosis. Evaluation by a qualified gynaecologist is mandatory before any
      treatment decision is made.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  PCOS Clinical Detection System &nbsp;·&nbsp; Random Forest Classifier &nbsp;·&nbsp;
  Department of CSE, Chitkara University &nbsp;·&nbsp;
  <a href="https://github.com/kavmakani/PCOS-Detection" target="_blank">GitHub Repository ↗</a>
</div>
""", unsafe_allow_html=True)