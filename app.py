import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Detection System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Quicksand:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #fff0f6 0%, #f0f4ff 40%, #f0fff8 100%);
    background-attachment: fixed;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ff6b9d 0%, #c44dff 50%, #6b8cff 100%) !important;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.9) !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.2) !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
    color: white !important;
    border-radius: 10px !important;
}

.sidebar-section {
    background: rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 8px 12px;
    margin: 14px 0 8px 0;
    font-weight: 800;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-left: 4px solid rgba(255,255,255,0.6);
}

.hero {
    background: linear-gradient(135deg, #ff6b9d, #c44dff, #6b8cff);
    border-radius: 24px;
    padding: 40px 48px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(196,77,255,0.3);
}
.hero h1 {
    font-family: 'Quicksand', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    margin: 0 0 10px 0 !important;
    color: white !important;
}
.hero p { font-size: 1rem; opacity: 0.92; margin: 0; }

.stats-row { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 130px;
    background: white; border-radius: 18px;
    padding: 20px 24px; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    border-top: 4px solid;
}
.stat-card:nth-child(1) { border-color: #ff6b9d; }
.stat-card:nth-child(2) { border-color: #c44dff; }
.stat-card:nth-child(3) { border-color: #6b8cff; }
.stat-card:nth-child(4) { border-color: #00c9a7; }
.stat-value { font-family: 'Quicksand', sans-serif; font-size: 2rem; font-weight: 700; color: #2d2d2d; }
.stat-label { font-size: 0.76rem; color: #999; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

.result-pcos {
    background: linear-gradient(135deg, #ff6b9d, #ff4757);
    border-radius: 20px; padding: 32px;
    color: white; text-align: center;
    box-shadow: 0 12px 40px rgba(255,107,157,0.4);
}
.result-no-pcos {
    background: linear-gradient(135deg, #00c9a7, #00b4d8);
    border-radius: 20px; padding: 32px;
    color: white; text-align: center;
    box-shadow: 0 12px 40px rgba(0,201,167,0.4);
}
.result-emoji { font-size: 3.5rem; }
.result-title { font-family: 'Quicksand', sans-serif; font-size: 1.8rem; font-weight: 700; }
.result-subtitle { opacity: 0.9; margin-top: 8px; font-size: 0.95rem; }

.conf-wrap { background: white; border-radius: 18px; padding: 26px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); }
.conf-label { font-weight: 700; color: #666; font-size: 0.83rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.conf-bar-bg { background: #f0f0f0; border-radius: 999px; height: 14px; overflow: hidden; margin-bottom: 4px; }
.conf-fill-pcos { height:100%; border-radius:999px; background: linear-gradient(90deg,#ff6b9d,#ff4757); }
.conf-fill-no   { height:100%; border-radius:999px; background: linear-gradient(90deg,#00c9a7,#00b4d8); }
.conf-pct { font-size: 0.88rem; color: #aaa; text-align: right; margin-bottom: 16px; }

.info-box {
    background: linear-gradient(135deg, #fff8ff, #f0f4ff);
    border: 1.5px solid #e0d4ff; border-radius: 16px;
    padding: 18px 22px; margin-top: 20px;
    font-size: 0.87rem; color: #666; line-height: 1.7;
}
.info-box strong { color: #c44dff; }

.stButton > button {
    background: linear-gradient(135deg, #ff6b9d, #c44dff, #6b8cff) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important; padding: 14px 0 !important;
    font-size: 1.05rem !important; font-weight: 800 !important;
    font-family: 'Quicksand', sans-serif !important;
    width: 100% !important;
    box-shadow: 0 8px 25px rgba(196,77,255,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns.tolist(), X.median().to_dict(), acc, len(df)

model, feature_cols, medians, accuracy, total_patients = load_model()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Patient Details")
    st.markdown("Enter clinical parameters and click **Predict**.")
    st.markdown("---")

    st.markdown('<div class="sidebar-section"> Personal & Physical</div>', unsafe_allow_html=True)
    age        = st.number_input("Age (years)",          15, 55, 28)
    weight     = st.number_input("Weight (Kg)",          30.0, 150.0, 60.0, 0.5)
    height     = st.number_input("Height (Cm)",          130.0, 200.0, 160.0, 0.5)
    bmi        = st.number_input("BMI",                  10.0, 55.0, 24.0, 0.1)
    waist      = st.number_input("Waist (inch)",         20.0, 60.0, 30.0, 0.5)
    hip        = st.number_input("Hip (inch)",           25.0, 65.0, 37.0, 0.5)
    pulse      = st.number_input("Pulse rate (bpm)",     40, 120, 72)
    rr         = st.number_input("RR (breaths/min)",     10, 40, 18)

    st.markdown('<div class="sidebar-section"> Hormonal Levels</div>', unsafe_allow_html=True)
    fsh        = st.number_input("FSH (mIU/mL)",         0.0, 30.0, 6.5, 0.1)
    lh         = st.number_input("LH (mIU/mL)",          0.0, 40.0, 5.0, 0.1)
    fsh_lh     = st.number_input("FSH/LH ratio",         0.0, 10.0, 1.3, 0.01)
    amh        = st.number_input("AMH (ng/mL)",          0.0, 20.0, 3.5, 0.1)
    tsh        = st.number_input("TSH (mIU/L)",          0.0, 10.0, 2.0, 0.1)
    prl        = st.number_input("Prolactin (ng/mL)",    0.0, 100.0, 15.0, 0.5)
    vit_d      = st.number_input("Vitamin D3 (ng/mL)",   0.0, 100.0, 25.0, 0.5)
    prg        = st.number_input("Progesterone (ng/mL)", 0.0, 20.0, 0.5, 0.1)
    rbs        = st.number_input("RBS (mg/dl)",          50.0, 300.0, 90.0, 1.0)

    st.markdown('<div class="sidebar-section"> Blood & Ultrasound</div>', unsafe_allow_html=True)
    hb           = st.number_input("Hb (g/dl)",           6.0, 18.0, 12.5, 0.1)
    cycle_len    = st.number_input("Cycle length (days)", 20, 60, 28)
    follicle_l   = st.number_input("Follicle No. (L)",    0, 30, 5)
    follicle_r   = st.number_input("Follicle No. (R)",    0, 30, 5)
    avg_f_size_l = st.number_input("Avg. F size (L) mm",  0.0, 30.0, 10.0, 0.5)
    avg_f_size_r = st.number_input("Avg. F size (R) mm",  0.0, 30.0, 10.0, 0.5)
    endometrium  = st.number_input("Endometrium (mm)",    0.0, 20.0, 7.0, 0.1)

    st.markdown('<div class="sidebar-section"> Symptoms & Lifestyle</div>', unsafe_allow_html=True)
    cycle_ri     = st.selectbox("Cycle", [2,4], format_func=lambda x: "Regular" if x==2 else "Irregular")
    pregnant     = st.selectbox("Pregnant",          [0,1], format_func=lambda x: "Yes" if x else "No")
    abortions    = st.number_input("No. of abortions", 0, 10, 0)
    weight_gain  = st.selectbox("Weight gain",       [0,1], format_func=lambda x: "Yes" if x else "No")
    hair_growth  = st.selectbox("Hair growth",       [0,1], format_func=lambda x: "Yes" if x else "No")
    skin_dark    = st.selectbox("Skin darkening",    [0,1], format_func=lambda x: "Yes" if x else "No")
    hair_loss    = st.selectbox("Hair loss",         [0,1], format_func=lambda x: "Yes" if x else "No")
    pimples      = st.selectbox("Pimples",           [0,1], format_func=lambda x: "Yes" if x else "No")
    fast_food    = st.selectbox("Fast food",         [0,1], format_func=lambda x: "Yes" if x else "No")
    reg_exercise = st.selectbox("Regular exercise",  [0,1], format_func=lambda x: "Yes" if x else "No")

    st.markdown("---")
    predict_btn = st.button(" Run PCOS Prediction", use_container_width=True)


# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1> PCOS Detection System</h1>
    <p>An intelligent ML-powered tool for early detection of <strong>Polycystic Ovary Syndrome</strong>
    using clinical &amp; hormonal parameters. Fill patient details in the sidebar and click Predict.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-row">
    <div class="stat-card"><div class="stat-value">{accuracy*100:.1f}%</div><div class="stat-label">Accuracy</div></div>
    <div class="stat-card"><div class="stat-value">{total_patients:,}</div><div class="stat-label">Patients</div></div>
    <div class="stat-card"><div class="stat-value">44</div><div class="stat-label">Features</div></div>
    <div class="stat-card"><div class="stat-value">RF</div><div class="stat-label">Algorithm</div></div>
</div>
""", unsafe_allow_html=True)

if not predict_btn:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 🩺 How to Use
        1. Fill patient's clinical details in the **sidebar** on the left
        2. Enter hormonal levels, ultrasound findings & symptoms
        3. Click **Run PCOS Prediction**
        4. Get instant result with confidence score
        """)
    with c2:
        st.markdown("""
        ### 📊 About the Model
        - **Algorithm:** Random Forest (100 trees)
        - **Dataset:** 2000 patient records, 44 features
        - **Best performer** among 5 tested algorithms
        - Trained on hormonal, physical & ultrasound data
        """)
    st.markdown("""
    <div class="info-box">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is built for <strong>educational purposes</strong>
        as part of a university ML project. It is <strong>not</strong> a substitute for clinical diagnosis.
        Always consult a qualified gynaecologist for medical decisions.
    </div>
    """, unsafe_allow_html=True)

else:
    # Build input
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

    col1, col2 = st.columns([1,1], gap="large")
    with col1:
        if prediction == 1:
            st.markdown("""
            <div class="result-pcos">
                <div class="result-emoji">⚠️</div>
                <div class="result-title">PCOS Detected</div>
                <div class="result-subtitle">Please consult a gynaecologist for clinical confirmation.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-no-pcos">
                <div class="result-emoji">✅</div>
                <div class="result-title">No PCOS Detected</div>
                <div class="result-subtitle">Parameters appear within normal range. Stay healthy!</div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="conf-wrap">
            <div style="font-family:Quicksand;font-weight:700;font-size:1.1rem;margin-bottom:20px;color:#333;">Prediction Confidence</div>
            <div class="conf-label"> PCOS Likelihood</div>
            <div class="conf-bar-bg"><div class="conf-fill-pcos" style="width:{pcos_pct}%"></div></div>
            <div class="conf-pct">{pcos_pct}%</div>
            <div class="conf-label">✅ No PCOS Likelihood</div>
            <div class="conf-bar-bg"><div class="conf-fill-no" style="width:{no_pct}%"></div></div>
            <div class="conf-pct">{no_pct}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Key Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("FSH/LH Ratio", f"{fsh_lh:.2f}", "↑ Risk" if fsh_lh > 2 else "Normal")
    k2.metric("AMH (ng/mL)",  f"{amh:.1f}",    "↑ Risk" if amh > 3.5 else "Normal")
    k3.metric("Follicles (L+R)", f"{follicle_l+follicle_r}", "↑ Risk" if (follicle_l+follicle_r) > 12 else "Normal")
    k4.metric("BMI",          f"{bmi:.1f}",    "Overweight" if bmi > 25 else "Normal")

    st.markdown("""
    <div class="info-box">
        <strong>⚠️ Disclaimer:</strong> This prediction is for <strong>educational purposes only</strong>
        and is not a medical diagnosis. Consult a qualified healthcare professional for clinical evaluation.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<center><small style='color:#bbb'> PCOS Detection System &nbsp;·&nbsp; Random Forest ML &nbsp;·&nbsp; "
    "Chitkara University &nbsp;·&nbsp; "
    "<a href='https://github.com/kavmakani/PCOS-Detection' style='color:#c44dff'>GitHub ↗</a></small></center>",
    unsafe_allow_html=True
)