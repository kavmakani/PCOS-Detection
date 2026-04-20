import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Detection System",
    page_icon="🩺",
    layout="wide"
)

# ── Load & preprocess (same surgical fix as your notebook) ───────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("PCOS_extended_dataset.csv")
    df = df.drop(columns=["Sl. No", "Patient File No."], errors="ignore")

    # Only fix the two known dirty columns
    for col in ["II    beta-HCG(mIU/mL)", "AMH(ng/mL)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(df.median(numeric_only=True))

    X = df.drop("PCOS (Y/N)", axis=1)
    y = df["PCOS (Y/N)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns.tolist(), X.median().to_dict(), acc

model, feature_cols, medians, accuracy = load_model()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🩺 PCOS Detection System")
st.markdown(
    "A machine learning tool to assist in early detection of "
    "**Polycystic Ovary Syndrome (PCOS)** using clinical parameters."
)
st.info(f"Model: Random Forest  |  Test Accuracy: **{accuracy*100:.1f}%**  |  "
        f"Dataset: 2000 patients, 44 features")

st.divider()

# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal & Physical**")
    age        = st.number_input("Age (years)",          min_value=15,  max_value=55,  value=28)
    bmi        = st.number_input("BMI",                  min_value=10.0, max_value=55.0, value=24.0, step=0.1)
    weight     = st.number_input("Weight (Kg)",          min_value=30.0, max_value=150.0, value=60.0, step=0.5)
    height     = st.number_input("Height (Cm)",          min_value=130.0, max_value=200.0, value=160.0, step=0.5)
    waist      = st.number_input("Waist (inch)",         min_value=20.0, max_value=60.0, value=30.0, step=0.5)
    hip        = st.number_input("Hip (inch)",           min_value=25.0, max_value=65.0, value=37.0, step=0.5)
    pulse      = st.number_input("Pulse rate (bpm)",     min_value=40,  max_value=120,  value=72)
    rr         = st.number_input("RR (breaths/min)",     min_value=10,  max_value=40,   value=18)

with col2:
    st.markdown("**🔬 Hormonal Levels**")
    fsh        = st.number_input("FSH (mIU/mL)",         min_value=0.0, max_value=30.0,  value=6.5,  step=0.1)
    lh         = st.number_input("LH (mIU/mL)",          min_value=0.0, max_value=40.0,  value=5.0,  step=0.1)
    fsh_lh     = st.number_input("FSH/LH ratio",         min_value=0.0, max_value=10.0,  value=1.3,  step=0.01)
    amh        = st.number_input("AMH (ng/mL)",          min_value=0.0, max_value=20.0,  value=3.5,  step=0.1)
    tsh        = st.number_input("TSH (mIU/L)",          min_value=0.0, max_value=10.0,  value=2.0,  step=0.1)
    prl        = st.number_input("Prolactin (ng/mL)",    min_value=0.0, max_value=100.0, value=15.0, step=0.5)
    vit_d      = st.number_input("Vitamin D3 (ng/mL)",   min_value=0.0, max_value=100.0, value=25.0, step=0.5)
    prg        = st.number_input("Progesterone (ng/mL)", min_value=0.0, max_value=20.0,  value=0.5,  step=0.1)
    rbs        = st.number_input("RBS (mg/dl)",          min_value=50.0, max_value=300.0, value=90.0, step=1.0)

with col3:
    st.markdown("**🫀 Blood & Ultrasound**")
    hb         = st.number_input("Hb (g/dl)",            min_value=6.0, max_value=18.0, value=12.5, step=0.1)
    cycle_len  = st.number_input("Cycle length (days)",  min_value=20,  max_value=60,   value=28)
    follicle_l = st.number_input("Follicle No. (L)",     min_value=0,   max_value=30,   value=5)
    follicle_r = st.number_input("Follicle No. (R)",     min_value=0,   max_value=30,   value=5)
    avg_f_size_l = st.number_input("Avg. F size (L) mm", min_value=0.0, max_value=30.0, value=10.0, step=0.5)
    avg_f_size_r = st.number_input("Avg. F size (R) mm", min_value=0.0, max_value=30.0, value=10.0, step=0.5)
    endometrium  = st.number_input("Endometrium (mm)",   min_value=0.0, max_value=20.0, value=7.0,  step=0.1)

    st.markdown("**🔘 Symptoms**")
    cycle_ri     = st.selectbox("Cycle (R/I)", options=[2, 4], format_func=lambda x: "Regular" if x==2 else "Irregular")
    pregnant     = st.selectbox("Pregnant",        [0, 1], format_func=lambda x: "Yes" if x else "No")
    abortions    = st.number_input("No. of abortions", min_value=0, max_value=10, value=0)
    weight_gain  = st.selectbox("Weight gain",     [0, 1], format_func=lambda x: "Yes" if x else "No")
    hair_growth  = st.selectbox("Hair growth",     [0, 1], format_func=lambda x: "Yes" if x else "No")
    skin_dark    = st.selectbox("Skin darkening",  [0, 1], format_func=lambda x: "Yes" if x else "No")
    hair_loss    = st.selectbox("Hair loss",       [0, 1], format_func=lambda x: "Yes" if x else "No")
    pimples      = st.selectbox("Pimples",         [0, 1], format_func=lambda x: "Yes" if x else "No")
    fast_food    = st.selectbox("Fast food",       [0, 1], format_func=lambda x: "Yes" if x else "No")
    reg_exercise = st.selectbox("Regular exercise",[0, 1], format_func=lambda x: "Yes" if x else "No")

# ── Build input vector using medians as fallback for any missing feature ─────
def build_input():
    user_values = {
        "Age (yrs)":              age,
        "Weight (Kg)":            weight,
        "Height(Cm) ":            height,
        "BMI":                    bmi,
        "Pulse rate(bpm) ":       pulse,
        "RR (breaths/min)":       rr,
        "Hb(g/dl)":               hb,
        "Cycle(R/I)":             cycle_ri,
        "Cycle length(days)":     cycle_len,
        "Pregnant(Y/N)":          pregnant,
        "No. of abortions":       abortions,
        "FSH(mIU/mL)":            fsh,
        "LH(mIU/mL)":             lh,
        "FSH/LH":                 fsh_lh,
        "Hip(inch)":              hip,
        "Waist(inch)":            waist,
        "Waist:Hip Ratio":        round(waist / hip, 3) if hip > 0 else medians.get("Waist:Hip Ratio", 0.8),
        "TSH (mIU/L)":            tsh,
        "AMH(ng/mL)":             amh,
        "PRL(ng/mL)":             prl,
        "Vit D3 (ng/mL)":         vit_d,
        "PRG(ng/mL)":             prg,
        "RBS(mg/dl)":             rbs,
        "Weight gain(Y/N)":       weight_gain,
        "hair growth(Y/N)":       hair_growth,
        "Skin darkening (Y/N)":   skin_dark,
        "Hair loss(Y/N)":         hair_loss,
        "Pimples(Y/N)":           pimples,
        "Fast food (Y/N)":        fast_food,
        "Reg.Exercise(Y/N)":      reg_exercise,
        "Follicle No. (L)":       follicle_l,
        "Follicle No. (R)":       follicle_r,
        "Avg. F size (L) (mm)":   avg_f_size_l,
        "Avg. F size (R) (mm)":   avg_f_size_r,
        "Endometrium (mm)":       endometrium,
    }
    # Build row: use user value if column matches, else fall back to median
    row = {col: user_values.get(col, medians.get(col, 0)) for col in feature_cols}
    return pd.DataFrame([row])

# ── Predict button ────────────────────────────────────────────────────────────
st.divider()
if st.button("🔍 Run PCOS Prediction", use_container_width=True, type="primary"):
    input_df = build_input()
    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]

    st.divider()
    st.subheader("Result")

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        if prediction == 1:
            st.error("### ⚠️ PCOS Detected")
            st.metric("Confidence", f"{proba[1]*100:.1f}%")
        else:
            st.success("### ✅ No PCOS Detected")
            st.metric("Confidence", f"{proba[0]*100:.1f}%")

    with res_col2:
        st.markdown("**Prediction probability breakdown**")
        prob_df = pd.DataFrame({
            "Outcome": ["No PCOS", "PCOS"],
            "Probability": [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"]
        })
        st.table(prob_df)

    st.caption(
        "⚠️ This tool is for educational purposes only and is **not** a substitute "
        "for clinical diagnosis. Please consult a qualified medical professional."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>Built with Streamlit · Random Forest · Dataset: 2000 patients · "
    "Chitkara University ML Project</small>",
    unsafe_allow_html=True
)