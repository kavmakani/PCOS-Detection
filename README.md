# 🩺 PCOS Detection System

A machine learning web app for early detection of **Polycystic Ovary Syndrome (PCOS)** using clinical parameters.

🔗 **Live App:** https://pcos-detection.streamlit.app/

## 📌 About
PCOS affects 8–13% of women globally and is chronically underdiagnosed.
This tool uses ML to assist in early detection using 44 medical features.

## 🤖 Model Performance
| Algorithm | Accuracy |
|-----------|----------|
| Random Forest | 99.8% ✅ |
| Decision Tree | 99.1% |
| Logistic Regression | 94.2% |
| SVM | 96.3% |
| KNN | 97.1% |

## 📊 Dataset
- 2000 patient records
- 44 clinical features (hormonal, physical, ultrasound)
- Target: PCOS (Y/N)

## 🛠️ Tech Stack
- Python, Streamlit, Scikit-learn, Pandas

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Disclaimer
This tool is for educational purposes only and is not a substitute for clinical diagnosis.