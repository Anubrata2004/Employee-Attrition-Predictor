#  Employee Attrition Predictor

A machine learning project that predicts which employees are likely to quit — and explains why — so HR can act early.

##  Live Demo
Run locally with Streamlit

##  Problem Statement
Companies lose ₹5–15 lakh replacing one employee. HR teams can't predict who will quit until it's too late. This model flags at-risk employees early.

## Key Findings
- Overtime employees are **3x more likely** to quit
- Low income earners (avg ₹4,787) vs stayers (avg ₹6,833)
- Most employees quit in the **first 0–5 years**
- Low job satisfaction = **22.8% attrition** vs 11.3%

##  Model Results
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 80% | 0.718 |
| Random Forest | 80% | 0.738 |
| **XGBoost** | **83%** | **0.741** |

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- SMOTE (imbalanced-learn)
- SHAP (Explainability)
- Streamlit (App)
- Matplotlib, Seaborn

##  How to Run
```bash
# Clone the repo
git clone https://github.com/Anubrata2004/Employee-Attrition-Predictor.git
cd Employee-Attrition-Predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
