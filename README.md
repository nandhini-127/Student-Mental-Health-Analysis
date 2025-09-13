# 🧠 Student Mental Health Prediction System

This project is a **Machine Learning-based classification system** that predicts the **mental health risk level of students** based on various lifestyle, psychological, and demographic factors.  

It uses **Random Forest, XGBoost, and Voting Classifier (Ensemble Model)** to classify students into **Low Risk, Medium Risk, or High Risk**, and also provides tailored recommendations.

---

## 📊 Features

- Predicts **student mental health risk levels**:
  - **Low Risk** 🟢 → Healthy lifestyle, maintain balance.  
  - **Medium Risk** 🟡 → Early signs of stress, adopt preventive measures.  
  - **High Risk** 🔴 → Immediate attention required, seek counseling.  

- Uses **ensemble learning (Voting Classifier)** with:
  - **Random Forest Classifier (with SMOTE for imbalance handling)**
  - **XGBoost Classifier (with hyperparameter tuning)**

- **Preprocessing pipeline**:
  - Numerical features → `StandardScaler`
  - Categorical features → `OneHotEncoder`
  - Oversampling with `SMOTE`

- **Custom engineered features**:
  - `mental_score = Stress + Depression + Anxiety`
  - `healthy_lifestyle = Sleep Hours - Headache + Physical Activity Score`

- **User interactive mode**:
  - Collects student details via console input  
  - Predicts risk level and gives **personalized recommendations**

---

## ⚙️ Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Imbalanced-learn (SMOTE)**
- **Matplotlib** (for visualizations)

---

## 🚀 How It Works

1. **Load Dataset**  
   A student dataset containing psychological, lifestyle, and demographic features.

2. **Data Preprocessing & Feature Engineering**  
   - Missing values handled  
   - Standardization & One-Hot Encoding applied  
   - New features created (`mental_score`, `healthy_lifestyle`)

3. **Model Training**  
   - Train-test split (80:20)  
   - Random Forest and XGBoost tuned using `GridSearchCV`  
   - Combined using **Voting Classifier**  

4. **Evaluation Metrics**  
   - Accuracy  
   - F1-score (macro)  
   - Confusion Matrix  
   - Classification Report  

5. **Prediction & Recommendations**  
   - Takes user input from console  
   - Predicts risk level  
   - Provides **recommendations based on risk category**

---

## 📈 Example Output

**XGBoost Confusion Matrix:**
- [[50 2 1]
- [ 4 45 6]
- [ 0 5 47]]

**XGBoost Classification Report:**
- precision recall f1-score support
- Low Risk 0.92 0.95 0.93 53
- Medium Risk 0.86 0.81 0.83 55
- High Risk 0.88 0.90 0.89 52

---

## 👩‍⚕️ Recommendations

- **Low Risk:** Keep up the good work! Continue maintaining a balanced lifestyle.  
- **Medium Risk:** Take proactive steps now—small changes can prevent bigger issues later.  
- **High Risk:** Seek immediate support. Consider consulting a counselor or mental health professional.  

---

## 📌 Future Enhancements

- Deploy as a **Flask / Streamlit web app**  
- Add **real-time student surveys** integration  
- Enhance feature set with **social, academic, and environmental factors**  
- Implement **deep learning models (LSTMs for temporal trends)**  

---

## 🏆 Conclusion

This project demonstrates how **Machine Learning and AI** can be used to analyze **student mental health** patterns and provide **early intervention strategies**. It serves as a **decision-support system** for students, teachers, and counselors.  

---
