# heart-disease-prediction-ml
End-to-end heart disease prediction system built with Python, machine learning models, and explainable AI for real-world medical risk analysis.
# ❤️ End-to-End Heart Disease Risk Prediction System

## 📌 Overview

Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve treatment outcomes and save lives. This project presents an **end-to-end machine learning system** that predicts the risk of heart disease based on patient clinical data.

The project goes beyond basic model training and focuses on **real-world ML practices**, including advanced data preprocessing, training multiple models, model comparison, explainability (XAI), and deployment readiness.

---

## 🎯 Problem Statement

To build a robust machine learning system that can accurately predict whether a patient is at risk of heart disease using clinical and demographic features, while ensuring the model is interpretable and reliable for healthcare use.

---

## 📊 Dataset Description

### 🔹 Data Sources

The dataset is created by combining multiple heart disease datasets from the **UCI Machine Learning Repository**:

* Cleveland
* Hungarian
* Switzerland
* Long Beach VA

These datasets share common clinical features and were merged to improve **data diversity and generalization**.

### 🔹 Dataset Size

* Total Records: ~1190
* Features: 13 clinical attributes
* Target Variable:

  * `0` → No Heart Disease
  * `1` → Heart Disease

### 🔹 Key Features

* Age
* Sex
* Chest Pain Type (cp)
* Resting Blood Pressure (trestbps)
* Serum Cholesterol (chol)
* Fasting Blood Sugar (fbs)
* Resting ECG (restecg)
* Maximum Heart Rate (thalach)
* Exercise Induced Angina (exang)

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

* Missing value handling using:

  * Median Imputation
  * KNN Imputer (for comparison)
* Feature scaling comparison:

  * StandardScaler
  * MinMaxScaler
* Stratified train-test split to maintain class balance

A preprocessing pipeline was built to ensure consistency and reusability.

---

## 🤖 Machine Learning Models

Multiple models were trained and evaluated:

| Model                        | Purpose                         |
| ---------------------------- | ------------------------------- |
| Logistic Regression          | Baseline linear model           |
| Random Forest                | Strong tree-based ensemble      |
| Support Vector Machine (SVM) | Non-linear decision boundaries  |
| Gradient Boosting            | High-performance ensemble model |

---

## 📈 Model Evaluation & Comparison

Models were evaluated using multiple metrics relevant to medical classification:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### 🔹 Performance Summary 

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC  |
| ------------------- | -------- | --------- | ------ | -------- | -------- |
| Logistic Regression | 0.85     | 0.83      | 0.81   | 0.82     | 0.88     |
| Random Forest       | **0.91** | 0.89      | 0.93   | 0.91     | **0.95** |
| SVM                 | 0.88     | 0.86      | 0.87   | 0.86     | 0.90     |
| Gradient Boosting   | 0.90     | 0.88      | 0.92   | 0.90     | 0.94     |

📌 **Random Forest** was selected as the final model based on the highest ROC-AUC score.

---

## 🧠 Model Explainability (XAI)

To make the model interpretable and suitable for healthcare applications, explainability techniques were added.

### 🔹 SHAP (SHapley Additive exPlanations)

* Identifies the most influential features contributing to predictions
* Explains both global model behavior and individual predictions
* Helps understand how clinical parameters impact heart disease risk

SHAP summary plots highlight features such as:

* Age
* Chest pain type
* Maximum heart rate
* Cholesterol

### 🔹 Permutation Importance

* Model-agnostic technique
* Validates feature importance by measuring performance drop after feature shuffling
* Used to cross-check SHAP results

📊 Explainability plots are saved in the `images/` directory.

---

## 🗂️ Project Structure

```
Heart-Disease-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocess.py
│   ├── train_models.py
│   ├── explainability.py
│   └── predict.py
├── models/
│   └── best_model.pkl
├── images/
│   ├── shap_summary.png
│   └── permutation_importance.png
├── app.py
├── requirements.txt
└── README.md
```

---

## 🌐 Deployment 

The trained model can be deployed using:

* **Streamlit** for interactive web application
* **Flask** for REST API integration

Users can input clinical details and receive real-time heart disease risk predictions.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* SHAP
* Matplotlib
* Streamlit / Flask (optional)

---

## 🚀 How to Run the Project

```bash
git clone <your-repo-link>
cd Heart-Disease-Prediction
pip install -r requirements.txt
python src/train_models.py
```

(Optional – Web App)

```bash
streamlit run app.py
```

---

## 📌  Highlights

* Built an end-to-end heart disease prediction system using machine learning
* Trained and compared multiple ML models with advanced evaluation metrics
* Applied explainable AI (SHAP, permutation importance) for medical interpretability
* Designed modular, production-ready ML pipeline

---

## 🙏 Acknowledgements

* UCI Machine Learning Repository for the heart disease datasets
* Original inspiration from open-source ML projects

---

## 👤 Author

**Gourav Sharma**
Machine Learning & Data Science Enthusiast

---

⭐ If you find this project useful, feel free to star the repository!
