

1️⃣ Importing Libraries and Loading Dataset
import pandas as pd
Imports the Pandas library, which is used for data manipulation and analysis.

df = pd.read_csv("C:\\Users\\Aspire...heart-disease-dataset.csv")
Loads the heart disease dataset from a CSV file into a Pandas DataFrame named df.

print(df.shape)
Displays the number of rows and columns in the dataset.

print(df.head())
Displays the first five rows of the dataset to understand its structure.

2️⃣ Checking Missing Values
df.isna().sum()
Checks for missing (NaN) values in each column.

Helps decide how missing data should be handled.

3️⃣ Handling Missing Values using Mean Imputation
from sklearn.impute import SimpleImputer
Imports SimpleImputer from Scikit-Learn for missing value handling.

imputer = SimpleImputer(strategy="mean")
Creates an imputer that replaces missing values with the mean of each column.

df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
Applies the imputer to the dataset and converts it back to a DataFrame.

4️⃣ Handling Missing Values using KNN Imputer
from sklearn.impute import KNNImputer
Imports K-Nearest Neighbors Imputer.

imputer = KNNImputer(n_neighbors=5)
Creates a KNN imputer using 5 nearest neighbors.

df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
Fills missing values based on similarity between rows.

5️⃣ Splitting Features and Target Variable
X = df.drop(columns=['target'])
Separates input features (independent variables).

y = df['target']
Stores the target variable (presence or absence of heart disease).

6️⃣ Feature Scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
Imports StandardScaler for standardization.

scaler = StandardScaler()
Initializes the scaler.

X_scaled_std = scaler.fit_transform(X)
Scales features so they have mean = 0 and standard deviation = 1.

7️⃣ Feature Scaling using Min-Max Scaler
from sklearn.preprocessing import MinMaxScaler
Imports Min-Max scaling.

scaler = MinMaxScaler()
Initializes the scaler.

X_scaled_mm = scaler.fit_transform(X)
Scales features to a range between 0 and 1.

8️⃣ Train-Test Split
from sklearn.model_selection import train_test_split
Imports function to split data.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
Splits dataset into 80% training and 20% testing

stratify=y ensures class balance

random_state=42 ensures reproducibility

9️⃣ Importing Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
Imports different classification algorithms.

🔟 Creating Model Dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Gradient Boosting": GradientBoostingClassifier()
}
Stores multiple ML models in a dictionary for easy comparison.

1️⃣1️⃣ Model Evaluation Function
def evaluate_model(model, X_test, y_test):
Defines a function to evaluate models.

y_pred = model.predict(X_test)
Predicts class labels.

y_prob = model.predict_proba(X_test)[:, 1]
Predicts probability scores for ROC-AUC calculation.

return {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}
Returns performance metrics.

1️⃣2️⃣ Evaluating All Models
results = []
Empty list to store results.

for model_name, model in models.items():
Iterates through each model.

model.fit(X_train, y_train)
Trains the model.

metrics = evaluate_model(model, X_test, y_test)
Evaluates the model.

metrics["Model"] = model_name
results.append(metrics)
Stores metrics with model name.

1️⃣3️⃣ Displaying Results
results_df = pd.DataFrame(results)
Converts results into a DataFrame.

results_df = results_df.set_index("Model")
print(results_df)
Displays performance comparison.

1️⃣4️⃣ Selecting Best Model
best_model_name = results_df["ROC-AUC"].idxmax()
Selects model with highest ROC-AUC score.

print("Best Model:", best_model_name)
Prints best model name.

1️⃣5️⃣ Saving the Best Model
import joblib
import os
Imports libraries for saving models.

os.makedirs("models", exist_ok=True)
Creates directory if not present.

best_model = models[best_model_name]
joblib.dump(best_model, "models/best_model.pkl")
Saves trained model as a .pkl file.

1️⃣6️⃣ Cross-Validation
from sklearn.model_selection import cross_val_score
Imports cross-validation function.

scores = cross_val_score(
    best_model, X, y, cv=5, scoring="roc_auc"
)
Performs 5-fold cross-validation.

print("Mean ROC-AUC:", scores.mean())
Prints average performance score.

1️⃣7️⃣ Hyperparameter Tuning (Grid Search)
from sklearn.model_selection import GridSearchCV
Imports Grid Search.

grid = GridSearchCV(
    GradientBoostingClassifier(),
    param_grid,
    cv=5
)
Creates grid search object.

grid.fit(X_train, y_train)
Finds best hyperparameters.

1️⃣8️⃣ Model Explainability using SHAP
import shap
import matplotlib.pyplot as plt
Imports SHAP library for interpretability.

explainer = shap.TreeExplainer(best_model)
Creates SHAP explainer.

shap_values = explainer.shap_values(X_test)
Computes SHAP values.

shap.summary_plot(shap_values, X_test, plot_type="bar")
Displays feature importance.

1️⃣9️⃣ Permutation Feature Importance
from sklearn.inspection import permutation_importance
Imports permutation importance.

result = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc"
)
Calculates feature importance.

plt.barh(features, importances)
plt.title("Permutation Feature Importance")
plt.show()
Displays feature importance graph.

Conclusion

This notebook:

Preprocesses data

Trains multiple ML models

Evaluates performance

Selects the best model

Saves the trained model

Explains predictions using SHAP







No file chosenNo file chosen
ChatGPT can make mistakes. Check important info. See Cookie Preferences.
