import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("STEP 2: Loading data")
df = pd.read_excel("E Commerce Dataset.xlsx", sheet_name="E Comm")

print("Rows and columns:", df.shape)
print("Churn rate: {:.1f}%".format(df["Churn"].mean() * 100))
print(df.head())

numeric_columns = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

text_columns = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

all_feature_columns = numeric_columns + text_columns

print("\nSTEP 4: Cleaning data")

df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace({"Phone": "Mobile Phone"})
df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
df["PreferedOrderCat"] = df["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})

rows_before = len(df)
df = df.drop_duplicates(subset=all_feature_columns + ["Churn"])
print("Duplicate customers removed:", rows_before - len(df))

print("Missing values before filling:")
print(df.isnull().sum()[df.isnull().sum() > 0])

fill_values = {}
for column in numeric_columns:
    fill_values[column] = float(df[column].median())
for column in text_columns:
    fill_values[column] = df[column].mode()[0]

df = df.fillna(fill_values)
print("Missing values after filling:", df.isnull().sum().sum())

df.to_csv("data.csv", index=False)
print("Cleaned data saved to data.csv")

print("\nSTEP 5: Making EDA charts")
os.makedirs("eda_charts", exist_ok=True)

counts = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts()
plt.figure(figsize=(5, 5))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["#22c55e", "#ef4444"])
plt.title("Churned vs Stayed")
plt.savefig("eda_charts/01_churn_split.png")
plt.close()

chart_list = [
    ("PreferedOrderCat", "Churn Rate by Product Category", "02_category.png"),
    ("PreferredPaymentMode", "Churn Rate by Payment Mode", "03_payment.png"),
    ("CityTier", "Churn Rate by City Tier", "04_city_tier.png"),
    ("PreferredLoginDevice", "Churn Rate by Device", "05_device.png"),
    ("Complain", "Churn Rate: Complaint (1) vs No Complaint (0)", "06_complain.png"),
    ("SatisfactionScore", "Churn Rate by Satisfaction Score", "07_satisfaction.png"),
]

for column, title, file_name in chart_list:
    churn_rate = df.groupby(column)["Churn"].mean() * 100
    plt.figure(figsize=(7, 4))
    sns.barplot(x=churn_rate.index.astype(str), y=churn_rate.values, color="#4f46e5")
    plt.title(title)
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("eda_charts/" + file_name)
    plt.close()

df_temp = df.copy()
df_temp["TenureGroup"] = pd.cut(df_temp["Tenure"], bins=[-1, 3, 12, 1000], labels=["0-3 months", "4-12 months", "1 year+"])
churn_rate = df_temp.groupby("TenureGroup", observed=True)["Churn"].mean() * 100
plt.figure(figsize=(7, 4))
sns.barplot(x=churn_rate.index.astype(str), y=churn_rate.values, color="#4f46e5")
plt.title("Churn Rate by Tenure")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("eda_charts/08_tenure.png")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_columns + ["Churn"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_charts/09_correlation.png")
plt.close()

print("Charts saved in folder: eda_charts")

print("\nSTEP 6: Preparing data for the model")

X = df[all_feature_columns]
y = df["Churn"]

X = pd.get_dummies(X, columns=text_columns).astype(float)
model_columns = list(X.columns)
print("Number of columns after get_dummies:", len(model_columns))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nSTEP 7: Training models")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "SVM": SVC(kernel="rbf", C=10, gamma=0.1, probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3, weights="distance", metric="euclidean"),
    "Gradient Boosting": GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, max_depth=5, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "trained")

print("\nSTEP 8: Testing models")

results = []
for name, model in models.items():
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
    })

    plt.figure(figsize=(4, 4))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("eda_charts/cm_" + name.lower().replace(" ", "_") + ".png")
    plt.close()

results_df = pd.DataFrame(results).round(3)
print(results_df.to_string(index=False))

good_models = results_df[results_df["F1"] >= 0.90]
if len(good_models) == 0:
    good_models = results_df
best_name = good_models.sort_values(["Recall", "F1"], ascending=False).iloc[0]["Model"]
best_model = models[best_name]
print("\nSTEP 9: Best model is", best_name)

if best_name == "Logistic Regression":
    importance = pd.Series(np.abs(best_model.coef_[0]), index=model_columns)
elif best_name in ["SVM", "K-Nearest Neighbors"]:
    permutation = permutation_importance(best_model, X_test, y_test, scoring="f1", n_repeats=5, random_state=42)
    importance = pd.Series(permutation.importances_mean, index=model_columns)
else:
    importance = pd.Series(best_model.feature_importances_, index=model_columns)
importance = importance.sort_values(ascending=False)

top_10 = importance.head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_10.values, y=top_10.index, color="#4f46e5")
plt.title("Top 10 Reasons for Churn (" + best_name + ")")
plt.tight_layout()
plt.savefig("eda_charts/10_feature_importance.png")
plt.close()

print("\nSTEP 10: Training the Core model (mandatory columns only)")

mandatory_columns = [
    "Tenure",
    "CashbackAmount",
    "Complain",
    "PreferedOrderCat",
    "WarehouseToHome",
    "DaySinceLastOrder",
    "NumberOfAddress",
    "PreferredPaymentMode",
]
mandatory_text_columns = ["PreferedOrderCat", "PreferredPaymentMode"]

X_core = pd.get_dummies(df[mandatory_columns], columns=mandatory_text_columns).astype(float)
core_model_columns = list(X_core.columns)

X_core_train, X_core_test, y_core_train, y_core_test = train_test_split(X_core, y, test_size=0.2, random_state=42, stratify=y)
core_scaler = StandardScaler()
X_core_train = core_scaler.fit_transform(X_core_train)
X_core_test = core_scaler.transform(X_core_test)

core_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
core_model.fit(X_core_train, y_core_train)

core_probabilities = core_model.predict_proba(X_core_test)[:, 1]
core_predictions = (core_probabilities >= 0.5).astype(int)
core_results = {
    "Accuracy": round(accuracy_score(y_core_test, core_predictions), 3),
    "Precision": round(precision_score(y_core_test, core_predictions), 3),
    "Recall": round(recall_score(y_core_test, core_predictions), 3),
    "F1": round(f1_score(y_core_test, core_predictions), 3),
    "ROC_AUC": round(roc_auc_score(y_core_test, core_probabilities), 3),
}
print("Core model results:", core_results)

churner_means = df[df["Churn"] == 1][numeric_columns].mean().to_dict()
stayer_means = df[df["Churn"] == 0][numeric_columns].mean().to_dict()
overall_std = df[numeric_columns].std().to_dict()

options = {}
for column in text_columns:
    options[column] = sorted(df[column].unique().tolist())

print("\nSTEP 12: Saving model")

bundle = {
    "model": best_model,
    "model_name": best_name,
    "scaler": scaler,
    "model_columns": model_columns,
    "fill_values": fill_values,
    "options": options,
    "results": results_df,
    "importance": importance,
    "churner_means": churner_means,
    "stayer_means": stayer_means,
    "overall_std": overall_std,
    "mandatory_columns": mandatory_columns,
    "core_model": core_model,
    "core_scaler": core_scaler,
    "core_model_columns": core_model_columns,
    "core_results": core_results,
    "rows_trained": len(df),
}

with open("model.pkl", "wb") as file:
    pickle.dump(bundle, file)
print("Model saved to model.pkl")

sample = df.drop(columns=["Churn"]).sample(200, random_state=1)
sample.to_csv("sample_customers.csv", index=False)
print("Test file saved to sample_customers.csv")

print("\nDONE! Now run:  streamlit run app.py")
