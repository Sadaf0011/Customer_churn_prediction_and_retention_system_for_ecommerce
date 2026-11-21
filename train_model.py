# =====================================================================
#  train_model.py  —  MODEL TRAINING
# ---------------------------------------------------------------------
#  This file is run ONLY ONCE:   python train_model.py
#  It reads the data, cleans it, makes charts, trains 6 models,
#  picks the best one and saves it as model.pkl for app.py to use.
#  The code runs from top to bottom, step by step.
# =====================================================================


# ---------------------------------------------------------------------
# STEP 1: Import the libraries we need
# ---------------------------------------------------------------------
import os                         # to create a folder for charts
import pickle                     # to save the trained model in a file

import pandas as pd               # to read and work with tables (DataFrames)
import numpy as np                # for number operations
import matplotlib
matplotlib.use("Agg")             # lets matplotlib save charts without opening a window
import matplotlib.pyplot as plt   # to draw charts
import seaborn as sns             # to draw nicer charts easily

from sklearn.model_selection import train_test_split    # splits data into train and test
from sklearn.preprocessing import StandardScaler        # puts all numbers on the same scale
from sklearn.linear_model import LogisticRegression     # model 1
from sklearn.tree import DecisionTreeClassifier         # model 2
from sklearn.ensemble import RandomForestClassifier     # model 3
from sklearn.svm import SVC                             # model 4 (Support Vector Machine)
from sklearn.neighbors import KNeighborsClassifier      # model 5 (K-Nearest Neighbors)
from sklearn.ensemble import GradientBoostingClassifier # model 6
from sklearn.inspection import permutation_importance  # feature importance for SVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# ---------------------------------------------------------------------
# STEP 2: Load the dataset
# The Kaggle file is an Excel file. The data is in the sheet "E Comm".
# ---------------------------------------------------------------------
print("STEP 2: Loading data")
df = pd.read_excel("E Commerce Dataset.xlsx", sheet_name="E Comm")

print("Rows and columns:", df.shape)
print("Churn rate: {:.1f}%".format(df["Churn"].mean() * 100))   # Churn = 1 means the customer left
print(df.head())


# ---------------------------------------------------------------------
# STEP 3: Make a list of the columns we will use
# Number columns and text columns are handled differently later,
# so we keep them in two separate lists.
# ---------------------------------------------------------------------
numeric_columns = [
    "Tenure",                       # months with the company
    "CityTier",                     # 1, 2 or 3
    "WarehouseToHome",              # delivery distance
    "HourSpendOnApp",               # time spent on app
    "NumberOfDeviceRegistered",     # devices used
    "SatisfactionScore",            # 1 to 5
    "NumberOfAddress",              # saved addresses
    "Complain",                     # 1 = complained, 0 = no complaint
    "OrderAmountHikeFromlastYear",  # % increase in orders
    "CouponUsed",                   # coupons used last month
    "OrderCount",                   # orders last month
    "DaySinceLastOrder",            # days since last order
    "CashbackAmount",               # average cashback
]

text_columns = [
    "PreferredLoginDevice",   # Mobile Phone / Computer
    "PreferredPaymentMode",   # UPI, Debit Card, COD ...
    "Gender",
    "PreferedOrderCat",       # product category
    "MaritalStatus",
]

all_feature_columns = numeric_columns + text_columns


# ---------------------------------------------------------------------
# STEP 4: Clean the data
# ---------------------------------------------------------------------
print("\nSTEP 4: Cleaning data")

# 4a. Some values mean the same thing but are written differently.
#     We replace them so the model sees them as ONE category.
df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace({"Phone": "Mobile Phone"})
df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
df["PreferedOrderCat"] = df["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})

# 4b. Remove duplicate customers.
#     Some rows are exactly the same except for CustomerID.
#     If we keep them, the same customer can be in both train and test,
#     and the accuracy will look higher than it really is.
rows_before = len(df)
df = df.drop_duplicates(subset=all_feature_columns + ["Churn"])
print("Duplicate customers removed:", rows_before - len(df))

# 4c. Fill missing values.
#     Number columns → fill with the median (the middle value).
#     Text columns   → fill with the mode (the most common value).
#     We save these values in fill_values, because app.py will need them too.
print("Missing values before filling:")
print(df.isnull().sum()[df.isnull().sum() > 0])

fill_values = {}
for column in numeric_columns:
    fill_values[column] = float(df[column].median())
for column in text_columns:
    fill_values[column] = df[column].mode()[0]

df = df.fillna(fill_values)
print("Missing values after filling:", df.isnull().sum().sum())

# 4d. Save the cleaned data as a CSV file
df.to_csv("data.csv", index=False)
print("Cleaned data saved to data.csv")


# ---------------------------------------------------------------------
# STEP 5: EDA (Exploratory Data Analysis) — understand the data with charts
# All charts are saved as images in the folder "eda_charts".
# ---------------------------------------------------------------------
print("\nSTEP 5: Making EDA charts")
os.makedirs("eda_charts", exist_ok=True)

# 5a. Pie chart: how many customers left vs stayed
counts = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts()
plt.figure(figsize=(5, 5))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["#22c55e", "#ef4444"])
plt.title("Churned vs Stayed")
plt.savefig("eda_charts/01_churn_split.png")
plt.close()

# 5b. Bar charts: churn rate (%) for each group of a column
#     groupby(column)["Churn"].mean() gives the share of churned customers in each group.
#     We multiply by 100 to get a percentage.
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

# 5c. Tenure groups: new customers vs old customers
#     pd.cut puts each tenure value into a group (0-3, 4-12, 12+ months)
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

# 5d. Correlation heatmap: shows which number columns move together with Churn
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_columns + ["Churn"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_charts/09_correlation.png")
plt.close()

print("Charts saved in folder: eda_charts")


# ---------------------------------------------------------------------
# STEP 6: Prepare the data for the model
# ---------------------------------------------------------------------
print("\nSTEP 6: Preparing data for the model")

# 6a. X = input columns (features), y = what we want to predict (Churn)
X = df[all_feature_columns]
y = df["Churn"]

# 6b. Models only understand numbers, not text.
#     pd.get_dummies turns each text value into its own 0/1 column.
#     Example: Gender → Gender_Male (1 or 0) and Gender_Female (1 or 0)
X = pd.get_dummies(X, columns=text_columns).astype(float)
model_columns = list(X.columns)      # we save this list so app.py makes the same columns
print("Number of columns after get_dummies:", len(model_columns))

# 6c. Split the data: 80% to train the model, 20% to test it.
#     stratify=y keeps the same churn % in both parts.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))

# 6d. Scale the numbers so big values (like cashback = 300) don't
#     dominate small values (like complain = 1).
#     fit_transform on train data, only transform on test data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------------------------
# STEP 7: Train 6 models
# class_weight="balanced" tells the model to pay extra attention to churners,
# because only about 17% of customers churned (the data is imbalanced).
# ---------------------------------------------------------------------
print("\nSTEP 7: Training models")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),

    # SVM with an RBF kernel (it can learn curved boundaries between churners and non-churners)
    #   C=10      → how strictly it tries to classify every training customer correctly
    #   gamma=0.1 → how far the influence of one customer reaches (small = far, big = close)
    #   probability=True → lets the SVM give a churn probability (needed for risk levels and coupons)
    "SVM": SVC(kernel="rbf", C=10, gamma=0.1, probability=True, random_state=42),

    # KNN: looks at the 3 most similar customers and copies their answer
    #   n_neighbors=3      → use the 3 closest customers
    #   weights="distance" → closer customers count more
    #   metric="euclidean" → normal straight-line distance
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3, weights="distance", metric="euclidean"),

    # Gradient Boosting: builds many small trees, each one fixing the mistakes of the previous ones
    #   learning_rate=0.1 → how much each new tree is allowed to change the result
    #   n_estimators=300  → number of trees
    #   max_depth=5       → how deep each tree can grow
    "Gradient Boosting": GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, max_depth=5, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "trained")


# ---------------------------------------------------------------------
# STEP 8: Test and compare the models
#   Accuracy  = how many predictions were correct overall
#   Precision = of the customers we said will churn, how many really churned
#   Recall    = of the customers who really churned, how many we caught
#   F1        = balance of precision and recall
#   ROC_AUC   = how well the model separates churners from non-churners (1.0 = perfect)
# ---------------------------------------------------------------------
print("\nSTEP 8: Testing models")

results = []
for name, model in models.items():
    probabilities = model.predict_proba(X_test)[:, 1]      # chance of churn (0 to 1)
    predictions = (probabilities >= 0.5).astype(int)       # 1 if chance is 50% or more (same rule as app.py)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
    })

    # Confusion matrix chart: correct and wrong predictions for each model
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


# ---------------------------------------------------------------------
# STEP 9: Pick the best model
# Our rule has 2 parts:
#   1. Keep only the good models: F1 score of 0.90 or more
#      (F1 checks that the model is good at BOTH finding churners and not raising false alarms)
#   2. From those, pick the one with the highest Recall
#      (Recall = how many real churners we catch. For a business, missing a customer
#       who is about to leave costs more than giving an extra coupon.)
# ---------------------------------------------------------------------
good_models = results_df[results_df["F1"] >= 0.90]
if len(good_models) == 0:                     # safety: if no model reaches 0.90, use all of them
    good_models = results_df
best_name = good_models.sort_values(["Recall", "F1"], ascending=False).iloc[0]["Model"]
best_model = models[best_name]
print("\nSTEP 9: Best model is", best_name)

# Feature importance: which columns the model used the most.
# Tree models have feature_importances_, Logistic Regression has coef_.
# SVM and KNN have neither, so we use permutation importance:
#   shuffle one column at a time and see how much the F1 score drops.
#   A big drop means the model depends a lot on that column.
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


# ---------------------------------------------------------------------
# STEP 10: Train a second, smaller "Core" model
# Real businesses may not have all 18 columns. So we also train a model
# that uses ONLY 8 important columns (Tenure, Cashback, Complain ... chosen by Random Forest importance).
# These 8 columns become MANDATORY in the app's CSV upload. The other 10 are optional.
#   - CSV has all 18 columns → app uses the full model (STEP 7-9)
#   - CSV has only some     → app uses this Core model
# ---------------------------------------------------------------------
print("\nSTEP 10: Training the Core model (mandatory columns only)")

mandatory_columns = [
    "Tenure",                 # most important: new customers leave more
    "CashbackAmount",
    "Complain",
    "PreferedOrderCat",
    "WarehouseToHome",
    "DaySinceLastOrder",
    "NumberOfAddress",
    "PreferredPaymentMode",
]
mandatory_text_columns = ["PreferedOrderCat", "PreferredPaymentMode"]    # the text ones among them

# 10a. Same preparation as STEP 6, but only with the mandatory columns
X_core = pd.get_dummies(df[mandatory_columns], columns=mandatory_text_columns).astype(float)
core_model_columns = list(X_core.columns)

# Same split settings (random_state=42) so it is tested on the same customers
X_core_train, X_core_test, y_core_train, y_core_test = train_test_split(X_core, y, test_size=0.2, random_state=42, stratify=y)
core_scaler = StandardScaler()
X_core_train = core_scaler.fit_transform(X_core_train)
X_core_test = core_scaler.transform(X_core_test)

# 10b. Train a Random Forest on these columns.
#      Why not SVM here? We tested it: with only 8 columns the SVM caught just 63% of churners,
#      but the Random Forest caught 88%. So SVM is best with all columns, Random Forest with few columns.
core_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
core_model.fit(X_core_train, y_core_train)

# 10c. Test it with the same metrics as STEP 8
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


# ---------------------------------------------------------------------
# STEP 11: Collect extra information that app.py needs
# ---------------------------------------------------------------------

# 11a. Average of each number column for churned and stayed customers.
#      app.py uses these to show "possible reasons" for one customer.
churner_means = df[df["Churn"] == 1][numeric_columns].mean().to_dict()
stayer_means = df[df["Churn"] == 0][numeric_columns].mean().to_dict()
overall_std = df[numeric_columns].std().to_dict()

# 11b. All options for each text column (used in the app's dropdowns)
options = {}
for column in text_columns:
    options[column] = sorted(df[column].unique().tolist())


# ---------------------------------------------------------------------
# STEP 12: Save everything into ONE file: model.pkl
# A dictionary lets us keep the model and all its helpers together.
# ---------------------------------------------------------------------
print("\nSTEP 12: Saving model")

bundle = {
    "model": best_model,              # the trained model
    "model_name": best_name,          # its name, shown in the app
    "scaler": scaler,                 # the same scaler must be used in the app
    "model_columns": model_columns,   # column order the model expects
    "fill_values": fill_values,       # values to fill missing data in uploaded CSVs
    "options": options,               # dropdown options
    "results": results_df,            # comparison table, shown in the app
    "importance": importance,         # feature importance
    "churner_means": churner_means,
    "stayer_means": stayer_means,
    "overall_std": overall_std,
    "mandatory_columns": mandatory_columns,      # the 8 must-have columns
    "core_model": core_model,                    # the smaller model (STEP 10)
    "core_scaler": core_scaler,
    "core_model_columns": core_model_columns,
    "core_results": core_results,
    "rows_trained": len(df),                     # number of customers after cleaning
}

with open("model.pkl", "wb") as file:     # "wb" = write in binary mode
    pickle.dump(bundle, file)
print("Model saved to model.pkl")


# ---------------------------------------------------------------------
# STEP 13: Make a small test file for the app's Bulk CSV Upload section
# 200 random customers, without the Churn column (because the app predicts it)
# ---------------------------------------------------------------------
sample = df.drop(columns=["Churn"]).sample(200, random_state=1)
sample.to_csv("sample_customers.csv", index=False)
print("Test file saved to sample_customers.csv")

print("\nDONE! Now run:  streamlit run app.py")
