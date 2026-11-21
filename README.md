# Churnio — Retention Hub

AI-based customer churn prediction and retention system for e-commerce

A Streamlit web app for an Amazon/Flipkart-style e-commerce business. It predicts which customers are likely to stop shopping (churn) and recommends a retention action for each customer based on their churn risk.

## Files

| File | What it is |
|---|---|
| `E Commerce Dataset.xlsx` | Original Kaggle dataset |
| `train_model.py` | Cleans data, makes EDA charts, trains 6 models, saves the best one |
| `app.py` | The Streamlit website (Home, Single Customer, Bulk Analysis) |
| `style.html` | Design: CSS, top menu bar logo/status, Home page section |
| `.streamlit/config.toml` | Forces the light theme and hides the Deploy button |
| `requirements.txt` | Libraries to install |
| `model.pkl` | Saved model (created by `train_model.py`) |
| `data.csv` | Cleaned data (created by `train_model.py`) |
| `sample_customers.csv` | 200 customers to test the bulk upload (created by `train_model.py`) |
| `eda_charts/` | EDA, confusion matrix and feature importance charts (created by `train_model.py`) |

## How to run

```
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## What train_model.py does

1. Loads the Excel file (pandas)
2. Cleans it: merges repeated labels (Phone → Mobile Phone, CC → Credit Card, COD → Cash on Delivery, Mobile → Mobile Phone), removes duplicate customers, fills missing values with the median (pandas, numpy)
3. Saves EDA charts (matplotlib, seaborn)
4. Converts text columns to numbers with `pd.get_dummies`, splits 80/20, scales with `StandardScaler` (scikit-learn)
5. Trains Logistic Regression, Decision Tree, Random Forest, SVM (RBF kernel, C=10, gamma=0.1), K-Nearest Neighbors (k=3, distance weights, euclidean) and Gradient Boosting (learning_rate=0.1, 300 trees, max_depth=5) (scikit-learn)
6. Compares Accuracy, Precision, Recall, F1 and ROC-AUC. Picks the best model: among models with F1 ≥ 0.90, the one with the highest Recall (SVM) (scikit-learn)
7. Saves the model, scaler, columns and averages into `model.pkl` (pickle)

## The app

**Single Customer:** fill in one customer's details and get the churn prediction, churn probability, risk category, recommended offer, detailed retention strategy and possible reasons.

**Bulk CSV Upload:** check the columns (8 mandatory + CustomerID, 10 optional), upload a CSV, and get KPI cards, 9 charts, auto-generated business insights, a plan by risk level, the top 20 risky customers and a downloadable results file.

## Mandatory vs optional columns

The 8 mandatory columns are the most important ones by feature importance: Tenure, CashbackAmount, Complain, PreferedOrderCat, WarehouseToHome, DaySinceLastOrder, NumberOfAddress, PreferredPaymentMode.

- All 18 columns in the CSV → the full model (SVM) is used (97.8% accuracy, 94.0% recall)
- Only some optional columns → a smaller Core model (Random Forest) trained on the 8 mandatory columns is used (95.6% accuracy, 88.1% recall). SVM was tested for this too, but with only 8 columns it caught just 63% of churners.

Filling missing columns with average values was tested and rejected: recall dropped to about 54%, so the model missed almost half of the churners.

## Retention plan

| Churn probability | Risk category | Recommended action |
|---|---|---|
| 0–20% | Very Low | Regular loyalty points |
| 21–40% | Low | 5% discount or reward points |
| 41–60% | Medium | 10% discount or free shipping |
| 61–80% | High | 15% discount + personalized offer |
| 81–100% | Critical | 20% discount + priority retention campaign |

Detailed strategy per category:

- **Very Low (0–20%):** no special discount, loyalty points, new product recommendations, encourage repeat purchases
- **Low (21–40%):** 5% discount on the next order, free shipping coupons, personalized product suggestions
- **Medium (41–60%):** 10% discount coupon, cashback offers, special promotional emails, recommendations based on purchase history
- **High (61–80%):** 15% discount, free shipping, personalized comeback offers, contact the customer if there are unresolved complaints
- **Critical (81–100%):** 20% discount, special cashback offers, priority customer support, exclusive comeback campaign, personalized retention communication

## How the insights are made

Insights are not written by AI. pandas `groupby` calculates the churn rate of each group, compares it with the overall churn rate, and if a group is clearly higher, an f-string fills a ready-made sentence with the numbers.

## Dataset

E-commerce Customer Churn Analysis and Prediction, by Ankit Verma, Kaggle.
=======

