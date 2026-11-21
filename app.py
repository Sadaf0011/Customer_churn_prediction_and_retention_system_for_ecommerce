
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Churnio | Retention Hub", layout="wide", initial_sidebar_state="collapsed")

sns.set_theme(style="whitegrid")
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

PRIMARY = "#4f46e5"
risk_colors = {"Critical": "#991b1b", "High": "#dc2626", "Medium": "#ea580c", "Low": "#ca8a04", "Very Low": "#16a34a"}

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

bundle = load_model()

with open("style.html", encoding="utf-8") as file:
    design_parts = file.read().split("<!-- SPLIT -->")

css_part = design_parts[0]
logo_part = design_parts[1]
status_part = design_parts[2]
home_part = design_parts[3]

best_row = bundle["results"].set_index("Model").loc[bundle["model_name"]]
placeholders = {
    "[[MODEL_NAME]]": bundle["model_name"],
    "[[ACCURACY]]": f"{best_row['Accuracy'] * 100:.1f}",
    "[[RECALL]]": f"{best_row['Recall'] * 100:.1f}",
    "[[F1]]": f"{best_row['F1']:.2f}",
    "[[AUC]]": f"{best_row['ROC_AUC']:.2f}",
    "[[GAUGE_DEG]]": f"{best_row['Accuracy'] * 180:.0f}",
    "[[CUSTOMERS]]": f"{bundle.get('rows_trained', 0):,}",
    "[[CORE_ACCURACY]]": f"{bundle['core_results']['Accuracy'] * 100:.1f}",
}

for key, value in placeholders.items():
    status_part = status_part.replace(key, value)
    home_part = home_part.replace(key, value)

st.markdown(css_part, unsafe_allow_html=True)

nav_left, nav_middle, nav_right = st.columns([2, 4, 2])

with nav_left:
    st.markdown(logo_part, unsafe_allow_html=True)

with nav_middle:
    page = st.radio(
        "Menu",
        ["Home", "Single Customer", "Bulk Analysis"],
        horizontal=True,
        label_visibility="collapsed",
        key="page",
    )

with nav_right:
    st.markdown(status_part, unsafe_allow_html=True)

numeric_columns = [
    "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
    "Complain", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount",
]

text_columns = ["PreferredLoginDevice", "PreferredPaymentMode", "Gender", "PreferedOrderCat", "MaritalStatus"]
all_feature_columns = numeric_columns + text_columns

mandatory_columns = ["CustomerID"] + bundle["mandatory_columns"]
optional_columns = [column for column in all_feature_columns if column not in mandatory_columns]
all_upload_columns = ["CustomerID"] + all_feature_columns

column_meaning = {
    "CustomerID": "Unique customer ID",
    "Tenure": "Months with the company",
    "CityTier": "City tier (1, 2 or 3)",
    "WarehouseToHome": "Distance from warehouse to home (km)",
    "HourSpendOnApp": "Hours spent on app",
    "NumberOfDeviceRegistered": "Devices registered",
    "SatisfactionScore": "Satisfaction score (1 to 5)",
    "NumberOfAddress": "Saved addresses",
    "Complain": "Complaint last month (1 = Yes, 0 = No)",
    "OrderAmountHikeFromlastYear": "Order amount increase from last year (%)",
    "CouponUsed": "Coupons used last month",
    "OrderCount": "Orders last month",
    "DaySinceLastOrder": "Days since last order",
    "CashbackAmount": "Average cashback (Rs)",
    "PreferredLoginDevice": "Mobile Phone / Computer",
    "PreferredPaymentMode": "Debit Card, Credit Card, UPI, E wallet, Cash on Delivery",
    "Gender": "Male / Female",
    "PreferedOrderCat": "Laptop & Accessory, Mobile Phone, Fashion, Grocery, Others",
    "MaritalStatus": "Single / Married / Divorced",
}

risk_order = ["Critical", "High", "Medium", "Low", "Very Low"]

risk_ranges = {
    "Very Low": "0-20%",
    "Low": "21-40%",
    "Medium": "41-60%",
    "High": "61-80%",
    "Critical": "81-100%",
}

recommended_action = {
    "Very Low": "Regular loyalty points",
    "Low": "5% discount or reward points",
    "Medium": "10% discount or free shipping",
    "High": "15% discount + personalized offer",
    "Critical": "20% discount + priority retention campaign",
}

short_offer = {
    "Very Low": "Loyalty points",
    "Low": "5% off",
    "Medium": "10% off",
    "High": "15% off",
    "Critical": "20% off",
}

retention_strategy = {
    "Very Low": [
        "No special discount required.",
        "Offer loyalty points.",
        "Send new product recommendations.",
        "Encourage repeat purchases.",
    ],
    "Low": [
        "5% discount on the next order.",
        "Free shipping coupons.",
        "Personalized product suggestions.",
    ],
    "Medium": [
        "10% discount coupon.",
        "Cashback offers.",
        "Special promotional emails.",
        "Recommendations based on purchase history.",
    ],
    "High": [
        "15% discount.",
        "Free shipping.",
        "Personalized comeback offers.",
        "Contact customer if there are unresolved complaints.",
    ],
    "Critical": [
        "20% discount.",
        "Special cashback offers.",
        "Priority customer support.",
        "Exclusive comeback campaign.",
        "Personalized retention communication.",
    ],
}

def strategy_card(risk):
    steps = "<br>".join("&bull; " + step for step in retention_strategy[risk])
    return (
        f'<div class="sk-insight" style="border-left-color: {risk_colors[risk]};">'
        f'<div class="sk-insight-text">{risk} risk &middot; {risk_ranges[risk]}</div>'
        f'<div class="sk-insight-action">{steps}</div></div>'
    )

def predict(customers, use_core=False):

    if use_core:
        model = bundle["core_model"]
        scaler = bundle["core_scaler"]
        model_columns = bundle["core_model_columns"]
    else:
        model = bundle["model"]
        scaler = bundle["scaler"]
        model_columns = bundle["model_columns"]

    data = customers.copy()
    for column in all_feature_columns:
        if column not in data.columns:
            data[column] = None
    data = data[all_feature_columns]

    data["PreferredLoginDevice"] = data["PreferredLoginDevice"].replace({"Phone": "Mobile Phone"})
    data["PreferredPaymentMode"] = data["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
    data["PreferedOrderCat"] = data["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.fillna(bundle["fill_values"])

    X = pd.get_dummies(data, columns=text_columns)
    X = X.reindex(columns=model_columns, fill_value=0).astype(float)

    X = scaler.transform(X)
    probability = model.predict_proba(X)[:, 1]

    result = customers.copy()
    result["Churn_Probability"] = probability.round(3)
    result["Churn_Prediction"] = (probability >= 0.5).astype(int)

    percent = (probability * 100).round()
    result["Risk_Level"] = pd.cut(percent, bins=[-1, 20, 40, 60, 80, 100], labels=["Very Low", "Low", "Medium", "High", "Critical"]).astype(str)
    result["Recommended_Action"] = result["Risk_Level"].map(recommended_action)
    return result

def churn_rate_chart(table, column, title):
    churn_rate = (table.groupby(column, observed=True)["Churn_Prediction"].mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sns.barplot(x=churn_rate.index.astype(str), y=churn_rate.values, color=PRIMARY, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Churn rate (%)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

if page == "Home":

    st.markdown(home_part, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model comparison")
        st.caption("All 6 models were tested on the same 20% of customers they never saw during training.")
        st.dataframe(bundle["results"].set_index("Model"))
        st.caption(
            f"Core model (only the 8 mandatory columns): accuracy {bundle['core_results']['Accuracy'] * 100:.1f}%, "
            f"recall {bundle['core_results']['Recall'] * 100:.1f}%. Used when an uploaded file has missing optional columns."
        )

    with col2:
        st.subheader("Churn percentage and recommended offers")
        st.caption("Each customer gets one recommended action based on how likely they are to leave.")
        low_to_high = list(reversed(risk_order))
        rules = pd.DataFrame({
            "Churn probability": [risk_ranges[r] for r in low_to_high],
            "Risk category": low_to_high,
            "Recommended action": [recommended_action[r] for r in low_to_high],
        })
        st.dataframe(rules, hide_index=True)

    st.subheader("Detailed retention strategy")
    strategy_columns = st.columns(5)
    for column_box, risk in zip(strategy_columns, reversed(risk_order)):
        with column_box:
            st.markdown(strategy_card(risk), unsafe_allow_html=True)

elif page == "Single Customer":

    st.title("Single Customer Prediction")
    st.write("Enter one customer's details to check if they are likely to leave.")
    options = bundle["options"]

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Orders**")
            tenure = st.number_input("Tenure (months)", 0, 100, 10)
            order_count = st.number_input("Orders last month", 0, 50, 2)
            days_since = st.number_input("Days since last order", 0, 100, 3)
            coupon_used = st.number_input("Coupons used last month", 0, 50, 1)
            cashback = st.number_input("Average cashback (Rs)", 0.0, 1000.0, 150.0)
            order_hike = st.number_input("Order amount hike from last year (%)", 0, 100, 15)

        with col2:
            st.markdown("**Experience**")
            satisfaction = st.slider("Satisfaction score", 1, 5, 3)
            complain = st.radio("Complaint last month?", ["No", "Yes"], horizontal=True)
            hours_app = st.number_input("Hours spent on app", 0, 24, 3)
            devices = st.number_input("Devices registered", 1, 10, 3)
            addresses = st.number_input("Saved addresses", 1, 30, 3)
            distance = st.number_input("Warehouse to home (km)", 1, 200, 15)

        with col3:
            st.markdown("**Profile**")
            city_tier = st.selectbox("City tier", [1, 2, 3])
            category = st.selectbox("Preferred category", options["PreferedOrderCat"])
            payment = st.selectbox("Payment mode", options["PreferredPaymentMode"])
            device = st.selectbox("Login device", options["PreferredLoginDevice"])
            gender = st.selectbox("Gender", options["Gender"])
            marital = st.selectbox("Marital status", options["MaritalStatus"])

        submitted = st.form_submit_button("Predict churn")

    if not submitted:
        st.stop()

    customer = pd.DataFrame([{
        "Tenure": tenure,
        "CityTier": city_tier,
        "WarehouseToHome": distance,
        "HourSpendOnApp": hours_app,
        "NumberOfDeviceRegistered": devices,
        "SatisfactionScore": satisfaction,
        "NumberOfAddress": addresses,
        "Complain": 1 if complain == "Yes" else 0,
        "OrderAmountHikeFromlastYear": order_hike,
        "CouponUsed": coupon_used,
        "OrderCount": order_count,
        "DaySinceLastOrder": days_since,
        "CashbackAmount": cashback,
        "PreferredLoginDevice": device,
        "PreferredPaymentMode": payment,
        "Gender": gender,
        "PreferedOrderCat": category,
        "MaritalStatus": marital,
    }])

    result = predict(customer).iloc[0]
    probability = result["Churn_Probability"]
    risk = result["Risk_Level"]

    st.divider()
    st.subheader("Result")

    if result["Churn_Prediction"] == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")

    st.markdown(f'<span class="sk-badge sk-badge-{risk.replace(" ", "")}">{risk} risk</span>', unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn probability", f"{probability * 100:.1f}%")
    col2.metric("Risk level", risk)
    col3.metric("Recommended offer", short_offer[risk])
    st.progress(int(round(probability * 100)))

    st.subheader("Recommended action")
    st.info(recommended_action[risk])
    st.subheader("Retention strategy")
    st.markdown(strategy_card(risk), unsafe_allow_html=True)

    if risk != "Very Low":
        st.subheader("Possible reasons")
        reasons = []
        for column in numeric_columns:
            left_avg = bundle["churner_means"][column]
            stay_avg = bundle["stayer_means"][column]
            spread = bundle["overall_std"][column]

            if spread == 0 or abs(left_avg - stay_avg) / spread < 0.15:
                continue

            middle = (left_avg + stay_avg) / 2
            value = float(result[column])

            if left_avg > stay_avg:
                closer_to_left = value > middle
            else:
                closer_to_left = value < middle

            if closer_to_left:
                reasons.append(f"**{column}** is {value:g} (customers who left: avg {left_avg:.1f}, customers who stayed: avg {stay_avg:.1f})")

        if reasons:
            for reason in reasons:
                st.markdown("- " + reason)
        else:
            st.write("No single value stands out. The risk comes from the combination of details.")

else:

    st.title("Bulk Analysis")
    st.write("Upload a CSV of many customers to see how many may leave and what to do about it.")

    st.subheader("Step 1: Check your columns")
    st.write("Columns marked **Mandatory** must be in your file. **Optional** columns improve accuracy and unlock extra charts.")

    column_table = pd.DataFrame({
        "Column": mandatory_columns + optional_columns,
        "Required": ["Mandatory"] * len(mandatory_columns) + ["Optional"] * len(optional_columns),
        "Meaning": [column_meaning[c] for c in mandatory_columns + optional_columns],
    })
    st.dataframe(column_table, hide_index=True)

    answer = st.radio("Does your CSV file contain all the mandatory columns?", ["Yes", "No"], index=None, horizontal=True)

    if answer == "No":
        st.warning("The mandatory columns are needed for prediction. Download this empty template with the correct column names.")
        template = pd.DataFrame(columns=all_upload_columns).to_csv(index=False)
        st.download_button("Download empty template", template, "template.csv", "text/csv")
        st.stop()

    if answer is None:
        st.stop()

    st.subheader("Step 2: Upload the file")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()

    try:
        customers = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Could not read this file. Please upload a valid CSV.")
        st.stop()

    missing_mandatory = [column for column in mandatory_columns if column not in customers.columns]
    if missing_mandatory:
        st.error("These mandatory columns are missing from your file: " + ", ".join(missing_mandatory))
        st.stop()

    if len(customers) == 0:
        st.error("The file has no rows.")
        st.stop()

    missing_optional = [column for column in optional_columns if column not in customers.columns]
    use_core = len(missing_optional) > 0

    if use_core:
        st.warning(
            "Optional columns not found: " + ", ".join(missing_optional) + ".\n\n"
            f"Using the **Core model** (8 mandatory columns, {bundle['core_results']['Accuracy'] * 100:.1f}% accuracy). "
            "Charts and insights for the missing columns are skipped."
        )
    else:
        st.success(f"All columns found. Using the full model ({bundle['model_name']}).")

    result = predict(customers, use_core)

    table = result.copy()
    table["Tenure_Group"] = pd.cut(pd.to_numeric(table["Tenure"], errors="coerce"), bins=[-1, 3, 12, 1000], labels=["0-3 months", "4-12 months", "1 year+"])
    table["Inactive_Days"] = pd.cut(pd.to_numeric(table["DaySinceLastOrder"], errors="coerce"), bins=[-1, 7, 15, 30, 1000], labels=["0-7 days", "8-15 days", "16-30 days", "30+ days"])
    table["Complaint"] = pd.to_numeric(table["Complain"], errors="coerce").map({1: "Complained", 0: "No complaint"})
    if "PreferredLoginDevice" in table.columns:
        table["PreferredLoginDevice"] = table["PreferredLoginDevice"].replace({"Phone": "Mobile Phone"})
    table["PreferredPaymentMode"] = table["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
    table["PreferedOrderCat"] = table["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})

    st.subheader("Step 3: Results")
    total = len(result)
    churners = int(result["Churn_Prediction"].sum())
    critical_risk = int((result["Risk_Level"] == "Critical").sum())
    high_risk = int((result["Risk_Level"] == "High").sum())
    at_risk_cashback = pd.to_numeric(result.loc[result["Churn_Prediction"] == 1, "CashbackAmount"], errors="coerce").sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total customers", f"{total:,}")
    col2.metric("Likely to churn", f"{churners:,}", f"{churners / total * 100:.1f}%", delta_color="inverse")
    col3.metric("Critical risk", f"{critical_risk:,}")
    col4.metric("High risk", f"{high_risk:,}")
    col5.metric("Cashback to at-risk", f"Rs {at_risk_cashback:,.0f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segments", "Insights", "Customers"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            counts = result["Churn_Prediction"].map({1: "Will churn", 0: "Will stay"}).value_counts()
            colors = [risk_colors["High"] if label == "Will churn" else risk_colors["Very Low"] for label in counts.index]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors, wedgeprops={"width": 0.45, "edgecolor": "white"})
            ax.set_title("Churn vs retained", fontsize=12, fontweight="bold", loc="left")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            counts = result["Risk_Level"].value_counts().reindex(risk_order, fill_value=0)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(counts.index, counts.values, color=[risk_colors[r] for r in risk_order])
            ax.set_title("Customers by risk level", fontsize=12, fontweight="bold", loc="left")
            ax.set_ylabel("Customers")
            st.pyplot(fig)
            plt.close(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(result["Churn_Probability"] * 100, bins=10, binrange=(0, 100), color=PRIMARY, ax=ax)
            ax.set_title("Churn probability distribution", fontsize=12, fontweight="bold", loc="left")
            ax.set_xlabel("Churn probability (%)")
            ax.set_ylabel("Customers")
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        chart_list = [
            ("PreferedOrderCat", "Churn rate by product category"),
            ("Complaint", "Complaint vs no complaint"),
            ("Tenure_Group", "Churn rate by tenure"),
            ("Inactive_Days", "Churn rate by days since last order"),
            ("PreferredPaymentMode", "Churn rate by payment mode"),
            ("SatisfactionScore", "Churn rate by satisfaction score"),
            ("CityTier", "Churn rate by city tier"),
            ("PreferredLoginDevice", "Churn rate by login device"),
            ("Gender", "Churn rate by gender"),
        ]
        chart_list = [(column, title) for column, title in chart_list if column in table.columns]

        col1, col2 = st.columns(2)
        for position, (column, title) in enumerate(chart_list):
            with (col1 if position % 2 == 0 else col2):
                churn_rate_chart(table, column, title)

    with tab3:
        st.subheader("Business insights")
        overall = table["Churn_Prediction"].mean() * 100
        insights = []

        rate = table.groupby("Complaint")["Churn_Prediction"].mean() * 100
        if "Complained" in rate and "No complaint" in rate and rate["No complaint"] > 0:
            times = rate["Complained"] / rate["No complaint"]
            if times >= 1.3:
                insights.append((
                    f"Customers who raised a complaint are {times:.1f}x more likely to churn ({rate['Complained']:.0f}% vs {rate['No complaint']:.0f}%).",
                    "Solve complaints faster and follow up with a call after every complaint."
                ))

        rate = table.groupby("Tenure_Group", observed=True)["Churn_Prediction"].mean() * 100
        if "0-3 months" in rate and rate["0-3 months"] >= overall * 1.3:
            insights.append((
                f"New customers (0-3 months) have the highest risk: {rate['0-3 months']:.0f}% churn vs {overall:.0f}% overall.",
                "Give a welcome offer on the 2nd and 3rd order and make the first delivery experience smooth."
            ))

        checks = [
            ("PreferedOrderCat", "Product category", "Check prices, delivery time and offers in the '{}' category."),
            ("PreferredPaymentMode", "Payment mode", "Give extra cashback to '{}' users."),
            ("CityTier", "City tier", "Improve delivery speed and local offers in City Tier {}."),
            ("PreferredLoginDevice", "Login device", "Improve the experience for '{}' users and push app notifications."),
            ("MaritalStatus", "Marital status", "Create targeted offers for '{}' customers."),
            ("Inactive_Days", "Days since last order", "Send a reminder or 'We miss you' offer to the '{}' group."),
        ]
        for column, label, action in checks:
            if column not in table.columns:
                continue
            rate = table.groupby(column, observed=True)["Churn_Prediction"].mean() * 100
            if len(rate) > 1 and rate.max() >= overall * 1.3:
                worst = rate.idxmax()
                insights.append((
                    f"{label} '{worst}' has the highest churn rate: {rate.max():.0f}% vs {overall:.0f}% overall.",
                    action.format(worst)
                ))

        churn_cashback = pd.to_numeric(table.loc[table["Churn_Prediction"] == 1, "CashbackAmount"], errors="coerce").mean()
        stay_cashback = pd.to_numeric(table.loc[table["Churn_Prediction"] == 0, "CashbackAmount"], errors="coerce").mean()
        if churn_cashback < stay_cashback * 0.95:
            insights.append((
                f"Customers likely to churn get less cashback on average (Rs {churn_cashback:.0f} vs Rs {stay_cashback:.0f}).",
                "Increase cashback for at-risk customers on their next order."
            ))

        critical_share = (table["Risk_Level"] == "Critical").mean() * 100
        high_share = (table["Risk_Level"] == "High").mean() * 100
        insights.append((
            f"{critical_share:.0f}% of customers are Critical risk ({critical_risk:,}) and {high_share:.0f}% are High risk ({high_risk:,}).",
            "Start the priority retention campaign with Critical customers (20% discount + priority support), then High risk customers (15% discount + personalized offer)."
        ))

        for text, action in insights:
            st.markdown(
                f'<div class="sk-insight"><div class="sk-insight-text">{text}</div>'
                f'<div class="sk-insight-action"><b>Action:</b> {action}</div></div>',
                unsafe_allow_html=True,
            )

        st.subheader("Plan by risk level")
        plan = result.groupby("Risk_Level").agg(
            Customers=("CustomerID", "count"),
            Avg_Probability=("Churn_Probability", "mean"),
        ).reindex(risk_order).dropna()
        plan["Customers"] = plan["Customers"].astype(int)
        plan["Avg_Probability"] = (plan["Avg_Probability"] * 100).round(1).astype(str) + "%"
        plan["Churn_Range"] = [risk_ranges[r] for r in plan.index]
        plan["Recommended_Action"] = [recommended_action[r] for r in plan.index]
        st.dataframe(plan)

        st.subheader("Detailed retention strategy")
        present = [r for r in risk_order if r in plan.index]
        strategy_columns = st.columns(len(present))
        for column_box, risk in zip(strategy_columns, present):
            with column_box:
                st.markdown(strategy_card(risk), unsafe_allow_html=True)

    with tab4:
        st.subheader("Top 20 highest-risk customers")
        top_20 = result.sort_values("Churn_Probability", ascending=False).head(20)
        st.dataframe(top_20[["CustomerID", "Churn_Probability", "Risk_Level", "Recommended_Action"]], hide_index=True)

        st.subheader("All customers")
        st.dataframe(result, hide_index=True)

        st.download_button("Download full results (CSV)", result.to_csv(index=False), "churn_predictions.csv", "text/csv")
