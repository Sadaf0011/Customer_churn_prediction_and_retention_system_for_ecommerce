# =====================================================================
#  app.py  —  THE STREAMLIT WEBSITE
# ---------------------------------------------------------------------
#  Run it with:   streamlit run app.py
#  (Run train_model.py first, so that model.pkl exists.)
#
#  The website has 3 pages, chosen from the menu bar at the top:
#    Home             → welcome section + model performance
#    Single Customer  → predict for one customer
#    Bulk Analysis    → upload a CSV, predict for many customers + insights
# =====================================================================


# ---------------------------------------------------------------------
# STEP 1: Import the libraries we need
# ---------------------------------------------------------------------
import pickle                     # to load the saved model
import pandas as pd               # to work with tables
import matplotlib.pyplot as plt   # to draw charts
import seaborn as sns             # to draw nicer charts
import streamlit as st            # to make the website


# ---------------------------------------------------------------------
# STEP 2: Page settings
# set_page_config must be the FIRST Streamlit command.
# ---------------------------------------------------------------------
st.set_page_config(page_title="Churnio | Retention Hub", layout="wide", initial_sidebar_state="collapsed")

# Clean chart style for every matplotlib/seaborn chart
sns.set_theme(style="whitegrid")
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Colours used in charts (same as in style.html)
PRIMARY = "#4f46e5"
risk_colors = {"Critical": "#991b1b", "High": "#dc2626", "Medium": "#ea580c", "Low": "#ca8a04", "Very Low": "#16a34a"}


# ---------------------------------------------------------------------
# STEP 3: Load the trained model
# model.pkl is a dictionary made by train_model.py (model, scaler, columns ...)
# @st.cache_resource loads it only once, not every time the page refreshes.
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:      # "rb" = read in binary mode
        return pickle.load(file)


bundle = load_model()


# ---------------------------------------------------------------------
# STEP 4: Load the design file (style.html) and split it into 4 parts
#   design_parts[0] = CSS (the look of the app)
#   design_parts[1] = logo (left of the menu bar)
#   design_parts[2] = status badge (right of the menu bar)
#   design_parts[3] = home page section
# ---------------------------------------------------------------------
with open("style.html", encoding="utf-8") as file:
    design_parts = file.read().split("<!-- SPLIT -->")

css_part = design_parts[0]
logo_part = design_parts[1]
status_part = design_parts[2]
home_part = design_parts[3]

# Replace the placeholders in the HTML with real numbers from the model
best_row = bundle["results"].set_index("Model").loc[bundle["model_name"]]
placeholders = {
    "[[MODEL_NAME]]": bundle["model_name"],
    "[[ACCURACY]]": f"{best_row['Accuracy'] * 100:.1f}",
    "[[RECALL]]": f"{best_row['Recall'] * 100:.1f}",
    "[[F1]]": f"{best_row['F1']:.2f}",
    "[[AUC]]": f"{best_row['ROC_AUC']:.2f}",
    "[[GAUGE_DEG]]": f"{best_row['Accuracy'] * 180:.0f}",       # half circle = 180 degrees
    "[[CUSTOMERS]]": f"{bundle.get('rows_trained', 0):,}",
    "[[CORE_ACCURACY]]": f"{bundle['core_results']['Accuracy'] * 100:.1f}",
}
for key, value in placeholders.items():
    status_part = status_part.replace(key, value)
    home_part = home_part.replace(key, value)

# Apply the CSS to the whole app
st.markdown(css_part, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# STEP 5: The top menu bar (logo | menu | status)
# The menu is a real Streamlit radio button, so clicking it really changes the page.
# style.html makes the radio look like tab buttons.
# ---------------------------------------------------------------------
nav_left, nav_middle, nav_right = st.columns([2, 4, 2])

with nav_left:
    st.markdown(logo_part, unsafe_allow_html=True)

with nav_middle:
    page = st.radio(
        "Menu",
        ["Home", "Single Customer", "Bulk Analysis"],
        horizontal=True,
        label_visibility="collapsed",      # hide the word "Menu"
        key="page",
    )

with nav_right:
    st.markdown(status_part, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# STEP 6: Settings used across the app
# ---------------------------------------------------------------------

# 6a. The same column lists as in train_model.py
numeric_columns = [
    "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
    "Complain", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount",
]
text_columns = ["PreferredLoginDevice", "PreferredPaymentMode", "Gender", "PreferedOrderCat", "MaritalStatus"]
all_feature_columns = numeric_columns + text_columns

# 6b. Mandatory and optional columns for the CSV upload
#     Mandatory = CustomerID + the 8 most important columns (decided in train_model.py, STEP 10)
#     Optional  = all other columns. If any are missing, the app uses the smaller Core model.
mandatory_columns = ["CustomerID"] + bundle["mandatory_columns"]
optional_columns = [column for column in all_feature_columns if column not in mandatory_columns]
all_upload_columns = ["CustomerID"] + all_feature_columns

# 6c. Short meaning of each column (shown in Bulk Analysis)
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

# 6d. Retention plan: 5 risk categories based on churn probability
#     Order from most risky to least risky (used in tables and charts)
risk_order = ["Critical", "High", "Medium", "Low", "Very Low"]

# Churn probability range of each risk category
risk_ranges = {
    "Very Low": "0-20%",
    "Low": "21-40%",
    "Medium": "41-60%",
    "High": "61-80%",
    "Critical": "81-100%",
}

# ONE recommended action for each risk category (shown in tables and the CSV)
recommended_action = {
    "Very Low": "Regular loyalty points",
    "Low": "5% discount or reward points",
    "Medium": "10% discount or free shipping",
    "High": "15% discount + personalized offer",
    "Critical": "20% discount + priority retention campaign",
}

# Short offer (shown on the small result card)
short_offer = {
    "Very Low": "Loyalty points",
    "Low": "5% off",
    "Medium": "10% off",
    "High": "15% off",
    "Critical": "20% off",
}

# DETAILED retention strategy for each risk category (list of steps)
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


# 6e. strategy_card(): makes the HTML for one risk category's strategy card
#     (style comes from .sk-insight in style.html; the left line gets the risk colour)
def strategy_card(risk):
    steps = "<br>".join("&bull; " + step for step in retention_strategy[risk])
    return (
        f'<div class="sk-insight" style="border-left-color: {risk_colors[risk]};">'
        f'<div class="sk-insight-text">{risk} risk &middot; {risk_ranges[risk]}</div>'
        f'<div class="sk-insight-action">{steps}</div></div>'
    )


# ---------------------------------------------------------------------
# STEP 7: Helper functions (only 2, because more than one page uses them)
# ---------------------------------------------------------------------

# 7a. predict(): takes a table of customers and adds 5 new columns:
#     Churn_Probability, Churn_Prediction, Risk_Level, Recommended_Action
#     use_core=False → full model (all 18 columns)
#     use_core=True  → Core model (only the 8 mandatory columns)
def predict(customers, use_core=False):

    # Pick which model, scaler and column list to use
    if use_core:
        model = bundle["core_model"]
        scaler = bundle["core_scaler"]
        model_columns = bundle["core_model_columns"]
    else:
        model = bundle["model"]
        scaler = bundle["scaler"]
        model_columns = bundle["model_columns"]

    # Add any missing optional column as empty, so the code below works the same way.
    # The Core model does not use these columns, so the empty values do not matter.
    data = customers.copy()
    for column in all_feature_columns:
        if column not in data.columns:
            data[column] = None
    data = data[all_feature_columns]

    # Same cleaning as train_model.py (merge repeated labels)
    data["PreferredLoginDevice"] = data["PreferredLoginDevice"].replace({"Phone": "Mobile Phone"})
    data["PreferredPaymentMode"] = data["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
    data["PreferedOrderCat"] = data["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})

    # Make sure number columns are numbers, then fill any missing values
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.fillna(bundle["fill_values"])

    # Text → 0/1 columns, in the SAME order the model was trained on
    X = pd.get_dummies(data, columns=text_columns)
    X = X.reindex(columns=model_columns, fill_value=0).astype(float)

    # Scale with the saved scaler, then get the churn probability (0 to 1)
    X = scaler.transform(X)
    probability = model.predict_proba(X)[:, 1]

    # Add the results to the customer table
    result = customers.copy()
    result["Churn_Probability"] = probability.round(3)
    result["Churn_Prediction"] = (probability >= 0.5).astype(int)    # 1 = will churn

    # Risk category from the churn percentage (rounded to a whole number):
    #   0-20 Very Low | 21-40 Low | 41-60 Medium | 61-80 High | 81-100 Critical
    percent = (probability * 100).round()
    result["Risk_Level"] = pd.cut(percent, bins=[-1, 20, 40, 60, 80, 100], labels=["Very Low", "Low", "Medium", "High", "Critical"]).astype(str)
    result["Recommended_Action"] = result["Risk_Level"].map(recommended_action)
    return result


# 7b. churn_rate_chart(): bar chart of churn % for each group of a column
#     (same groupby → mean → ×100 idea as in train_model.py)
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


# =====================================================================
# PAGE 1: HOME
# =====================================================================
if page == "Home":

    # -----------------------------------------------------------------
    # STEP 8: Welcome section from style.html (headline, steps, model card, stats)
    # -----------------------------------------------------------------
    st.markdown(home_part, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # STEP 9: Model comparison table and coupon rules
    # -----------------------------------------------------------------
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
        low_to_high = list(reversed(risk_order))      # Very Low first, Critical last
        rules = pd.DataFrame({
            "Churn probability": [risk_ranges[r] for r in low_to_high],
            "Risk category": low_to_high,
            "Recommended action": [recommended_action[r] for r in low_to_high],
        })
        st.dataframe(rules, hide_index=True)

    # Detailed retention strategy: one card per risk category, side by side
    st.subheader("Detailed retention strategy")
    strategy_columns = st.columns(5)
    for column_box, risk in zip(strategy_columns, reversed(risk_order)):
        with column_box:
            st.markdown(strategy_card(risk), unsafe_allow_html=True)


# =====================================================================
# PAGE 2: SINGLE CUSTOMER PREDICTION
# =====================================================================
elif page == "Single Customer":

    # -----------------------------------------------------------------
    # STEP 10: Show the input form
    # st.form groups all inputs, so the page updates only when "Predict" is clicked
    # -----------------------------------------------------------------
    st.title("Single Customer Prediction")
    st.write("Enter one customer's details to check if they are likely to leave.")
    options = bundle["options"]      # dropdown choices saved during training

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)      # 3 columns side by side

        with col1:   # order details
            st.markdown("**Orders**")
            tenure = st.number_input("Tenure (months)", 0, 100, 10)
            order_count = st.number_input("Orders last month", 0, 50, 2)
            days_since = st.number_input("Days since last order", 0, 100, 3)
            coupon_used = st.number_input("Coupons used last month", 0, 50, 1)
            cashback = st.number_input("Average cashback (Rs)", 0.0, 1000.0, 150.0)
            order_hike = st.number_input("Order amount hike from last year (%)", 0, 100, 15)

        with col2:   # experience details
            st.markdown("**Experience**")
            satisfaction = st.slider("Satisfaction score", 1, 5, 3)
            complain = st.radio("Complaint last month?", ["No", "Yes"], horizontal=True)
            hours_app = st.number_input("Hours spent on app", 0, 24, 3)
            devices = st.number_input("Devices registered", 1, 10, 3)
            addresses = st.number_input("Saved addresses", 1, 30, 3)
            distance = st.number_input("Warehouse to home (km)", 1, 200, 15)

        with col3:   # profile details
            st.markdown("**Profile**")
            city_tier = st.selectbox("City tier", [1, 2, 3])
            category = st.selectbox("Preferred category", options["PreferedOrderCat"])
            payment = st.selectbox("Payment mode", options["PreferredPaymentMode"])
            device = st.selectbox("Login device", options["PreferredLoginDevice"])
            gender = st.selectbox("Gender", options["Gender"])
            marital = st.selectbox("Marital status", options["MaritalStatus"])

        submitted = st.form_submit_button("Predict churn")

    # Stop here until the user clicks Predict
    if not submitted:
        st.stop()

    # -----------------------------------------------------------------
    # STEP 11: Put the inputs into a one-row table and predict
    # -----------------------------------------------------------------
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

    result = predict(customer).iloc[0]      # .iloc[0] = take the first (only) row
    probability = result["Churn_Probability"]
    risk = result["Risk_Level"]

    # -----------------------------------------------------------------
    # STEP 12: Show the result
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("Result")

    if result["Churn_Prediction"] == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")

    # Coloured risk badge (style comes from .sk-badge in style.html)
    # (the class name cannot have a space, so "Very Low" becomes "VeryLow")
    st.markdown(f'<span class="sk-badge sk-badge-{risk.replace(" ", "")}">{risk} risk</span>', unsafe_allow_html=True)
    st.write("")

    # 3 result cards side by side
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn probability", f"{probability * 100:.1f}%")
    col2.metric("Risk level", risk)
    col3.metric("Recommended offer", short_offer[risk])
    st.progress(int(round(probability * 100)))      # bar from 0 to 100

    # One main action + the detailed strategy steps for this risk category
    st.subheader("Recommended action")
    st.info(recommended_action[risk])
    st.subheader("Retention strategy")
    st.markdown(strategy_card(risk), unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # STEP 13: Possible reasons
    # For each number column we compare the customer's value with:
    #   - the average of customers who LEFT
    #   - the average of customers who STAYED
    # If the value is closer to the "left" average, we show it as a reason.
    # -----------------------------------------------------------------
    if risk != "Very Low":
        st.subheader("Possible reasons")
        reasons = []
        for column in numeric_columns:
            left_avg = bundle["churner_means"][column]
            stay_avg = bundle["stayer_means"][column]
            spread = bundle["overall_std"][column]

            # Skip columns where both groups are almost the same
            if spread == 0 or abs(left_avg - stay_avg) / spread < 0.15:
                continue

            middle = (left_avg + stay_avg) / 2
            value = float(result[column])

            # Is the value on the "left" side of the middle point?
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


# =====================================================================
# PAGE 3: BULK ANALYSIS (CSV UPLOAD)
# =====================================================================
else:

    # -----------------------------------------------------------------
    # STEP 14: Show the columns (mandatory and optional) and ask the user
    # -----------------------------------------------------------------
    st.title("Bulk Analysis")
    st.write("Upload a CSV of many customers to see how many may leave and what to do about it.")

    st.subheader("Step 1: Check your columns")
    st.write("Columns marked **Mandatory** must be in your file. **Optional** columns improve accuracy and unlock extra charts.")

    # Build the table: mandatory columns first, then optional ones
    column_table = pd.DataFrame({
        "Column": mandatory_columns + optional_columns,
        "Required": ["Mandatory"] * len(mandatory_columns) + ["Optional"] * len(optional_columns),
        "Meaning": [column_meaning[c] for c in mandatory_columns + optional_columns],
    })
    st.dataframe(column_table, hide_index=True)

    answer = st.radio("Does your CSV file contain all the mandatory columns?", ["Yes", "No"], index=None, horizontal=True)

    # If "No": offer an empty template file and stop
    if answer == "No":
        st.warning("The mandatory columns are needed for prediction. Download this empty template with the correct column names.")
        template = pd.DataFrame(columns=all_upload_columns).to_csv(index=False)
        st.download_button("Download empty template", template, "template.csv", "text/csv")
        st.stop()

    # If nothing is selected yet: wait
    if answer is None:
        st.stop()

    # -----------------------------------------------------------------
    # STEP 15: Upload and check the file
    # -----------------------------------------------------------------
    st.subheader("Step 2: Upload the file")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()

    # Try to read the file; show an error if it is not a real CSV
    try:
        customers = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Could not read this file. Please upload a valid CSV.")
        st.stop()

    # Check that every MANDATORY column is present
    missing_mandatory = [column for column in mandatory_columns if column not in customers.columns]
    if missing_mandatory:
        st.error("These mandatory columns are missing from your file: " + ", ".join(missing_mandatory))
        st.stop()

    if len(customers) == 0:
        st.error("The file has no rows.")
        st.stop()

    # -----------------------------------------------------------------
    # STEP 16: Choose the model and predict for every customer
    # All optional columns present → full model
    # Some optional columns missing → Core model
    # -----------------------------------------------------------------
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

    # Add some group columns, used only for charts and insights
    table = result.copy()
    table["Tenure_Group"] = pd.cut(pd.to_numeric(table["Tenure"], errors="coerce"), bins=[-1, 3, 12, 1000], labels=["0-3 months", "4-12 months", "1 year+"])
    table["Inactive_Days"] = pd.cut(pd.to_numeric(table["DaySinceLastOrder"], errors="coerce"), bins=[-1, 7, 15, 30, 1000], labels=["0-7 days", "8-15 days", "16-30 days", "30+ days"])
    table["Complaint"] = pd.to_numeric(table["Complain"], errors="coerce").map({1: "Complained", 0: "No complaint"})
    if "PreferredLoginDevice" in table.columns:      # optional column, so check first
        table["PreferredLoginDevice"] = table["PreferredLoginDevice"].replace({"Phone": "Mobile Phone"})
    table["PreferredPaymentMode"] = table["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
    table["PreferedOrderCat"] = table["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})

    # -----------------------------------------------------------------
    # STEP 17: KPI cards (the big numbers on top)
    # -----------------------------------------------------------------
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

    # 4 tabs to keep the page tidy
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segments", "Insights", "Customers"])

    # -----------------------------------------------------------------
    # STEP 18: Overview tab — 3 main charts
    # -----------------------------------------------------------------
    with tab1:
        col1, col2, col3 = st.columns(3)

        # Chart 1: Pie — will churn vs will stay
        with col1:
            counts = result["Churn_Prediction"].map({1: "Will churn", 0: "Will stay"}).value_counts()
            colors = [risk_colors["High"] if label == "Will churn" else risk_colors["Very Low"] for label in counts.index]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors, wedgeprops={"width": 0.45, "edgecolor": "white"})
            ax.set_title("Churn vs retained", fontsize=12, fontweight="bold", loc="left")
            st.pyplot(fig)
            plt.close(fig)

        # Chart 2: Bar — customers in each risk level
        with col2:
            counts = result["Risk_Level"].value_counts().reindex(risk_order, fill_value=0)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(counts.index, counts.values, color=[risk_colors[r] for r in risk_order])
            ax.set_title("Customers by risk level", fontsize=12, fontweight="bold", loc="left")
            ax.set_ylabel("Customers")
            st.pyplot(fig)
            plt.close(fig)

        # Chart 3: Histogram — spread of churn probabilities
        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(result["Churn_Probability"] * 100, bins=10, binrange=(0, 100), color=PRIMARY, ax=ax)
            ax.set_title("Churn probability distribution", fontsize=12, fontweight="bold", loc="left")
            ax.set_xlabel("Churn probability (%)")
            ax.set_ylabel("Customers")
            st.pyplot(fig)
            plt.close(fig)

    # -----------------------------------------------------------------
    # STEP 19: Segments tab — churn rate by each customer group
    # A chart is drawn only if its column is in the uploaded file
    # -----------------------------------------------------------------
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

        # Put the charts in 2 columns: even positions on the left, odd on the right
        col1, col2 = st.columns(2)
        for position, (column, title) in enumerate(chart_list):
            with (col1 if position % 2 == 0 else col2):
                churn_rate_chart(table, column, title)

    # -----------------------------------------------------------------
    # STEP 20: Insights tab — business insights written automatically
    # How it works:
    #   1. groupby() finds the churn rate of each group
    #   2. compare it with the overall churn rate
    #   3. if a group is clearly higher, fill a ready-made sentence (f-string)
    # -----------------------------------------------------------------
    with tab3:
        st.subheader("Business insights")
        overall = table["Churn_Prediction"].mean() * 100
        insights = []      # list of (insight sentence, action sentence)

        # Insight 1: complaints
        rate = table.groupby("Complaint")["Churn_Prediction"].mean() * 100
        if "Complained" in rate and "No complaint" in rate and rate["No complaint"] > 0:
            times = rate["Complained"] / rate["No complaint"]
            if times >= 1.3:
                insights.append((
                    f"Customers who raised a complaint are {times:.1f}x more likely to churn ({rate['Complained']:.0f}% vs {rate['No complaint']:.0f}%).",
                    "Solve complaints faster and follow up with a call after every complaint.",
                ))

        # Insight 2: new customers
        rate = table.groupby("Tenure_Group", observed=True)["Churn_Prediction"].mean() * 100
        if "0-3 months" in rate and rate["0-3 months"] >= overall * 1.3:
            insights.append((
                f"New customers (0-3 months) have the highest risk: {rate['0-3 months']:.0f}% churn vs {overall:.0f}% overall.",
                "Give a welcome offer on the 2nd and 3rd order and make the first delivery experience smooth.",
            ))

        # Insights 3-8: the worst group in each column
        checks = [
            ("PreferedOrderCat", "Product category", "Check prices, delivery time and offers in the '{}' category."),
            ("PreferredPaymentMode", "Payment mode", "Give extra cashback to '{}' users."),
            ("CityTier", "City tier", "Improve delivery speed and local offers in City Tier {}."),
            ("PreferredLoginDevice", "Login device", "Improve the experience for '{}' users and push app notifications."),
            ("MaritalStatus", "Marital status", "Create targeted offers for '{}' customers."),
            ("Inactive_Days", "Days since last order", "Send a reminder or 'We miss you' offer to the '{}' group."),
        ]
        for column, label, action in checks:
            if column not in table.columns:      # skip optional columns that were not uploaded
                continue
            rate = table.groupby(column, observed=True)["Churn_Prediction"].mean() * 100
            if len(rate) > 1 and rate.max() >= overall * 1.3:
                worst = rate.idxmax()      # the group with the highest churn rate
                insights.append((
                    f"{label} '{worst}' has the highest churn rate: {rate.max():.0f}% vs {overall:.0f}% overall.",
                    action.format(worst),
                ))

        # Insight 9: cashback of churners vs stayers
        churn_cashback = pd.to_numeric(table.loc[table["Churn_Prediction"] == 1, "CashbackAmount"], errors="coerce").mean()
        stay_cashback = pd.to_numeric(table.loc[table["Churn_Prediction"] == 0, "CashbackAmount"], errors="coerce").mean()
        if churn_cashback < stay_cashback * 0.95:
            insights.append((
                f"Customers likely to churn get less cashback on average (Rs {churn_cashback:.0f} vs Rs {stay_cashback:.0f}).",
                "Increase cashback for at-risk customers on their next order.",
            ))

        # Insight 10: share of Critical and High risk customers (always shown)
        critical_share = (table["Risk_Level"] == "Critical").mean() * 100
        high_share = (table["Risk_Level"] == "High").mean() * 100
        insights.append((
            f"{critical_share:.0f}% of customers are Critical risk ({critical_risk:,}) and {high_share:.0f}% are High risk ({high_risk:,}).",
            "Start the priority retention campaign with Critical customers (20% discount + priority support), then High risk customers (15% discount + personalized offer).",
        ))

        # Show every insight as a card (style comes from .sk-insight in style.html)
        for text, action in insights:
            st.markdown(
                f'<div class="sk-insight"><div class="sk-insight-text">{text}</div>'
                f'<div class="sk-insight-action"><b>Action:</b> {action}</div></div>',
                unsafe_allow_html=True,
            )

        # Plan table: one row per risk level
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

        # Detailed strategy cards only for the risk categories present in this file
        st.subheader("Detailed retention strategy")
        present = [r for r in risk_order if r in plan.index]
        strategy_columns = st.columns(len(present))
        for column_box, risk in zip(strategy_columns, present):
            with column_box:
                st.markdown(strategy_card(risk), unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # STEP 21: Customers tab — top 20 risky customers + download
    # -----------------------------------------------------------------
    with tab4:
        st.subheader("Top 20 highest-risk customers")
        top_20 = result.sort_values("Churn_Probability", ascending=False).head(20)
        st.dataframe(top_20[["CustomerID", "Churn_Probability", "Risk_Level", "Recommended_Action"]], hide_index=True)

        st.subheader("All customers")
        st.dataframe(result, hide_index=True)

        # Download all results as a CSV file
        st.download_button("Download full results (CSV)", result.to_csv(index=False), "churn_predictions.csv", "text/csv")
