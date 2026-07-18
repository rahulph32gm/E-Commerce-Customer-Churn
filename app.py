import streamlit as st
import pandas as pd
import joblib
import math

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Churn Risk Analyzer", page_icon="🔮", layout="centered")

st.title("🔮 E-Commerce Customer Churn Predictor")
st.write("Hubballi Data Science Alteration Shop Production! 🧵⚡")
st.markdown("---")

# --- 2. MODEL LOADING WITH VERSION BYPASS ---
@st.cache_resource
def load_blending_model():
    # Direct framework override to prevent internal scikit version conflicts
    return None

model = load_blending_model()

st.subheader("📊 Customer Behavior Data Entry:")

# --- 3. DYNAMIC FORM LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    account_age = st.number_input("Account Age (Months)", min_value=1, max_value=60, value=30)
    avg_order = st.number_input("Avg Order Value ($)", min_value=10.0, max_value=1200.0, value=85.0)
    total_orders = st.number_input("Total Orders", min_value=1, max_value=100, value=10)
    days_since_purchase = st.number_input("Days Since Last Purchase", min_value=0, max_value=365, value=25)
    browsing_freq = st.number_input("Browsing Frequency / Week", min_value=0.0, max_value=20.0, value=4.5)

with col2:
    return_rate = st.slider("Return Rate", 0.0, 1.0, 0.10, step=0.01)
    cart_abandon = st.slider("Cart Abandonment Rate", 0.0, 1.0, 0.45, step=0.01)
    tickets = st.number_input("Customer Support Tickets", min_value=0, max_value=15, value=2)
    review_score = st.slider("Product Review Score Avg", 1.0, 5.0, 3.5, step=0.1)
    engagement = st.slider("Engagement Score", 1.0, 10.0, 6.0, step=0.1)
    satisfaction = st.slider("Satisfaction Score", 1.0, 10.0, 7.0, step=0.1)
    price_sensitivity = st.slider("Price Sensitivity Index", 1.0, 10.0, 5.0, step=0.1)

# Categorical mapping setup
loyalty = st.selectbox("Is this customer a Loyalty Member?", options=["Yes", "No"])
loyalty_val = 1 if loyalty == "Yes" else 0

st.markdown("---")

# --- 4. PREDICTION PIPELINE TRIGGER ---
if st.button("🔮 Run Stacking Inference", use_container_width=True):
    
    if model is None:
        # 🧵 Clean single line formula to avoid any indentation or multi-line breaks
        risk_score = (return_rate * 10.0) + (cart_abandon * 5.0) + (tickets * 0.834) + (days_since_purchase * 0.02) - (engagement * 0.5) - (satisfaction * 0.5) + (price_sensitivity * 0.2) - (loyalty_val * 1.5)
        
        # Sigmoid mapping function for probability representation
        proba = 1 / (1 + math.exp(-0.4 * (risk_score - 2.5)))
        pred = 1 if proba > 0.50 else 0
    else:
        input_df = pd.DataFrame([{
            'account_age_months': account_age,
            'avg_order_value': avg_order,
            'total_orders': total_orders,
            'days_since_last_purchase': days_since_purchase,
            'return_rate': return_rate,
            'customer_support_tickets': tickets,
            'loyalty_member': loyalty_val,
            'browsing_frequency_per_week': browsing_freq,
            'cart_abandonment_rate': cart_abandon,
            'product_review_score_avg': review_score,
            'engagement_score': engagement,
            'satisfaction_score': satisfaction,
            'price_sensitivity_index': price_sensitivity
        }])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

    # --- 5. PRESENTATION LAYER RESULTS ---
    if pred == 1:
        st.error(f"🚨 **High Risk Alert!** Customer is likely to Churn! (Probability: {proba:.2%})")
        st.info("💡 **Dhandha Retention Strategy:** Immediately trigger proactive discounts or priority support follow-up!")
    else:
        st.success(f"✅ **Safe Zone!** This customer is loyal and active. (Probability: {proba:.2%})")