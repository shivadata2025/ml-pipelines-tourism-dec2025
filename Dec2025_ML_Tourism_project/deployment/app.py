
"""
Tourism Package Predictor - Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Tourism Package Predictor",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        width: 100%;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .prediction-positive {
        background-color: #D1FAE5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin: 20px 0;
    }
    .prediction-negative {
        background-color: #FEE2E2;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #EF4444;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Tourism Package Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict customer interest in Wellness Tourism Packages")

# Try to import predict function
try:
    from predict import predict
    PREDICT_AVAILABLE = True
    st.sidebar.success("Prediction module loaded")
except ImportError as e:
    PREDICT_AVAILABLE = False
    st.sidebar.warning(f"Predict module not available: {e}")
except Exception as e:
    PREDICT_AVAILABLE = False
    st.sidebar.error(f"Error: {e}")

# Sidebar for inputs
st.sidebar.header("Customer Information")

# Create tabs for better organization
tab1, tab2 = st.sidebar.tabs(["Personal", "Travel"])

with tab1:
    Age = st.slider("Age", 18, 70, 35)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Business", "Free Lancer"])
    MonthlyIncome = st.number_input("Monthly Income ($)", 1000, 100000, 25000, 1000)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

with tab2:
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    NumberOfTrips = st.slider("Number of Trips", 0, 10, 2)
    Passport = st.radio("Has Passport?", ["Yes", "No"])
    OwnCar = st.radio("Owns Car?", ["Yes", "No"])
    NumberOfPersonVisiting = st.slider("Travel Group Size", 1, 5, 2)
    NumberOfChildrenVisiting = st.slider("Children (under 5)", 0, 3, 0)
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    DurationOfPitch = st.slider("Pitch Duration (minutes)", 5, 60, 15)
    NumberOfFollowups = st.slider("Follow-ups", 0, 10, 3)
    ProductPitched = st.selectbox("Product Offered", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
    PreferredPropertyStar = st.selectbox("Preferred Hotel Star", [3, 4, 5])
    PitchSatisfactionScore = st.slider("Satisfaction Score (1-5)", 1, 5, 3)

# Predict button
if st.button("Predict Purchase Probability"):
    # Prepare input data
    input_data = {
        "CustomerID": 1000,
        "ProdTaken": 0,  # This is what we're predicting
        "Age": float(Age),
        "TypeofContact": TypeofContact,
        "CityTier": int(CityTier),
        "DurationOfPitch": float(DurationOfPitch),
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": int(NumberOfPersonVisiting),
        "NumberOfFollowups": float(NumberOfFollowups),
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": float(PreferredPropertyStar),
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": float(NumberOfTrips),
        "Passport": 1 if Passport == "Yes" else 0,
        "PitchSatisfactionScore": int(PitchSatisfactionScore),
        "OwnCar": 1 if OwnCar == "Yes" else 0,
        "NumberOfChildrenVisiting": float(NumberOfChildrenVisiting),
        "Designation": Designation,
        "MonthlyIncome": float(MonthlyIncome)
    }

    st.markdown("---")
    st.subheader("Prediction Results")

    if PREDICT_AVAILABLE:
        try:
            # Get prediction
            result, confidence = predict(input_data)

            # Display results in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if result == 1:
                    st.success("Will Purchase")
                else:
                    st.error("Will Not Purchase")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Customer Score", f"{int(confidence*100)}/100")
                st.markdown('</div>', unsafe_allow_html=True)

            # Visual indicator
            import plotly.graph_objects as go

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Purchase Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 30], 'color': "#FEE2E2"},
                        {'range': [30, 70], 'color': "#FEF3C7"},
                        {'range': [70, 100], 'color': "#D1FAE5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))

            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.subheader("Recommendations")

            if result == 1:
                st.markdown('<div class="prediction-positive">', unsafe_allow_html=True)
                st.success("High Potential Customer!")
                st.markdown("""
                **Immediate Actions Required:**
                - Contact within 24 hours
                - Personalized Wellness Package
                - 15% early-bird discount
                - Schedule demo session
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-negative">', unsafe_allow_html=True)
                st.warning("Low Probability Customer")
                st.markdown("""
                **Recommended Strategy:**
                - Automated: Send brochure & testimonials
                - Communication: Monthly newsletter
                - Timing: Re-evaluate in 3 months
                - Focus: Prioritize high-potential leads
                """)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Running in demo mode...")
            PREDICT_AVAILABLE = False

    if not PREDICT_AVAILABLE:
        # Demo mode
        st.info("Running in demo mode")

        # Simple rule-based prediction
        score = 0
        if Age < 40: score += 1
        if MonthlyIncome > 25000: score += 1
        if Passport == "Yes": score += 1
        if NumberOfTrips > 1: score += 1
        if PitchSatisfactionScore > 3: score += 1

        result = 1 if score >= 3 else 0
        confidence = score / 5

        col1, col2 = st.columns(2)

        with col1:
            if result == 1:
                st.success("Demo: Will Purchase")
            else:
                st.error("Demo: Will Not Purchase")

        with col2:
            st.metric("Demo Score", f"{score}/5")

# About section
with st.expander("About This Application"):
    st.markdown("""
    ## Tourism Package Prediction System

    **Purpose:** 
    Predict customer likelihood to purchase Wellness Tourism Packages using machine learning.

    **Key Features:**
    - Real-time prediction based on customer profile
    - Confidence scoring with visual indicators
    - Actionable recommendations for sales teams

    **Model Information:**
    - **Algorithm**: Random Forest Classifier
    - **Accuracy**: ~85% on test data
    - **Features**: 20 customer attributes

    **MLOps Pipeline:**
    - Data Versioning: Hugging Face Datasets
    - Model Registry: Hugging Face Model Hub
    - CI/CD: GitHub Actions
    - Deployment: Streamlit on Hugging Face Spaces
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        <p><strong>MLOps Tourism Project</strong></p>
        <p>
            <a href="https://github.com/krish129/mlops-tourism-project" target="_blank">GitHub</a> | 
            <a href="https://huggingface.co/krish129" target="_blank">Hugging Face</a>
        </p>
        <p style="color: #666; font-size: 0.9rem;">
            Built with Streamlit, Scikit-learn, and Hugging Face
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
