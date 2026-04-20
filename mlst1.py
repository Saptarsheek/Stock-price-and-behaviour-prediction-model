import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Nifty 50 Predictor",
    page_icon="📊",
    layout="wide"
)
 
#we inject CSS using markdown
# and apply a gradient background to the whole app
#KPI Card styling 
#Metric text styling 
st.markdown(""" 
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
 
/* KPI Cards */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    text-align: center;
}

/* Metric text */
div[data-testid="stMetric"] label {
    color: #555;
    font-weight: 600;
}

div[data-testid="stMetric"] div {
    color: black;
    font-size: 24px;
    font-weight: bold;
}

/* Titles */
h1, h2, h3 {
    color: #00c6ff;
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.title("Nifty 50 Price Prediction Dashboard")
st.markdown("### Predict Stock Prices using Machine Learning")

#LOAD MODEL
model = joblib.load(r"C:\Users\saptu\Downloads\xgb_model.pkl")

#FILE UPLOAD 
uploaded_file = st.file_uploader("Upload your CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    #SIDEBAR
    st.sidebar.title("Dashboard Controls")

    # FILTER
    st.sidebar.header("Filter Data")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())

    df = df[(df['Date'] >= pd.to_datetime(start_date)) &
            (df['Date'] <= pd.to_datetime(end_date))]
    df = df.sort_values('Date')

    # FEATURE ENGINEERING
    df['MA_5'] = df['Close'].rolling(5).mean().shift(1)
    df['MA_10'] = df['Close'].rolling(10).mean().shift(1)

    df['lag_1'] = df['Close'].shift(1)
    df['lag_2'] = df['Close'].shift(2)
    df['lag_3'] = df['Close'].shift(3)

    df['volatility'] = df['Close'].rolling(5).std().shift(1)

    df = df.dropna()

    # TREND LINE
    x_vals = np.arange(len(df))
    df['Trend_MA5'] = np.poly1d(np.polyfit(x_vals, df['MA_5'], 1))(x_vals)
    df['Trend_MA10'] = np.poly1d(np.polyfit(x_vals, df['MA_10'], 1))(x_vals)

    # FEATURE SELECTION
    st.sidebar.header("Feature Selection")
    features = ['MA_5', 'MA_10', 'lag_1', 'lag_2', 'lag_3', 'volatility', 'Volume']

    selected_features = st.sidebar.multiselect(
        "Select Features",
        features,
        default=features
    )

    # SETTINGS
    st.sidebar.header("⚙️ Settings")
    show_data = st.sidebar.checkbox("Show Raw Data", True)
    show_chart = st.sidebar.checkbox("Show Charts", True)
    show_prediction = st.sidebar.checkbox("Show Predictions", True)

    # STAKEHOLDER INSIGHTS
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Stakeholder Insights")
    st.sidebar.info("""
• Total Records indicate the volume of historical data used for analysis  

• Latest Actual Price reflects the current market value  

• Predicted Price represents the model’s forecast  

• Comparison helps identify bullish/bearish trends  

• Enables data-driven decision making  
""")

    # MODEL INPUT
    X = df[selected_features]
    predictions = model.predict(X)
    df['Predicted_Close'] = predictions

    # KPI
    st.markdown("## ♦ Key Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric("♦ Total Records", len(df))
    col2.metric("♦ Latest Actual", round(df['Close'].iloc[-1], 2))
    col3.metric("♦ Predicted", round(df['Predicted_Close'].iloc[-1], 2))

    # FEATURE IMPORTANCE (FIXED)
    with st.expander("Feature Importance"):
        if hasattr(model, "feature_importances_"):
            base_features = ['MA_5', 'MA_10', 'lag_1', 'lag_2', 'lag_3', 'volatility', 'Volume']

            importance_df = pd.DataFrame({
                'Feature': base_features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            st.bar_chart(importance_df.set_index('Feature'))
            st.dataframe(importance_df)

        else:
            st.warning("⚠️ No feature importance available")

    # CHARTS
    if show_chart:
        st.subheader("Price vs Prediction")
        st.line_chart(df[['Close', 'Predicted_Close']])

        st.subheader("Volume Analysis")
        st.bar_chart(df['Volume'])

        st.subheader("Moving Averages with Trend Line")
        st.line_chart(df[['MA_5', 'MA_10', 'Trend_MA5', 'Trend_MA10']])

        df['Error'] = abs(df['Close'] - df['Predicted_Close'])
        st.subheader("Prediction Error")
        st.line_chart(df['Error'])

    # DATA
    if show_data:
        st.subheader("Dataset Preview")
        st.dataframe(df.tail())

    # FUTURE PREDICTION
    if show_prediction:
        st.subheader("Future Prediction")
        last_input = X.iloc[[-1]]
        future_pred = model.predict(last_input)

        st.success(f"Next Predicted Price: ₹ {round(future_pred[0], 2)}")

        if future_pred[0] > df['Close'].iloc[-1]:
            st.info("Market Trend: Bullish")
        else:
            st.warning("Market Trend: Bearish")

    # DOWNLOAD
    st.subheader("Download Results")
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )

else:
    st.info("Upload a CSV file to get started")