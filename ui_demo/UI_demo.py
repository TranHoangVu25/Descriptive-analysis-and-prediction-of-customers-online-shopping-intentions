import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🔮",
    layout="wide"
)

# Title
st.title("🛒 Customer Purchase Prediction")
st.markdown("---")

# file_path = 'D:\Tran Hoang Vu\Semester 6\Big Data Analytics\\assigment\model\model.pkl'

# import os
# file_path = os.path.join('.', 'model', 'model.pkl')
file_path = '.\\model\\model.pkl'
# Load model from sidebar
# st.sidebar.header("Configuration")
# model_path = st.sidebar.text_input("Model file path", file_path)
model_path = file_path
@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

model = load_model(model_path)
if not model:
    st.error("❌ Không tải được model.")

# File uploader for prediction data
uploaded_file = st.file_uploader("📂 Upload CSV for Prediction", type="csv")

if uploaded_file and model:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(5))

    try:
        # Predict
        preds = model.predict(df)
        df['predicted_purchase'] = preds

        # Show results
        st.subheader("Prediction Results")
        st.dataframe(df)

        # Plot purchase ratio
        st.subheader("Purchase Ratio")
        counts = df['predicted_purchase'].value_counts().rename(index={0: 'Not Purchased', 1: 'Purchased'})
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {e}")

else:
    st.info("⬆️ Vui lòng tải lên file CSV để bắt đầu.")

st.markdown("---")
st.caption("Designed by VoHoangTran.")
