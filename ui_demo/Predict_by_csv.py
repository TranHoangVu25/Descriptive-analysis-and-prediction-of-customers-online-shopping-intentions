import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Set page config trước
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🔮",
    layout="wide",
)

# Title
st.title("🛒 Customer Purchase Prediction")
st.markdown("---")

# file_path = 'D:\\Tran Hoang Vu\\Semester 6\\Big Data Analytics\\assigment\\model\\model.pkl'
file_path = '..\\model\\model.pkl'


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
    st.error("❌ Fail in loading model.")

# File uploader for prediction data
uploaded_file = st.file_uploader("📂 Upload CSV for Prediction", type="csv")

def draw_pie_chart(df):
    st.subheader("ROC & Purchase Ratio")
    counts = df['predicted_purchase'].value_counts().rename(index={0: 'Not Purchased', 1: 'Purchased'})
    col1, col2, = st.columns([1, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(1.3, 1.3))
        ax.pie(counts, 
               labels=counts.index, 
               autopct='%1.1f%%', 
               startangle=90,
                textprops={'fontsize': 8},   # giảm fontsize xuống 5
                labeldistance=1.5,           # khoảng cách label so với tâm
                pctdistance=0.6              # khoảng cách số phần trăm so với tâm               
                )
        ax.axis('equal')
        st.pyplot(fig)
    with col1:
        st.image('ROC.png')


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

        draw_pie_chart(df)
        

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"❌ Fail in prediction: {e}")

else:
    st.info("⬆️ Please, upload csv.")

