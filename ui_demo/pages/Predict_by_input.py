import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import joblib
from pandas.api.types import CategoricalDtype
from joblib import load as load_scaler
import os
#file_path = 'D:\\Tran Hoang Vu\\Semester 6\\Big Data Analytics\\assigment\\model\\model.pkl'
file_path = '../model/model.pkl'
# scaler_path = 'D:\\Tran Hoang Vu\\Semester 6\\Big Data Analytics\\assigment\\model\\scaler.plk'
scaler_path = '..\\model\\scaler.plk'


def input():
    ad = st.sidebar.number_input('Administrative', key='Administrative')
    ad_Du = st.sidebar.number_input('Administrative_Duration', key='Administrative_Duration')
    info = st.sidebar.number_input('Informational', key='Informational')
    info_du = st.sidebar.number_input('Informational_Duration', key='Informational_Duration')
    prod = st.sidebar.number_input('ProductRelated', key='ProductRelated')
    pro_du = st.sidebar.number_input('ProductRelated_Duration', key='ProductRelated_Duration')
    bon = st.sidebar.number_input('BounceRates', key='BounceRates')
    exit_rate = st.sidebar.number_input('ExitRates', key='ExitRates')
    pv = st.sidebar.number_input('PageValues', key='PageValues')
    spe = st.sidebar.number_input('SpecialDay', key='SpecialDay')
    month = st.sidebar.selectbox("Choose month", 
                         ["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"], 
                         key='Month')
    os = st.sidebar.selectbox("Choose OperatingSystems", [i for i in range(1, 9)], key='OperatingSystems')
    browser = st.sidebar.selectbox("Choose Browser", [i for i in range(1, 14)], key='Browser')
    region = st.sidebar.selectbox("Choose Region", [i for i in range(1, 10)], key='Region')
    traffic = st.sidebar.selectbox("Choose TrafficType", [i for i in range(1, 21)], key='TrafficType')
    visitor = st.sidebar.selectbox("Choose VisitorType", ['Returning_Visitor ', 'New_Visitor ', 'Other '], key='VisitorType')
    weekend = st.sidebar.radio("Is Weekend", [True, False], key='Weekend')

    return ad, ad_Du, info, info_du, prod, pro_du, bon, exit_rate, pv, spe, month, os, browser, region, traffic, visitor, int(weekend)

def load_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        print(os.getcwd())
        print(f"[ERROR] Không thể load model từ {file_path}: {Exception}")

        return None
# a = load_model(scaler_path)
# b = load_model(file_path)
# print(a)
# print(b)

# 
# Danh sách 12 tháng và 3 kiểu visitor (theo dataset gốc)
ALL_MONTHS = ['Aug','Dec','Feb','Jul','June','Mar','May','Nov','Oct','Sep']
ALL_VISITORS = ['New_Visitor','Other','Returning_Visitor']

def predict_func(ad, ad_Du, info, info_du, prod, pro_du, bon,
                 exit_rate, pv, spe, month, os, browser,
                 region, traffic, visitor, weekend):

    # 1) Tạo DataFrame thô
    keys = ['Administrative','Administrative_Duration','Informational',
            'Informational_Duration','ProductRelated','ProductRelated_Duration',
            'BounceRates','ExitRates','PageValues','SpecialDay',
            'Month','OperatingSystems','Browser','Region',
            'TrafficType','VisitorType','Weekend']

    X_new = pd.DataFrame([dict(zip(keys, [
        ad, ad_Du, info, info_du, prod, pro_du, bon,
        exit_rate, pv, spe, month, os, browser,
        region, traffic, visitor, weekend
    ]))])

    # 2) Ép kiểu Categorical với đủ nhãn, rồi one-hot
    month_cat = CategoricalDtype(categories=ALL_MONTHS)
    visitor_cat = CategoricalDtype(categories=ALL_VISITORS)

    X_new['Month'] = X_new['Month'].astype(month_cat)
    X_new['VisitorType'] = X_new['VisitorType'].astype(visitor_cat)

    X_new = pd.get_dummies(X_new,
                           columns=['Month','VisitorType'],
                           prefix=['Month','VisitorType'])

    # 3) Kiểm tra và thêm cột thiếu (nếu lỡ get_dummies không sinh đủ)
    for m in ALL_MONTHS:
        col = f'Month_{m}'
        if col not in X_new.columns:
            X_new[col] = 0
    for v in ALL_VISITORS:
        col = f'VisitorType_{v}'
        if col not in X_new.columns:
            X_new[col] = 0

    # 4) Load model & scaler
    model  = load_model(file_path)
    scaler = load_scaler('D:\\Tran Hoang Vu\\Semester 6\\Big Data Analytics\\assigment\\model\\scaler.plk')

    # 5) Scale
    X_scaled = scaler.transform(X_new)
    X_scaled = pd.DataFrame(X_scaled, columns=X_new.columns)

    # 6) Tính biến tổng cho Month & VisitorType
    month_cols = [f'Month_{m}' for m in ALL_MONTHS]
    vis_cols   = [f'VisitorType_{v}' for v in ALL_VISITORS]

    X_scaled['Month_Total']        = X_scaled[month_cols].mean(axis=1)
    X_scaled['VisitorType_Total']  = X_scaled[vis_cols].mean(axis=1)

    # 7) Bỏ cột chi tiết, chỉ giữ Total
    X_scaled.drop(columns=month_cols + vis_cols, inplace=True)

    # 8) Predict
    return model.predict(X_scaled)

def reset_inputs():
    st.session_state["Administrative"] = 0.0
    st.session_state["Administrative_Duration"] = 0.0
    st.session_state["Informational"] = 0
    st.session_state["Informational_Duration"] = 0.0
    st.session_state["ProductRelated"] = 0
    st.session_state["ProductRelated_Duration"] = 0.0
    st.session_state["BounceRates"] = 0.0
    st.session_state["ExitRates"] = 0.0
    st.session_state["PageValues"] = 0.0
    st.session_state["SpecialDay"] = 0.0
    st.session_state["Month"] = "January"
    st.session_state["OperatingSystems"] = 1
    st.session_state["Browser"] = 1
    st.session_state["Region"] = 1
    st.session_state["TrafficType"] = 1
    st.session_state["VisitorType"] = 'Returning_Visitor '
    st.session_state["Weekend"] = False


def main():

    st.title('🛒 Customer Purchase Prediction')

   
    col1, col2 ,col3 = st.columns(3)
    with col2:
            if st.button("Reset"):
                reset_inputs()  
                st.rerun()
    with col3:
        values = input()




    with col1:
        if st.button("Predict"):
            result = predict_func(*values)
            if result == False:
                st.write('❌ Not purchase')
            else:
                st.write('✅ Purchase')
main()
