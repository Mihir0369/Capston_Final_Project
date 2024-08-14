import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('final_xgboost_regressor_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data['model']
le_department_name = data['le_department_name']

def show_predict_page():
    st.title("Supply Chain Sales Price Prediction")
    st.header('Kindly, Please Enter the following details')

    departments = ('Fitness', 'Apparel', 'Golf', 'Footwear', 'Outdoors', 'Fan Shop',
       'Technology', 'Book Shop', 'Discs Shop', 'Pet Shop',
       'Health and Beauty ')
    
    product_price = st.number_input('Product Price')
    order_item_total = st.number_input('Order Item Total')
    order_item_discount = st.number_input('Order Item Discount')
    department_name = st.selectbox("Department Name", departments)

    ok = st.button("Predict Sales")
    if ok:
        X = np.array([[product_price,order_item_total, order_item_discount, department_name]])
        X[:,3] = le_department_name.transform(X[:,3])
        X = X.astype(float)

        sales = regressor_loaded.predict(X)
        st.success(f"Predicted Sales price is ${sales[0]:.2f}")

show_predict_page()