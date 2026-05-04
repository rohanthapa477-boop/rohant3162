import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import numpy as np

st.set_page_config(page_title="AI SCM Dashboard", layout="wide")

st.title("🛒 AI Supply Chain Predictor")
st.markdown("Upload CSV → Get forecasts + inventory alerts!")

# File upload
uploaded_file = st.file_uploader("Choose your SCM CSV file", type=['csv'])

if uploaded_file is not None:
    # LOAD + PROCESS (your exact code)
    df = pd.read_csv(uploaded_file)
    df['ds'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    df['y'] = df['Number of products sold']
    
    model = Prophet()
    model.fit(df[['ds', 'y']])
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    # INVENTORY
    stock = 2000
    lead_days = 7
    reorder_point = forecast['yhat'][-30:].mean() * lead_days
    inventory_history = []
    current_stock = stock
    
    for i, row in forecast[-60:].iterrows():
        current_stock -= row['yhat']
        inventory_history.append(current_stock)
        if current_stock < reorder_point:
            order_qty = reorder_point * 1.1
            current_stock += order_qty
            st.success(f"🚨 AUTO-ORDER: {order_qty:.0f} units!")
    
    # DASHBOARD LAYOUT
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Reorder Point", f"{reorder_point:.0f} units")
    col2.metric("📦 Current Stock", f"{current_stock:.0f} units")
    col3.metric("⏰ Lead Time", f"{lead_days} days")
    
    # GRAPHS
    fig1 = px.line(forecast[-180:], x='ds', y='yhat', title='Demand Forecast')
    fig2 = px.line(pd.DataFrame({'date':forecast['ds'][-60:], 'stock':inventory_history}), 
                   x='date', y='stock', title='Inventory Simulation')
    
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.balloons()  # Party! 🎉
else:
    st.info("👆 Upload your CSV to see magic!")