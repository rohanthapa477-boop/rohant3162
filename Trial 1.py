import pandas as pd
from prophet import Prophet
import plotly.express as px
import numpy as np

# 1. Load your data
df = pd.read_csv('Experiment data.csv')

# 2. Create date and target columns for Prophet
df['ds'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

df['y'] = df['Number of products sold']   # numeric target

# 3. Fit Prophet model
model = Prophet()
model.fit(df[['ds', 'y']])

# 4. Make future dataframe and forecast
future = model.make_future_dataframe(periods=90)  # 90 more days
forecast = model.predict(future)

stock = 2000  # Starting stock (change as needed)
lead_days = 7
reorder_point = forecast['yhat'][-30:].mean() * lead_days  # Next month avg * lead

print("📊 INVENTORY SIMULATION (Next 60 Days):")
inventory_history = []
current_stock = stock

for i, row in forecast[-60:].iterrows():  # Simulate 2 months ahead
    current_stock -= row['yhat']  # Daily sales deduction
    inventory_history.append(current_stock)
    
    if current_stock < reorder_point:
        order_qty = reorder_point * 1.1  # 10% safety buffer
        current_stock += order_qty
        print(
            f"🚨 DAY {row['ds'].date()}: AUTO-ORDER {order_qty:.0f} units | New Stock: {current_stock:.0f}"
        )

# ===== UPGRADE 2: BETTER PLOTS =====
fig_forecast = px.line(forecast[-180:], x='ds', y='yhat', 
                       title='✅ Demand Forecast (Your Data!)')
fig_forecast.add_scatter(x=df['ds'][-60:], y=df['y'][-60:], 
                        name='Actual Sales', line=dict(color='red', width=3))
fig_forecast.show()

# Inventory plot
inv_df = pd.DataFrame({'date': forecast['ds'][-60:], 'stock': inventory_history})
fig_inv = px.line(inv_df, x='date', y='stock', 
                  title='📦 Inventory Levels + Auto-Reorder Triggers')
fig_inv.add_hline(y=reorder_point, line_dash="dash", line_color="red", 
                  annotation_text=f"Reorder Point: {reorder_point:.0f}")
fig_inv.show()

print(f"\n🎉 SUMMARY:")
print(f"   Reorder Point: {reorder_point:.0f} units")
print(f"   Final Stock: {current_stock:.0f} units")
print(f"   Orders Triggered: {len([s for s in inventory_history if s < reorder_point])}")
print("✅ INVENTORY + FORECASTING COMPLETE!")