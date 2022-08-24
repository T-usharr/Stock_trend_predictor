
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

# from fbprophet import Prophet
print('Running..')
nifty_50_stocks=('ADANIPORTS','APOLLOHOSP','ASIANPAINT','AXISBANK','BAJAJ-AUTO','BAJFINANCE',
'BAJAJFINSV','BPCL','BHARTIARTL','BRITANNIA','CIPLA','COALINDIA','DIVISLAB','DRREDDY',
'EICHERMOT','GRASIM','HCLTECH','HDFCBANK','HDFCLIFE','HEROMOTOCO','HINDALCO','HINDUNILVR',
'HDFC','ICICIBANK','ITC','INDUSINDBK','INFY','JSWSTEEL','KOTAKBANK','LT','MM','MARUTI',
'NTPC','NESTLEIND','ONGC','POWERGRID','RELIANCE','SBILIFE','SHREECEM','SBIN','SUNPHARMA',
'TCS','TATACONSUM','TATAMOTORS','TATASTEEL','TECHM','TITAN','UPL','ULTRACEMCO','WIPRO')

st.success('')
st.title('*Stock Forecast App📈*')
st.success('')
# Sidebar
st.sidebar.error('**Select Time Frame Of Dataset**')
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))


# Select box
st.info("**Select Stock to Forecast**")
selected_stock=st.selectbox('',nifty_50_stocks)

# stocks details part 
data_load_state = st.text('Loading data...')
data = yf.download(selected_stock+'.NS',start=start_date,end=end_date)
data.reset_index(inplace=True)
st.success(' Current data of   '+selected_stock)
st.dataframe(data.tail(50).sort_index(ascending=False),width=5100,height=300)
data_load_state.text('Loading data... done!')

# curent stock data chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
fig.layout.update(title_text='Stock Present Chart', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)
st.info('')

# FORECAST DATA 
st.warning('Select Years for forecasting')
n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365

data_train = data[['Date','Close']]
data_train = data_train.rename(columns={"Date": "date", "Close": "close"})

m = Prophet()
m.fit(data_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.header('Forecasted data')
st.write(forecast.tail())

# forcast chart
st.error(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)



