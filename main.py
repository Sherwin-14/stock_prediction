import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly


START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "BRK.A",
    "JNJ",
    "PG",
    "KO",
    "MCD",
    "INTC",
    "CSCO",
    "MMM",
    "V",
    "MA",
    "WMT",
    "XOM",
    "CVX",
    "T",
    "VZ",
    "HD",
    "UNH",
    "PFE",
    "IBM",
    "ORCL",
    "C",
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "AXP",
    "CAT",
    "DE",
    "UPS",
    "FDX",
    "AIG",
    "TRV",
    "PRU",
    "MET",
    "LNC",
    "AFL",
    "MMC",
    "CB",
    "HIG",
    "ALL",
    "PNC",
    "USB",
    "CMCSA",
    "CHTR",
    "TWC",
    "S",
    "VOD",
    "TMUS",
    "CTL",
    "F",
    "GM",
    "FCAU",
    "HMC",
    "TM",
    "NSANY",
    "RDS.A",
    "TOT",
    "BP",
    "XRX",
    "HPQ",
    "DHI",
    "LEN",
    "PHM",
    "TOL",
    "NVR",
    "MDC",
    "RYL",
    "NKE",
    "UA",
    "LULU",
    "GPS",
    "TJX",
    "ROST",
    "M",
    "JWN",
    "KSS",
    "DDS",
    "BBY",
    "COST",
    "TGT",
    "WMT",
    "LOW",
    "HD",
    "SHW",
    "MAS",
    "LLY",
    "MRK",
    "ABBV",
    "BMY",
    "GILD",
    "BIIB",
    "REGN"
)


selected_stock = st.selectbox("Select stock for prediction",stocks)
n_years = st.slider("Years of Prediction",1,4)
period = n_years * 365

@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace = True)
    print(data.head())
    return data


data_load_state = st.text("Load data ...")

data = load_data(selected_stock)

data_load_state.text("Loading data ... done!")


st.subheader("Raw Data")
st.dataframe(data.head().style.set_table_styles([{'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]}]).background_gradient(cmap='Blues'), use_container_width=True)

st.subheader("Plotting The Data")

def plot_raw_data():
    df = pd.DataFrame(data)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume','Index']
    df = pd.melt(df, id_vars='Date', value_vars=['Open','Close'])
    fig = px.line(df, x='Date', y='value', color = 'variable')
    fig.update_layout(title_text = "Date v/s Closing Price", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)


plot_raw_data()    

# Forecasting with fb prophet

data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns] 
data.rename(columns={'Date_': 'Date'}, inplace=True) 

for col in data.columns: 
    if col.startswith('Close_'): 
        data.rename(columns={col: 'Close'}, inplace=True)

data['Date'] = data['Date'].dt.tz_localize(None)

train = data[['Date','Close']]
train['Date'] = pd.to_datetime(train['Date']) # Ensure dates are in correct format
train = train.rename(columns = {"Date":"ds","Close":"y"})

m = Prophet()
try:
    m.fit(train) 
    future = m.make_future_dataframe(periods = period)
    forecast = m.predict(future) 
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
except Exception as e: 
    st.write(f"Error: {e}" )  


st.subheader("Forecast Data")
st.dataframe(forecast.tail().style.set_table_styles([{'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]}]).background_gradient(cmap='Blues'), use_container_width=True)


st.subheader("Forecast Data With Predicitions For The Coming Years")
fig1  = plot_plotly(m, forecast)
st.plotly_chart(fig1)


st.subheader('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)

