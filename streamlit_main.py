# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import yfinance as yf
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objects as go

st.title("Facebook Prophet - Crypto Price Prediction")
st.write("This is a demonstrator app created on Streamlit to show a use case of the Facebook Prophet "
         "timeseries model to forecast crypto market data.")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.experimental_memo
def load_crypto_data(ticker:str):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

tickers = ["BTC-USD","ETH-USD"]
ticker = st.selectbox("Select tickers", tickers)

if "months_slider"not in st.session_state:
    st.session_state.months_slider = 1

n_month = st.slider("months of prediction:", 1, 60, key="months_slider")
period = n_month * 30

data_load_state = st.text("Loading... {} data".format(ticker))
data = load_crypto_data(ticker)
data_load_state.text("Loading... Complete!")
st.write(data.tail())

def plot_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name="High"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name="Low"))
    fig_title = "{} data from {} to {}".format(ticker, START, TODAY)
    fig.update_layout(title_text=fig_title, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_data()

def make_prediction():
    df_train = data[["Date", "Adj Close"]]
    df_train['Date'] = df_train['Date'].dt.tz_localize(None)
    df_train = df_train.rename(columns={"Date":"ds", "Adj Close":"y"})
    prediction_progress = st.progress(0)
    prediction_progress.progress(10)

    pred = Prophet()
    pred.fit(df_train)

    prediction_progress.progress(20)

    future = pred.make_future_dataframe(periods=period)

    prediction_progress.progress(40)

    forecast_val = pred.predict(future)

    prediction_progress.progress(80)

    st.subheader("Forecast")

    st.write("Forecast timeseries")
    fig1 = plot_plotly(pred, forecast_val)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = pred.plot_components(forecast_val)
    st.write(fig2)

    prediction_progress.progress(100)

    st.success('Successfully forecasted {} stock close price'.format(ticker))

make_prediction()
if __name__ == '__main__':
    print("Run `streamlit run main.py` to run this app")
