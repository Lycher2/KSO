# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import yfinance as yf
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objects as go

st.title("Facebook Prophet - Stock Price Prediction")
st.write("This is a demonstrator app created on Streamlit to show a use case of the Facebook Prophet "
         "timeseries model to forecast stock market data.")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.experimental_memo
def load_stock_data(ticker:str):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

tickers = ["VOO", "AAPL", "GOOG", "TSLA", "MSFT", "GME"]

ticker = st.selectbox("Select tickers", tickers)

if "years_slider"not in st.session_state:
    st.session_state.years_slider = 1

n_years = st.slider("Years of prediction:", 1, 5, key="years_slider")
period = n_years * 365

data_load_state = st.text("Loading... {} data".format(ticker))
data = load_stock_data(ticker)
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
    df_train = data[["Date", "Close"]]
    df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})
    prediction_progress = st.progress(0)
    prediction_progress.progress(10)

    model = Prophet()
    model.fit(df_train)

    prediction_progress.progress(20)

    future = model.make_future_dataframe(periods=period)

    prediction_progress.progress(40)

    forecast = model.predict(future)

    prediction_progress.progress(80)

    st.subheader("Forecast")

    st.write("Forecast timeseries")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = model.plot_components(forecast)
    st.write(fig2)

    prediction_progress.progress(100)

    st.success('Successfully forecasted {} stock close price'.format(ticker))

make_prediction()

if __name__ == '__main__':
    print("Run `streamlit run main.py` to run this app")
