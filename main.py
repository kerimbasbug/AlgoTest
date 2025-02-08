import streamlit as st
import pandas as pd
import yfinance as yf
import base64
import matplotlib.pyplot as mplt
import numpy as np
import os
from datetime import datetime, timedelta

st.set_page_config(layout='centered')

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

class InputPage():
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.stock_last_close = self.stock.history(period="1d")["Close"].iloc[-1]

    def show_price(self):
        company_df = pd.read_csv('Data/companies_info.csv')
        company_info = company_df[company_df['ticker'] == self.ticker]
        
        if company_info.empty:
            st.error("Company information not found!")
            return
        
        company_name = company_info['short name'].values[0]
        company_sector = company_info['sector'].values[0]
        company_logo = encode_image_to_base64(f"Logos/{self.ticker}.png")

        stock_data = self.stock.history(period="3mo").reset_index(drop=False)
        last_day_return = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]
        plot_color = 'green' if last_day_return >= 0 else 'red'

        # Generate price plot
        mplt.figure(figsize=(2.5, 1.5))
        mplt.plot(stock_data['Date'], stock_data['Close'], color=plot_color, label='Stock Price')
        mplt.fill_between(stock_data['Date'], stock_data['Close'], color=plot_color, alpha=0.25)
        mplt.ylim(stock_data['Close'].min() * 0.9, stock_data['Close'].max() * 1.1)
        mplt.axis('off')
        mplt.savefig('stock_plot.png', bbox_inches='tight', pad_inches=0)
        
        plot_base64 = encode_image_to_base64('stock_plot.png')

        # Display stock info with HTML
        st.markdown(
            f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; color: rgba(54, 53, 53, 0.8); font-family: Arial; text-align: left; max-width: 800px; margin: 10px auto; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);">
                <div style="display: flex; align-items: center;">
                    <img src="{company_logo}" style="max-width: 80px; max-height: 80px; margin-right: 30px; object-fit: contain;">
                    <div>
                        <h3 style="margin: 0; padding: 0; line-height: 1.4;">{company_name}</h3>
                        <h4 style="margin: 0; padding: 0; line-height: 1.4;">{self.ticker}</h4>
                        <h7 style="margin: 0; padding: 0; line-height: 1.2;">{company_sector}</h7>
                    </div>
                    <img src="{plot_base64}" style="max-width: 300px; max-height: 200px; position: absolute; top: 10px; right: 30px; object-fit: contain;">
                </div>
                <hr style="border: none; border-top: 1px solid #ccc; margin: 10px 0;">
                <p style="margin: 0; text-align: center; display: flex; justify-content: space-evenly; align-items: center;">
                    <span style="display: flex; align-items: center;">
                        <b>Last Close Price:</b><span style="margin-left: 2px;">${stock_data['Close'].iloc[-1]:.1f}&nbsp;&nbsp;</span>
                        <span style="color: {plot_color};">
                                ({100 * last_day_return:.2f}%)
                        </span>
                    </span>
                    </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    def run(self):
        st.title("Strategy Backtester")
        self.show_price()

        sp100_tickers = pd.read_csv('Data/companies_info.csv')['ticker']

        def update_ticker():
            st.session_state.selected_ticker = st.session_state.ticker_select

        st.selectbox(
            "Select Ticker",
            sorted(sp100_tickers),
            index=sorted(sp100_tickers).index(st.session_state.selected_ticker),
            key="ticker_select",
            on_change=update_ticker
        )

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_date = st.date_input("Start Date",
                                        value=datetime.today()-timedelta(days=365),
                                        min_value=pd.to_datetime('2010-01-01'),
                                        max_value=datetime.today())
        with col2:
            st.session_state.end_date = st.date_input("End Date",
                                    value=datetime.today(),
                                    min_value=st.session_state.start_date,
                                    max_value=datetime.today())

        st.session_state.cash = st.number_input("Investment Amount ($):", min_value=0.0, value=10000.0, format="%.2f")

        st.markdown(
            """
            <style>
                div.stButton > button {
                    width: 100%; /* Makes the button full-width */
                    padding: 10px; /* Adjust padding for better appearance */
                    font-size: 16px; /* Increase font size if needed */
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Test your own strategy", expanded=True):
                st.markdown("##### Bolinger Bands")
                st.session_state.sma_period = st.slider("SMA Period (Days)", 5, 50, 21)
                st.session_state.std_dev_multiplier = st.slider("Std Dev Multiplier", 1.0, 3.0, 2.0, step=0.25)
                st.markdown("##### RSI")
                st.session_state.rsi_window = st.slider("RSI Window (Days)", 5, 30, 14)
            if st.button('Backtest', type="primary"):
                st.switch_page('pages/Backtest.py')

        with col2:
            with st.expander("Optimze Parameters", expanded=True):
                st.markdown("##### Bolinger Bands")
                st.session_state.min_sma, st.session_state.max_sma = st.select_slider(
                    "Range of SMA Period (Days)",
                    options=list(np.arange(5,51,1)),
                    value=(14,21),
                )
                st.session_state.min_bolinger, st.session_state.max_bolinger = st.select_slider(
                    "Range of Std Dev Multiplier",
                    options=list(np.arange(1,3.25,0.25)),
                    value=(1.5,2.5),
                )
                st.markdown("##### RSI")
                st.session_state.min_rsi, st.session_state.max_rsi = st.select_slider(
                    "Range of RSI Period (Days)",
                    options=list(np.arange(1,31,1)),
                    value=(7,14),
                )
            if st.button('Optimize', type="primary"):
                st.switch_page('pages/Optimize.py')
                
if __name__ == "__main__":
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = 'AAPL'

    page = InputPage(st.session_state.selected_ticker)
    page.run()
