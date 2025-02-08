import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from Strategy import run_test, rsi_val
import yfinance as yf

class BacktestApp:
    def __init__(self):
        st.set_page_config(layout='wide')
        st.title("Backtester")
        self.stock_data = self.load_data(
            st.session_state.selected_ticker, 
            st.session_state.start_date,
            st.session_state.end_date
        )
        self.calculate_indicators()
        self.results = run_test(
            use_bollinger=True, use_rsi=True, stock_data=self.stock_data, cash=st.session_state.cash
        )
        self.display_parameters()
        self.plot_results()
        self.display_metrics()
        self.display_trades()
        self.restart_button()

    def load_data(self, ticker, start_date, end_date):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            return stock_data
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
        
    def display_parameters(self):
        param_df = pd.DataFrame(
            {
                "Technical Indicator": ["SMA Period", "Std Dev Multiplier", "RSI Period"],
                "Value": [st.session_state.sma_period, st.session_state.std_dev_multiplier, st.session_state.rsi_window],
            }
        )

        _, center, _ = st.columns([10, 15, 10])
        center.dataframe(param_df, hide_index=True, width=600)

    def calculate_indicators(self):
        self.stock_data['SMA'] = self.stock_data['Close'].rolling(window=st.session_state.sma_period).mean()
        self.stock_data['STD'] = self.stock_data['Close'].rolling(window=st.session_state.sma_period).std()
        self.stock_data['Upper_Band'] = self.stock_data['SMA'] + (self.stock_data['STD'] * st.session_state.std_dev_multiplier)
        self.stock_data['Lower_Band'] = self.stock_data['SMA'] - (self.stock_data['STD'] * st.session_state.std_dev_multiplier)
        #self.stock_data['RSI'] = ta.rsi(self.stock_data['Close'], st.session_state.rsi_window)
        self.stock_data['RSI'] = rsi_val(self.stock_data['Close'].to_numpy(), st.session_state.rsi_window)

    def plot_results(self):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Plot price and indicators
        fig.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'], mode='lines', name='Close Price', line=dict(color='white', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['SMA'], mode='lines', name='SMA', line=dict(color='orange', width=0.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Upper_Band'], mode='lines', name='Upper Bollinger Band', line=dict(color='lightblue', width=0.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Lower_Band'], mode='lines', name='Lower Bollinger Band', line=dict(color='lightblue', width=0.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['RSI'], mode='lines', name='RSI', line=dict(color='white', width=1)), row=2, col=1)
        
        fig.add_hline(y=70, line=dict(color='orange', dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='orange', dash='dash', width=1), row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        fig.update_layout(height=800, showlegend=False, margin=dict(l=10, r=10, t=10, b=10), autosize=True)


        for _, trade in self.results._trades.iterrows():
            line_color = 'green' if trade['PnL'] > 0 else 'red'
            entry_symbol = 'triangle-up' if trade['Size'] > 0 else 'triangle-down'
            exit_symbol = 'triangle-down' if trade['Size'] > 0 else 'triangle-up'

            # If the trade is a long position, add a green transparent highlight
            if trade['Size'] > 0:  
                fig.add_shape(
                    type="rect",
                    x0=trade['EntryTime'], x1=trade['ExitTime'],
                    y0=np.min([self.stock_data['Close'].min(), self.stock_data['Lower_Band'].min()]), y1=np.max([self.stock_data['Close'].max(), self.stock_data['Upper_Band'].max()]),  
                    fillcolor="rgba(0, 256, 0, 0.1)",  # Transparent green
                    line=dict(width=0),
                    layer="below",
                )
                hower_text = f"LONG:<br>Entry: {trade['EntryPrice']:.2f}<br>Exit: {trade['ExitPrice']:.2f}<br>P&L: {trade['PnL']:.2f}"
                
                pnl_color = "green" if trade['PnL'] >= 0 else "red"
                hover_text = (
                    f"<span style='color:green'><b>LONG</b></span><br>"
                    f"Entry: {trade['EntryPrice']:.2f}<br>"
                    f"Exit: {trade['ExitPrice']:.2f}<br>"
                    f"Size: {trade['Size']:.2f}<br>"
                    f"P&L: <span style='color:{pnl_color}'>{trade['PnL']:.2f}</span>"
                )

                fig.add_trace(go.Scatter(
                    x=[trade['EntryTime'], trade['ExitTime'], trade['ExitTime'], trade['EntryTime'], trade['EntryTime']],
                    y=[
                        np.min([self.stock_data['Close'].min(), self.stock_data['Lower_Band'].min()]),
                        np.min([self.stock_data['Close'].min(), self.stock_data['Lower_Band'].min()]),
                        np.max([self.stock_data['Close'].max(), self.stock_data['Upper_Band'].max()]),
                        np.max([self.stock_data['Close'].max(), self.stock_data['Upper_Band'].max()]),
                        np.min([self.stock_data['Close'].min(), self.stock_data['Lower_Band'].min()])
                    ],
                    fill="toself",  # Fills the area
                    fillcolor="rgba(0, 256, 0, 0.1)",  # Same transparent color as the shape
                    line=dict(width=0),  # No outline
                    hoverinfo="text",
                    text=hover_text,
                    opacity=0.0  # Invisible, but captures hover events
                ))

            if trade['Size'] < 0:  
                fig.add_shape(
                    type="rect",
                    x0=trade['EntryTime'], x1=trade['ExitTime'],
                    y0=np.min([self.stock_data['Close'].min(), self.stock_data['Lower_Band'].min()]), y1=np.max([self.stock_data['Close'].max(), self.stock_data['Upper_Band'].max()]),  
                    fillcolor="rgba(256, 0, 0, 0.1)",  # Transparent green
                    line=dict(width=0),
                    layer="below"
                )

                pnl_color = "green" if trade['PnL'] >= 0 else "red"
                hover_text = (
                    f"<span style='color:red'><b>SHORT</b></span><br>"
                    f"Entry: {trade['EntryPrice']:.2f}<br>"
                    f"Exit: {trade['ExitPrice']:.2f}<br>"
                    f"Size: {trade['Size']:.2f}<br>"
                    f"P&L: <span style='color:{pnl_color}'>{trade['PnL']:.2f}</span>"
                )
                
                fig.add_trace(go.Scatter(
                    x=[trade['EntryTime'], trade['ExitTime'], trade['ExitTime'], trade['EntryTime'], trade['EntryTime']],
                    y=[
                        np.min([self.stock_data['Close'].min(), self.stock_data['Lower_Band'].min()]),
                        np.min([self.stock_data['Close'].min(), self.stock_data['Lower_Band'].min()]),
                        np.max([self.stock_data['Close'].max(), self.stock_data['Upper_Band'].max()]),
                        np.max([self.stock_data['Close'].max(), self.stock_data['Upper_Band'].max()]),
                        np.min([self.stock_data['Close'].min(), self.stock_data['Lower_Band'].min()])
                    ],
                    fill="toself",  # Fills the area
                    fillcolor="rgba(256, 0, 0, 0.1)",  # Same transparent color as the shape
                    line=dict(width=0),  # No outline
                    hoverinfo="text",
                    text=hover_text,
                    opacity=0.0
                ))

            # Add dashed line between entry and exit
            fig.add_trace(go.Scatter(
                x=[trade['EntryTime'], trade['ExitTime']], 
                y=[trade['EntryPrice'], trade['ExitPrice']],
                mode='lines', 
                line=dict(color=line_color, width=2.5, dash='dash'), 
                opacity=0.7, 
                name='Trade Line'
            ))

            # Add entry point
            fig.add_trace(go.Scatter(
                x=[trade['EntryTime']], 
                y=[trade['EntryPrice']],
                mode='markers', 
                marker=dict(color='yellow', size=10, symbol=entry_symbol),
                name='Entry Point'
            ))

            # Add exit point
            fig.add_trace(go.Scatter(
                x=[trade['ExitTime']], 
                y=[trade['ExitPrice']],
                mode='markers', 
                marker=dict(color='yellow', size=10, symbol=exit_symbol),
                name='Exit Point'
            ))

        st.plotly_chart(fig)

    def display_metrics(self):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Equity", f"${self.results['Equity Final [$]']:.2f}")
        c2.metric("Return", f"{self.results['Return [%]']:.2f}%")
        c3.metric("Sharpe Ratio", f"{self.results['Sharpe Ratio']:.2f}")
        c4.metric("Max Drawdown", f"{self.results['Max. Drawdown [%]']:.2f}%")
        
        c1.metric("Number of Trades", f"{self.results['# Trades']}")
        c2.metric("Win Rate", f"{self.results['Win Rate [%]']:.2f}%")
        c3.metric("Best Trade", f"{self.results['Best Trade [%]']:.2f}%")
        c4.metric("Worst Trade", f"{self.results['Worst Trade [%]']:.2f}%")

    def display_trades(self):
        trades_df = self.results._trades.drop(['EntryBar', 'ExitBar', 'ReturnPct', 'Tag', 'Duration'], axis=1)
        trades_df['Position'] = trades_df['Size'].apply(lambda x: 'Long' if x >= 0 else 'Short')
        trades_df['EntryTime'] = trades_df['EntryTime'].dt.date
        trades_df['ExitTime'] = trades_df['ExitTime'].dt.date
        trades_df['Trade #'] = trades_df.index + 1
        trades_df = trades_df.rename(columns={'Size': 'Position Size', 'EntryTime': 'EntryDate', 'ExitTime': 'ExitDate'})
        trades_df = trades_df[['Trade #', 'EntryDate', 'ExitDate', 'Position', 'Position Size', 'EntryPrice', 'ExitPrice', 'PnL']]
        
        st.write('---')
        left_space, center, right_space = st.columns([10, 40, 1])
        center.dataframe(trades_df, hide_index=True, width=800)

    def restart_button(self):
        if st.button('Restart', type="primary"):
            st.switch_page('main.py')

if __name__ == "__main__":
    BacktestApp()
