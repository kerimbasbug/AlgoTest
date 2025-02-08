import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from Strategy import run_test, rsi_val

class OptimizationApp:
    def __init__(self):
        st.set_page_config(layout='wide')
        st.title("Optimizer")
        self.display_parameters()
        self.optimize_parameters()
        self.display_results()

    def display_parameters(self):
        param_df = pd.DataFrame(
            {
                "Technical Indicator": ["SMA Period", "Std Dev Multiplier", "RSI Period"],
                "Min. Value": [st.session_state.min_sma, st.session_state.min_bolinger, st.session_state.min_rsi],
                "Max. Value": [st.session_state.max_sma, st.session_state.max_bolinger, st.session_state.max_rsi]
            }
        )

        _, center, _ = st.columns([10, 20, 10])
        center.dataframe(param_df, hide_index=True, width=600)

    def load_data(self, ticker, start_date, end_date):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            return stock_data
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

    def optimize_parameters(self):
        _, center, _ = st.columns([10, 20, 10])
        with center:
            with st.spinner('Optimizing parameters...'):
                opt_df = pd.DataFrame(columns=['sma_period', 'std_dev_multiplier', 'rsi_window', 'sharpe', 'ret'])
                
                for sma in range(st.session_state.min_sma, st.session_state.max_sma+1):
                    for std_dev in np.arange(st.session_state.min_bolinger, st.session_state.max_bolinger+0.25, 0.25):
                        for rsi_per in range(st.session_state.min_rsi, st.session_state.max_rsi+1):
                            temp = self.load_data(st.session_state.selected_ticker, st.session_state.start_date, st.session_state.end_date)
                            temp['SMA'] = temp['Close'].rolling(window=sma).mean()
                            temp['STD'] = temp['Close'].rolling(window=sma).std()
                            temp['Upper_Band'] = temp['SMA'] + (temp['STD'] * std_dev)
                            temp['Lower_Band'] = temp['SMA'] - (temp['STD'] * std_dev)
                            #temp['RSI'] = ta.rsi(temp['Close'], rsi)
                            temp['RSI'] = rsi_val(temp['Close'].to_numpy(), rsi_per)

                            temp_res = run_test(use_bollinger=True, use_rsi=True, stock_data=temp, cash=st.session_state.cash)
                            opt_df.loc[len(opt_df)] = [sma, std_dev, rsi_per, temp_res['Sharpe Ratio'], temp_res['Return [%]']]
                
                self.opt_sma = opt_df.sort_values(by='sharpe', ascending=False).iloc[0]['sma_period']
                self.opt_std_dev = opt_df.sort_values(by='sharpe', ascending=False).iloc[0]['std_dev_multiplier']
                self.opt_rsi = opt_df.sort_values(by='sharpe', ascending=False).iloc[0]['rsi_window']

        _, left, center, right, _ = st.columns([10, 5, 5, 5, 10])
        left.metric('Optimal SMA', int(self.opt_sma))
        center.metric('Optimal Std Dev', self.opt_std_dev)
        right.metric('Optimal RSI Window', self.opt_rsi)

    def display_results(self):
        stock_data = self.load_data(st.session_state.selected_ticker, st.session_state.start_date, st.session_state.end_date)
        stock_data['SMA'] = stock_data['Close'].rolling(window=int(self.opt_sma)).mean()
        stock_data['STD'] = stock_data['Close'].rolling(window=int(self.opt_sma)).std()
        stock_data['Upper_Band'] = stock_data['SMA'] + (stock_data['STD'] * self.opt_std_dev)
        stock_data['Lower_Band'] = stock_data['SMA'] - (stock_data['STD'] * self.opt_std_dev)
        #stock_data['RSI'] = ta.rsi(stock_data['Close'], int(self.opt_rsi))
        stock_data['RSI'] = rsi_val(stock_data['Close'].to_numpy(), int(self.opt_rsi))

        results = run_test(use_bollinger=True, use_rsi=True, stock_data=stock_data, cash=st.session_state.cash)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3])

        fig.add_trace(go.Scatter(x=stock_data.index, 
                                y=stock_data['Close'], 
                                mode='lines', 
                                name='Close Price', 
                                line=dict(color='white', width=1), opacity=1), row=1, col=1)


        fig.add_trace(go.Scatter(x=stock_data.index,
                                y=stock_data['SMA'],
                                mode='lines',
                                name='Sma',
                                line=dict(color='orange', width=0.5), opacity=1), row=1, col=1)

        fig.add_trace(go.Scatter(x=stock_data.index,
                                y=stock_data['Upper_Band'],
                                mode='lines',
                                name='Upper Bollinger Band',
                                line=dict(color='lightblue', width=0.5), opacity=1), row=1, col=1)

        fig.add_trace(go.Scatter(x=stock_data.index,
                                y=stock_data['Lower_Band'],
                                mode='lines',
                                name='Lower Bollinger Band',
                                line=dict(color='lightblue', width=0.5), opacity=1), row=1, col=1)

        fig.add_trace(go.Scatter(x=stock_data.index, 
                                y=stock_data['RSI'], 
                                mode='lines', 
                                name='RSI', 
                                line=dict(color='white', width=1), opacity=1), row=2, col=1)

        fig.add_hline(y=70, line=dict(color='orange', dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='orange', dash='dash', width=1), row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        fig.update_layout(
            height=800, 
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),  # Reduce margins
            autosize=True  # Let Plotly automatically adjust the size
        )

        for _, trade in results._trades.iterrows():
            line_color = 'green' if trade['PnL'] > 0 else 'red'
            entry_symbol = 'triangle-up' if trade['Size'] > 0 else 'triangle-down'
            exit_symbol = 'triangle-down' if trade['Size'] > 0 else 'triangle-up'

            # If the trade is a long position, add a green transparent highlight
            if trade['Size'] > 0:  
                fig.add_shape(
                    type="rect",
                    x0=trade['EntryTime'], x1=trade['ExitTime'],
                    y0=np.min([stock_data['Close'].min(), stock_data['Lower_Band'].min()]), y1=np.max([stock_data['Close'].max(), stock_data['Upper_Band'].max()]),  
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
                        np.min([stock_data['Close'].min(), stock_data['Lower_Band'].min()]),
                        np.min([stock_data['Close'].min(), stock_data['Lower_Band'].min()]),
                        np.max([stock_data['Close'].max(), stock_data['Upper_Band'].max()]),
                        np.max([stock_data['Close'].max(), stock_data['Upper_Band'].max()]),
                        np.min([stock_data['Close'].min(), stock_data['Lower_Band'].min()])
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
                    y0=np.min([stock_data['Close'].min(), stock_data['Lower_Band'].min()]), y1=np.max([stock_data['Close'].max(), stock_data['Upper_Band'].max()]),  
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
                        np.min([stock_data['Close'].min(), stock_data['Lower_Band'].min()]),
                        np.min([stock_data['Close'].min(), stock_data['Lower_Band'].min()]),
                        np.max([stock_data['Close'].max(), stock_data['Upper_Band'].max()]),
                        np.max([stock_data['Close'].max(), stock_data['Upper_Band'].max()]),
                        np.min([stock_data['Close'].min(), stock_data['Lower_Band'].min()])
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
        _, center, _ = st.columns([5, 40, 1])
        with center:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Equity", f"${results['Equity Final [$]']:.2f}")
            c2.metric("Return", f"{results['Return [%]']:.2f}%")
            c3.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
            c4.metric("Max Drawdown", f"{results['Max. Drawdown [%]']:.2f}%")

            c1.metric("Number of Trades", f"{results['# Trades']}")
            c2.metric("Win Rate", f"{results['Win Rate [%]']:.2f}%")
            c3.metric("Best Trade", f"{results['Best Trade [%]']:.2f}%")
            c4.metric("Worst Trade", f"{results['Worst Trade [%]']:.2f}%")

        trades_df = results._trades
        trades_df = trades_df.drop(['EntryBar', 'ExitBar', 'ReturnPct', 'Tag', 'Duration'], axis=1)
        trades_df['Position'] = trades_df['Size'].apply(lambda x: 'Long' if x >= 0 else 'Short')
        trades_df['EntryTime'] = trades_df['EntryTime'].dt.date
        trades_df['ExitTime'] = trades_df['ExitTime'].dt.date
        trades_df['Trade #'] = trades_df.index + 1
        trades_df = trades_df.rename(columns={'Size': 'Position Size', 'EntryTime': 'EntryDate', 'ExitTime': 'ExitDate'})
        trades_df = trades_df[['Trade #', 'EntryDate', 'ExitDate', 'Position', 'Position Size', 'EntryPrice', 'ExitPrice', 'PnL']]
        st.write('---')
        _, center, _ = st.columns([10, 40, 1])
        center.dataframe(trades_df, hide_index=True, width=800)

        if st.button('Restart', type="primary"):
            st.switch_page('main.py')

if __name__ == "__main__":
    OptimizationApp()