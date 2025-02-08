from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np

class HybridStrategy(Strategy):
    use_rsi = True
    use_bollinger = True

    @classmethod
    def set_parameters(cls, use_bollinger, use_rsi):
        cls.use_rsi = use_rsi
        cls.use_bollinger = use_bollinger

    def init(self):
        self.upper_band = self.I(lambda x: x['Upper_Band'], self.data.df)
        self.lower_band = self.I(lambda x: x['Lower_Band'], self.data.df)
        self.rsi = self.I(lambda x: x['RSI'], self.data.df)

    def next(self):
        long = False
        short = False

        if self.use_bollinger:
            if crossover(self.data.Close, self.lower_band):
                long = True
            elif crossover(self.upper_band, self.data.Close):
                short = True

        if self.use_rsi:
            if self.rsi[-1] < 30:
                long = True
            elif self.rsi[-1] > 70:
                short = True

        if long and not self.position.is_long:
            self.buy()
        elif short and not self.position.is_short:
            self.sell()

def run_test(use_bollinger, use_rsi, stock_data, cash):
    # Set strategy parameters before running the backtest
    HybridStrategy.set_parameters(use_rsi, use_bollinger)
    bt = Backtest(stock_data, HybridStrategy, cash=cash, exclusive_orders=True)
    stats = bt.run()
    return stats

def rsi_val(array, window):
    deltas = np.diff(array)
    gains = deltas[deltas > 0].sum() / window
    losses = -deltas[deltas < 0].sum() / window
    rs = gains / losses if losses != 0 else 0
    rsi = np.zeros_like(array)
    rsi[:window] = 100. - 100. / (1. + rs)

    for i in range(window, len(array)):
        delta = deltas[i - 1]
        gain = max(delta, 0)
        loss = max(-delta, 0)

        gains = (gains * (window - 1) + gain) / window
        losses = (losses * (window - 1) + loss) / window

        rs = gains / losses if losses != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi