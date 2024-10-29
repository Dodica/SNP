from AlgorithmImports import *
import numpy as np


class SMACrossAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 10, 28)
        self.SetCash(10000)

        self.ticker = "SPY"
        self.symbol = self.AddEquity(self.ticker, Resolution.Daily).Symbol
        self.short_window = 7
        self.long_window = 20

        self.short_sma = self.SMA(self.symbol, self.short_window, Resolution.Daily)
        self.long_sma = self.SMA(self.symbol, self.long_window, Resolution.Daily)

        self.previous_signal = None
        self.daily_portfolio_value = []

    def OnData(self, data):
        if not self.short_sma.IsReady or not self.long_sma.IsReady:
            return

        if self.short_sma.Current.Value > self.long_sma.Current.Value:
            if self.previous_signal != "long":
                self.SetHoldings(self.symbol, 1)
                self.previous_signal = "long"

        elif self.short_sma.Current.Value < self.long_sma.Current.Value:
            if self.previous_signal != "short":
                self.Liquidate(self.symbol)
                self.previous_signal = "short"

        self.daily_portfolio_value.append(self.Portfolio.TotalPortfolioValue)

    def OnEndOfAlgorithm(self):
        daily_returns = np.diff(self.daily_portfolio_value) / self.daily_portfolio_value[:-1]

        if len(daily_returns) > 1:
            avg_daily_return = np.mean(daily_returns)
            daily_return_std = np.std(daily_returns)

            if daily_return_std != 0:
                sharpe_ratio = (avg_daily_return / daily_return_std) * (252 ** 0.5)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        cumulative_returns = np.array(self.daily_portfolio_value) / self.daily_portfolio_value[0]
        drawdowns = 1 - cumulative_returns / np.maximum.accumulate(cumulative_returns)
        max_drawdown = np.max(drawdowns)

        self.Debug(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        self.Debug(f"Max Drawdown: {max_drawdown:.2%}")