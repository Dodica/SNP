import yfinance as yf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

matplotlib.use('Agg')


def download_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)


def calculate_moving_averages(data, short_window, long_window):
    data[f'SMA{short_window}'] = data['Close'].rolling(window=short_window).mean()
    data[f'SMA{long_window}'] = data['Close'].rolling(window=long_window).mean()
    return data


def generate_signals(data, short_window, long_window):
    data['Signal'] = np.where(data[f'SMA{short_window}'] > data[f'SMA{long_window}'], 1, 0)
    data['Position'] = data['Signal'].shift()
    return data


def calculate_strategy_performance(data, capital):
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * data['Position'].shift(1)

    data['Equity_Curve'] = capital * (1 + data['Strategy_Returns']).cumprod()

    data['Cumulative_Max'] = data['Equity_Curve'].cummax()
    data['Drawdown'] = (data['Equity_Curve'] - data['Cumulative_Max']) / data['Cumulative_Max']

    if data['Strategy_Returns'].std() != 0:
        sharpe_ratio = data['Strategy_Returns'].mean() / data['Strategy_Returns'].std() * (252 ** 0.5)
    else:
        sharpe_ratio = 0

    return sharpe_ratio, data['Drawdown'].min()


def plot_equity_curve(data, filename):
    data['Equity_Curve'].plot(title="Equity Curve")
    plt.savefig(filename)


def main():
    ticker = 'SPY'
    start_date = "2020-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    short_window = 7
    long_window = 20
    filename = 'SMA-Cross\\equity_curve.png'
    capital = 10000

    data = download_data(ticker, start_date, end_date)

    data = calculate_moving_averages(data, short_window, long_window)

    data = generate_signals(data, short_window, long_window)

    sharpe_ratio, max_drawdown = calculate_strategy_performance(data, capital)

    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    plot_equity_curve(data, filename)

    # print(data.head(1200).to_string())


if __name__ == "__main__":
    main()
