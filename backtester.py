import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

class VectorizedBacktester:
    def __init__(self, data_path="data", benchmark_path="benchmarks.csv", sell_cost=0.006):
        self.data_path = data_path
        self.sell_cost = sell_cost

        self.benchmarks = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
        self.benchmarks.index = pd.to_datetime(self.benchmarks.index).tz_localize(None)

        self.prices = {}
        self.fundamentals = {}
        self.volume = {}
        self.institutions = {}

        for f in os.listdir(data_path):
            if f.endswith('.csv'):
                ticker = f.split('.')[0]
                df = pd.read_csv(os.path.join(data_path, f), index_col='날짜', parse_dates=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)

                if df.empty: continue
                df = df.ffill().bfill()

                if '종가' in df.columns:
                    self.prices[ticker] = df['종가']
                if '거래량' in df.columns:
                    self.volume[ticker] = df['거래량']

                fund_cols = [c for c in ['PER', 'PBR', 'EPS', 'BPS', 'DIV', 'DPS'] if c in df.columns]
                if fund_cols:
                    self.fundamentals[ticker] = df[fund_cols]

                inst_cols = [c for c in ['기관합계', '외국인합계', '개인'] if c in df.columns]
                if inst_cols:
                    self.institutions[ticker] = df[inst_cols]

        self.prices_df = pd.DataFrame(self.prices).ffill().bfill()
        self.returns_df = self.prices_df.pct_change()

        self.benchmarks = self.benchmarks.reindex(self.prices_df.index).ffill().bfill()
        self.benchmark_returns = self.benchmarks.pct_change()

    def run_backtest(self, weights_df):
        holdings = weights_df.shift(1).fillna(0)
        weights_array = holdings.values
        returns_array = self.returns_df.fillna(0).values

        port_values = np.zeros(len(returns_array))
        port_values[0] = 1.0

        current_weights = np.zeros(weights_array.shape[1])

        for i in range(1, len(returns_array)):
            growth = 1 + returns_array[i]
            prev_value = port_values[i-1]
            holdings_value = prev_value * current_weights * growth
            current_value_pre_rebal = np.sum(holdings_value) + prev_value * (1 - np.sum(current_weights))

            target_weights = weights_array[i]

            if current_value_pre_rebal > 0:
                actual_weights_pre_rebal = holdings_value / current_value_pre_rebal
            else:
                actual_weights_pre_rebal = np.zeros_like(current_weights)

            weight_diffs = target_weights - actual_weights_pre_rebal
            sells = np.where(weight_diffs < 0, weight_diffs, 0)
            cost = np.sum(np.abs(sells)) * self.sell_cost

            current_value_post_rebal = current_value_pre_rebal * (1 - cost)
            port_values[i] = current_value_post_rebal
            current_weights = target_weights

        port_series = pd.Series(port_values, index=self.prices_df.index)
        return self.calculate_metrics(port_series)

    def calculate_metrics(self, port_series):
        returns = port_series.pct_change().dropna()

        days = (port_series.index[-1] - port_series.index[0]).days
        if days == 0:
            return {'CAGR': 0, 'MDD': 0, 'Alpha (KOSPI)': 0, 'Beta (KOSPI)': 0}

        years = days / 365.25
        cagr = (port_series.iloc[-1] / port_series.iloc[0]) ** (1 / years) - 1

        roll_max = port_series.cummax()
        drawdown = port_series / roll_max - 1.0
        mdd = drawdown.min()

        def calc_ab(benchmark_col):
            cov = np.cov(returns, self.benchmark_returns[benchmark_col].loc[returns.index].fillna(0))
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
            b_cagr = (self.benchmarks[benchmark_col].iloc[-1] / self.benchmarks[benchmark_col].iloc[0]) ** (1 / years) - 1
            alpha = cagr - (0.02 + beta * (b_cagr - 0.02))
            return alpha, beta

        ak, bk = calc_ab('KOSPI')
        aq, bq = calc_ab('KOSDAQ')
        as5, bs5 = calc_ab('SP500')

        return {
            'CAGR': cagr,
            'MDD': mdd,
            'Alpha (KOSPI)': ak, 'Beta (KOSPI)': bk,
            'Alpha (KOSDAQ)': aq, 'Beta (KOSDAQ)': bq,
            'Alpha (SP500)': as5, 'Beta (SP500)': bs5,
            'port_series': port_series
        }

if __name__ == '__main__':
    bt = VectorizedBacktester()
    weights = pd.DataFrame(1.0 / len(bt.prices_df.columns), index=bt.prices_df.index, columns=bt.prices_df.columns)
    metrics = bt.run_backtest(weights)
    print(f"Dummy Strategy CAGR: {metrics['CAGR']:.2%}, MDD: {metrics['MDD']:.2%}, Beta (KOSPI): {metrics['Beta (KOSPI)']:.2f}")
