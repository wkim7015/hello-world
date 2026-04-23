import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pykrx import stock
from pykrx.website.comm import webio
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

# ==============================================================================
# 0. KRX LOGIN & SESSION SETUP
# ==============================================================================
KRX_LOGIN_ID = os.environ.get("KRX_ID", "u5125048")
KRX_LOGIN_PW = os.environ.get("KRX_PW", "q6glq92j1!")
_krx_session = requests.Session()

def _session_post_read(self, **params):
    return _krx_session.post(self.url, headers=self.headers, data=params, verify=False)

def _session_get_read(self, **params):
    return _krx_session.get(self.url, headers=self.headers, params=params, verify=False)

webio.Post.read = _session_post_read
webio.Get.read = _session_get_read

def login_krx(login_id, login_pw):
    P = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
    J = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
    U = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
    A = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    _krx_session.get(P, headers={"User-Agent": A}, timeout=15, verify=False)
    _krx_session.get(J, headers={"User-Agent": A, "Referer": P}, timeout=15, verify=False)
    r = _krx_session.post(U, data={"mbrNm": "", "telNo": "", "di": "", "certType": "", "mbrId": login_id, "pw": login_pw}, headers={"User-Agent": A, "Referer": P}, timeout=15, verify=False)
    return r.json().get("_error_code", "") == "CD001"

# ==============================================================================
# 1. DATA ACQUISITION
# ==============================================================================
def download_data():
    login_krx(KRX_LOGIN_ID, KRX_LOGIN_PW)

    start_date_obj = datetime.today() - timedelta(days=365*10)
    end_date_obj = datetime.today()
    start_date = start_date_obj.strftime("%Y%m%d")
    end_date = end_date_obj.strftime("%Y%m%d")

    os.makedirs("data", exist_ok=True)
    print("Checking point-in-time universe history...")
    if not os.path.exists("historical_universe.csv"):
        print("Rebuilding historical point-in-time universe...")
        all_historical_tickers = set()
        universe_history = []
        current_date = start_date_obj
        while current_date <= end_date_obj:
            cd_str = current_date.strftime("%Y%m%d")
            try:
                kospi_caps = stock.get_market_cap(cd_str, market="KOSPI")
                if kospi_caps is None or kospi_caps.empty:
                    current_date += timedelta(days=1)
                    continue
                kosdaq_caps = stock.get_market_cap(cd_str, market="KOSDAQ")

                kospi_top = kospi_caps.sort_values("시가총액", ascending=False).head(150).index.tolist()
                kosdaq_top = kosdaq_caps.sort_values("시가총액", ascending=False).head(30).index.tolist()

                top_tickers = set(kospi_top + kosdaq_top)
                all_historical_tickers.update(top_tickers)

                for t in top_tickers:
                    universe_history.append({'Date': cd_str, 'Ticker': t, 'InUniverse': True})
            except Exception:
                pass
            current_date += timedelta(days=90)

        universe_df = pd.DataFrame(universe_history)
        universe_df['Date'] = pd.to_datetime(universe_df['Date'])
        universe_df.set_index('Date').to_csv("historical_universe.csv")
    else:
        universe_df = pd.read_csv("historical_universe.csv")
        all_historical_tickers = universe_df['Ticker'].unique().tolist()

    missing_tickers = [t for t in all_historical_tickers if not os.path.exists(f"data/{t}.csv")]
    print(f"Total unique historical tickers: {len(all_historical_tickers)}, Missing: {len(missing_tickers)}")

    def fetch_ticker(ticker):
        file_path = f"data/{ticker}.csv"
        if os.path.exists(file_path):
            return
        try:
            df_ohlcv = stock.get_market_ohlcv(start_date, end_date, ticker)
            df_fund = stock.get_market_fundamental(start_date, end_date, ticker)
            df_vol = stock.get_market_trading_volume_by_date(start_date, end_date, ticker)
            if df_ohlcv is not None and not df_ohlcv.empty:
                df = df_ohlcv.join(df_fund, how='left').join(df_vol, how='left')
                df.to_csv(file_path)
        except Exception:
            pass

    if missing_tickers:
        print("Downloading missing stock data...")
        with ThreadPoolExecutor(max_workers=30) as executor:
            list(tqdm(executor.map(fetch_ticker, missing_tickers), total=len(missing_tickers)))

    if not os.path.exists("benchmarks.csv"):
        print("Fetching Benchmarks...")
        kospi_idx = fdr.DataReader('KS11', start_date, end_date)
        kosdaq_idx = fdr.DataReader('KQ11', start_date, end_date)
        sp500_idx = fdr.DataReader('US500', start_date, end_date)

        benchmarks = pd.DataFrame({
            'KOSPI': kospi_idx['Close'],
            'KOSDAQ': kosdaq_idx['Close'],
            'SP500': sp500_idx['Close']
        })
        benchmarks.fillna(method='ffill', inplace=True)
        benchmarks.to_csv("benchmarks.csv")

    print("Data Download Complete.")

# ==============================================================================
# 2. VECTORIZED BACKTESTER
# ==============================================================================
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

        print("Loading data into memory for backtesting...")
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

    def apply_universe_mask(self, base_weights):
        if not os.path.exists("historical_universe.csv"):
            return base_weights
        univ_df = pd.read_csv("historical_universe.csv", parse_dates=['Date'])
        univ_df['Value'] = True
        mask = univ_df.pivot(index='Date', columns='Ticker', values='Value').fillna(False)
        mask = mask.reindex(index=base_weights.index, columns=base_weights.columns)
        mask = mask.ffill().fillna(False)
        mask.columns = mask.columns.astype(str)
        base_weights.columns = base_weights.columns.astype(str)
        return base_weights.multiply(mask, axis=0)

    def run_backtest(self, weights_df):
        weights_df = self.apply_universe_mask(weights_df)
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
            return {'CAGR': 0, 'MDD': 0, 'Alpha (KOSPI)': 0, 'Beta (KOSPI)': 0, 'Consistency': 0, 'Yearly Returns': pd.Series()}

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

        yearly = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        consistency = (yearly > 0).mean() if len(yearly) > 0 else 0

        return {
            'CAGR': cagr, 'MDD': mdd,
            'Alpha (KOSPI)': ak, 'Beta (KOSPI)': bk,
            'Alpha (KOSDAQ)': aq, 'Beta (KOSDAQ)': bq,
            'Alpha (SP500)': as5, 'Beta (SP500)': bs5,
            'Consistency': consistency, 'Yearly Returns': yearly,
            'port_series': port_series
        }

# ==============================================================================
# 3. STRATEGY GENERATION & WALK-FORWARD VALIDATION
# ==============================================================================
def run_strategy():
    bt = VectorizedBacktester()

    prices = bt.prices_df
    returns = prices.pct_change().fillna(0)
    volumes = pd.DataFrame(bt.volume).ffill().bfill()
    kospi_returns = bt.benchmark_returns['KOSPI'].fillna(0)

    # Train / Test split
    split_date = prices.index[int(len(prices) * 0.6)]

    def rank_cs(df):
        return df.rank(axis=1, pct=True)

    def zscore_cs(df):
        return df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1)

    mom_3m = prices.pct_change(60)
    mom_6m = prices.pct_change(120)
    mom_12m = prices.pct_change(240)

    inv_vol_1m = 1.0 / (returns.rolling(20).std() + 1e-8)
    inv_vol_3m = 1.0 / (returns.rolling(60).std() + 1e-8)

    per = pd.DataFrame(index=prices.index, columns=prices.columns)
    pbr = pd.DataFrame(index=prices.index, columns=prices.columns)
    for ticker in bt.fundamentals:
        if 'PER' in bt.fundamentals[ticker].columns:
            per[ticker] = bt.fundamentals[ticker]['PER']
        if 'PBR' in bt.fundamentals[ticker].columns:
            pbr[ticker] = bt.fundamentals[ticker]['PBR']

    earning_yield = 1.0 / per.replace(0, np.nan)
    book_yield = 1.0 / pbr.replace(0, np.nan)

    inst_sum = pd.DataFrame(index=prices.index, columns=prices.columns, data=0.0)
    for ticker in bt.institutions:
        if '기관합계' in bt.institutions[ticker].columns:
            inst_sum[ticker] = bt.institutions[ticker]['기관합계']
    inst_buy_3m = inst_sum.rolling(60).sum() / (volumes.rolling(60).sum() + 1e-8)

    market_vol = kospi_returns.rolling(20).std().fillna(0)
    market_vol_median = market_vol.rolling(252).median().fillna(method='bfill')
    is_low_vol_regime = market_vol < (market_vol_median * 1.2)

    ma_100 = bt.benchmarks['KOSPI'].rolling(100).mean()
    ma_200 = bt.benchmarks['KOSPI'].rolling(200).mean()

    kospi_trend_safe_100 = bt.benchmarks['KOSPI'] > ma_100
    kospi_trend_safe_200 = bt.benchmarks['KOSPI'] > ma_200

    strict_regime = is_low_vol_regime & kospi_trend_safe_200

    factor_dict = {
        'mom_3m': mom_3m, 'mom_6m': mom_6m, 'mom_12m': mom_12m,
        'inv_vol_1m': inv_vol_1m, 'inv_vol_3m': inv_vol_3m,
        'earning_yield': earning_yield, 'book_yield': book_yield,
        'inst_buy_3m': inst_buy_3m
    }

    factor_names = list(factor_dict.keys())

    def generate_weights(factors, top_n, sizing, freq, filter_type):
        combined_score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for f in factors:
            df_f = factor_dict[f].fillna(0)
            combined_score += zscore_cs(df_f).fillna(0)

        ranks = combined_score.rank(axis=1, ascending=False)
        selected = (ranks <= top_n)

        if sizing == 'risk_parity':
            raw_weights = selected.astype(float) * inv_vol_1m.fillna(0)
        elif sizing == 'vol_weighted':
            raw_weights = selected.astype(float) * inv_vol_3m.fillna(0)
        else:
            raw_weights = selected.astype(float)

        base_weights = raw_weights.div(raw_weights.sum(axis=1) + 1e-8, axis=0).fillna(0)
        reb_weights = base_weights.iloc[::freq].reindex(prices.index).fillna(method='ffill')

        if filter_type == 'strict_cash':
            final_weights = reb_weights.multiply(strict_regime, axis=0)
        elif filter_type == 'low_vol_cash':
            final_weights = reb_weights.multiply(is_low_vol_regime, axis=0)
        elif filter_type == 'ma200_cash':
            final_weights = reb_weights.multiply(kospi_trend_safe_200, axis=0)
        elif filter_type == 'ma100_cash':
            final_weights = reb_weights.multiply(kospi_trend_safe_100, axis=0)
        else:
            final_weights = reb_weights

        return final_weights

    import random
    random.seed(42)

    best_train_score = -9999
    best_params = {}

    print("Running Walk-Forward Optimization (In-Sample Training)...")
    for i in tqdm(range(100)):
        num_factors = random.randint(2, 4)
        selected_factors = random.sample(factor_names, num_factors)
        top_n = random.choice([5, 10, 15, 20])
        sizing = random.choice(['equal', 'risk_parity', 'vol_weighted'])
        freq = random.choice([10, 20])
        filter_type = random.choice(['strict_cash', 'low_vol_cash', 'ma200_cash', 'ma100_cash', 'none'])

        weights = generate_weights(selected_factors, top_n, sizing, freq, filter_type)
        metrics = bt.run_backtest(weights)
        train_series = metrics['port_series'].loc[:split_date]

        train_returns = train_series.pct_change().dropna()
        if len(train_returns) == 0: continue

        days = (train_series.index[-1] - train_series.index[0]).days
        years = days / 365.25
        if years <= 0: continue

        cagr = (train_series.iloc[-1] / train_series.iloc[0]) ** (1 / years) - 1
        roll_max = train_series.cummax()
        mdd = (train_series / roll_max - 1.0).min()

        train_kospi_returns = bt.benchmark_returns['KOSPI'].loc[train_returns.index].fillna(0)
        cov = np.cov(train_returns, train_kospi_returns)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0

        beta_penalty = 0 if beta <= 0.5 else (beta - 0.5) * 2.0
        mdd_penalty = 0 if mdd >= -0.20 else abs(mdd + 0.20) * 3.0

        score = cagr - mdd_penalty - beta_penalty

        if score > best_train_score:
            best_train_score = score
            best_params = {
                'factors': selected_factors,
                'top_n': top_n,
                'sizing': sizing,
                'freq': freq,
                'filter_type': filter_type
            }

    print("\nRunning Out-of-Sample Validation...")
    final_weights = generate_weights(**best_params)
    full_metrics = bt.run_backtest(final_weights)

    print("\n==================================================")
    print("FINAL STRATEGY (Walk-Forward Validated - Unbiased)")
    print("==================================================")
    print(f"Overall CAGR: {full_metrics['CAGR']:.2%}")
    print(f"Overall MDD: {full_metrics['MDD']:.2%}")
    print(f"Overall Beta (KOSPI): {full_metrics['Beta (KOSPI)']:.2f}")

    test_series = full_metrics['port_series'].loc[split_date:]
    test_days = (test_series.index[-1] - test_series.index[0]).days
    test_years = test_days / 365.25
    test_cagr = (test_series.iloc[-1] / test_series.iloc[0]) ** (1 / test_years) - 1
    test_roll_max = test_series.cummax()
    test_mdd = (test_series / test_roll_max - 1.0).min()

    print("\n--- Out-of-Sample Performance ---")
    print(f"Test CAGR: {test_cagr:.2%}")
    print(f"Test MDD: {test_mdd:.2%}")

    final_weights.to_csv("best_weights.csv")
    full_metrics['port_series'].to_csv("portfolio_values.csv")

    trades = []
    holdings = final_weights.shift(1).fillna(0)
    for i in range(1, len(final_weights)):
        prev_w = holdings.iloc[i]
        curr_w = final_weights.iloc[i]
        diff = curr_w - prev_w
        date = final_weights.index[i]
        for ticker in diff.index:
            if abs(diff[ticker]) > 0.001:
                trades.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Weight_Change': diff[ticker],
                    'Action': 'BUY' if diff[ticker] > 0 else 'SELL'
                })

    pd.DataFrame(trades).to_csv("trades.csv", index=False)
    print("\nOutputs saved to best_weights.csv, portfolio_values.csv, and trades.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantitative Strategy Backtester")
    parser.add_argument("--download", action="store_true", help="Download missing historical data")
    parser.add_argument("--optimize", action="store_true", help="Run strategy optimization and backtest")
    args = parser.parse_args()

    if args.download:
        download_data()
    elif args.optimize:
        run_strategy()
    else:
        print("Please specify an action: --download or --optimize")
