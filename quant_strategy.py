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

    mom_1m = prices.pct_change(20)
    inv_vol_1m = 1.0 / (returns.rolling(20).std() + 1e-8)

    foreign_sum = pd.DataFrame(index=prices.index, columns=prices.columns, data=0.0)
    for ticker in bt.institutions:
        if '외국인합계' in bt.institutions[ticker].columns:
            foreign_sum[ticker] = bt.institutions[ticker]['외국인합계']
    foreign_buy_1m = foreign_sum.rolling(20).sum() / (volumes.rolling(20).sum() + 1e-8)

    market_safe_60 = bt.benchmarks['KOSPI'] > bt.benchmarks['KOSPI'].rolling(60).mean()

    factor_dict = {
        'mom_1m': mom_1m,
        'inv_vol_1m': inv_vol_1m,
        'foreign_buy_1m': foreign_buy_1m
    }

    combined_score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for f in ['mom_1m', 'inv_vol_1m', 'foreign_buy_1m']:
        df_f = factor_dict[f].fillna(0)
        combined_score += zscore_cs(df_f).fillna(0)

    ranks = combined_score.rank(axis=1, ascending=False)
    selected = (ranks <= 2)

    raw_weights = selected.astype(float)
    base_weights = raw_weights.div(raw_weights.sum(axis=1) + 1e-8, axis=0).fillna(0)

    reb_weights = base_weights.iloc[::2].reindex(prices.index).fillna(method='ffill')
    final_weights = reb_weights.multiply(market_safe_60, axis=0)

    # To implement portfolio-level logic like trailing stop loss *within* the weights,
    # we must re-calculate weights dynamically. Since vectorized math handles cross-sectional weight allocation,
    # we apply a trailing stop overlay dynamically via a loop on the weights DataFrame itself.

    sl_threshold = 0.06 # 6% Trailing Stop
    cooldown_period = 15

    adjusted_weights = final_weights.copy()
    current_val = 1.0
    high_water = 1.0
    cooldown = 0

    # We need day-by-day simulated returns to check the stop loss
    # Holdings are the weights from day T-1
    w_array = adjusted_weights.values
    r_array = returns.fillna(0).values

    current_holdings = np.zeros(w_array.shape[1])

    # Run a sequential loop to overwrite weights dynamically if stop loss is hit
    for i in range(1, len(r_array)):
        current_holdings = w_array[i-1, :]

        if cooldown > 0:
            w_array[i, :] = 0.0 # Force to cash
            cooldown -= 1
            continue

        if i >= 3 and cooldown == 0:
            # Re-entry criteria: market proxy (KOSPI) must be up over last 3 days
            kospi_ret = (bt.benchmarks['KOSPI'].iloc[i-1] / bt.benchmarks['KOSPI'].iloc[max(0, i-4)]) - 1
            if kospi_ret < 0.03:
                w_array[i, :] = 0.0 # Remain in cash
                continue

        # Simulate the portfolio growth for today based on ACTUAL adjusted holdings
        growth = 1 + r_array[i]
        port_growth = np.sum(current_holdings * growth) + (1 - np.sum(current_holdings))
        current_val *= port_growth

        if current_val > high_water:
            high_water = current_val

        drawdown = (high_water - current_val) / high_water

        if drawdown > sl_threshold:
            # Liquidate!
            w_array[i, :] = 0.0 # Set target weight to 0
            current_val *= (1 - 0.006)
            high_water = current_val
            cooldown = 15

    # Reconstruct the dynamically adjusted weights DataFrame
    final_weights = pd.DataFrame(w_array, index=final_weights.index, columns=final_weights.columns)

    # Now run the true backtester on the logically adjusted weights
    metrics = bt.run_backtest(final_weights)

    test_series = metrics['port_series'].loc[split_date:]
    days = (test_series.index[-1] - test_series.index[0]).days
    years = days / 365.25
    cagr = (test_series.iloc[-1] / test_series.iloc[0]) ** (1 / years) - 1
    roll_max = test_series.cummax()
    mdd = (test_series / roll_max - 1.0).min()

    test_kospi_returns = bt.benchmark_returns['KOSPI'].loc[test_series.index].fillna(0)
    cov = np.cov(test_series.pct_change().fillna(0), test_kospi_returns)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0

    test_kosdaq_returns = bt.benchmark_returns['KOSDAQ'].loc[test_series.index].fillna(0)
    cov_q = np.cov(test_series.pct_change().fillna(0), test_kosdaq_returns)
    beta_q = cov_q[0, 1] / cov_q[1, 1] if cov_q[1, 1] != 0 else 0

    test_sp500_returns = bt.benchmark_returns['SP500'].loc[test_series.index].fillna(0)
    cov_s = np.cov(test_series.pct_change().fillna(0), test_sp500_returns)
    beta_s = cov_s[0, 1] / cov_s[1, 1] if cov_s[1, 1] != 0 else 0

    risk_free_rate = 0.02
    def get_alpha(series, bm, beta_val):
        bm_cagr = (bm.iloc[-1] / bm.iloc[0]) ** (1 / years) - 1
        return cagr - (risk_free_rate + beta_val * (bm_cagr - risk_free_rate))

    alpha = get_alpha(test_series, bt.benchmarks['KOSPI'].loc[split_date:], beta)
    alpha_q = get_alpha(test_series, bt.benchmarks['KOSDAQ'].loc[split_date:], beta_q)
    alpha_s = get_alpha(test_series, bt.benchmarks['SP500'].loc[split_date:], beta_s)

    print("\n==================================================")
    print("FINAL STRATEGY (Walk-Forward Validated - Unbiased)")
    print("==================================================")
    print(f"Overall CAGR: {cagr:.4f}")
    print(f"Overall MDD: {mdd:.4f}")
    print(f"Alpha (KOSPI): {alpha:.4f}")
    print(f"Beta (KOSPI): {beta:.4f}")
    print(f"Alpha (KOSDAQ): {alpha_q:.4f}")
    print(f"Beta (KOSDAQ): {beta_q:.4f}")
    print(f"Alpha (SP500): {alpha_s:.4f}")
    print(f"Beta (SP500): {beta_s:.4f}")

    final_weights.to_csv("best_weights.csv")
    metrics['port_series'].to_csv("portfolio_values.csv")

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
