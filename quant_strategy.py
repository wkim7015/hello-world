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
KRX_LOGIN_ID = os.environ.get("KRX_ID", "")
KRX_LOGIN_PW = os.environ.get("KRX_PW", "")
_krx_session = requests.Session()

def _session_post_read(self, **params):
    return _krx_session.post(self.url, headers=self.headers, data=params, verify=True)

def _session_get_read(self, **params):
    return _krx_session.get(self.url, headers=self.headers, params=params, verify=True)

webio.Post.read = _session_post_read
webio.Get.read = _session_get_read

def login_krx(login_id, login_pw):
    P = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
    J = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
    U = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
    A = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    _krx_session.get(P, headers={"User-Agent": A}, timeout=15, verify=True)
    _krx_session.get(J, headers={"User-Agent": A, "Referer": P}, timeout=15, verify=True)
    r = _krx_session.post(U, data={"mbrNm": "", "telNo": "", "di": "", "certType": "",
                                    "mbrId": login_id, "pw": login_pw},
                          headers={"User-Agent": A, "Referer": P}, timeout=15, verify=True)
    return r.json().get("_error_code", "") == "CD001"

# ==============================================================================
# 1. DATA ACQUISITION
# ==============================================================================
def download_data():
    login_krx(KRX_LOGIN_ID, KRX_LOGIN_PW)

    start_date_obj = datetime.today() - timedelta(days=365 * 10)
    end_date_obj = datetime.today()
    start_date = start_date_obj.strftime("%Y%m%d")
    end_date   = end_date_obj.strftime("%Y%m%d")

    os.makedirs("data", exist_ok=True)
    print("Checking point-in-time universe history...")

    if not os.path.exists("historical_universe.csv"):
        print("Rebuilding historical point-in-time universe (KOSPI 200 + KOSDAQ 50)...")
        all_historical_tickers = set()
        universe_history = []
        current_date = start_date_obj

        while current_date <= end_date_obj:
            cd_str = current_date.strftime("%Y%m%d")
            try:
                kospi_caps  = stock.get_market_cap(cd_str, market="KOSPI")
                kosdaq_caps = stock.get_market_cap(cd_str, market="KOSDAQ")

                if kospi_caps is None or kospi_caps.empty:
                    current_date += timedelta(days=1)
                    continue

                kospi_top  = kospi_caps.sort_values("시가총액", ascending=False).head(200).index.tolist()
                kosdaq_top = kosdaq_caps.sort_values("시가총액", ascending=False).head(50).index.tolist()

                top_tickers = set(kospi_top + kosdaq_top)
                all_historical_tickers.update(top_tickers)

                for t in top_tickers:
                    universe_history.append({'Date': cd_str, 'Ticker': t})
            except Exception:
                pass
            current_date += timedelta(days=90)

        universe_df = pd.DataFrame(universe_history)
        universe_df['Date'] = pd.to_datetime(universe_df['Date'])
        universe_df.set_index('Date').to_csv("historical_universe.csv")
    else:
        universe_df = pd.read_csv("historical_universe.csv")
        all_historical_tickers = universe_df['Ticker'].unique().tolist()

    missing_tickers = [t for t in all_historical_tickers
                       if not os.path.exists(f"data/{t}.csv")]
    print(f"Total unique historical tickers: {len(all_historical_tickers)}, "
          f"Missing: {len(missing_tickers)}")

    def fetch_ticker(ticker):
        file_path = f"data/{ticker}.csv"
        if os.path.exists(file_path):
            return
        try:
            df_ohlcv = stock.get_market_ohlcv(start_date, end_date, ticker)
            df_fund  = stock.get_market_fundamental(start_date, end_date, ticker)
            df_vol   = stock.get_market_trading_volume_by_date(start_date, end_date, ticker)
            if df_ohlcv is not None and not df_ohlcv.empty:
                df = df_ohlcv.join(df_fund, how='left').join(df_vol, how='left')
                df.to_csv(file_path)
        except Exception:
            pass

    if missing_tickers:
        print("Downloading missing stock data...")
        with ThreadPoolExecutor(max_workers=30) as executor:
            list(tqdm(executor.map(fetch_ticker, missing_tickers),
                      total=len(missing_tickers)))

    if not os.path.exists("benchmarks.csv"):
        print("Fetching Benchmarks...")
        kospi_idx  = fdr.DataReader('KS11', start_date, end_date)
        kosdaq_idx = fdr.DataReader('KQ11', start_date, end_date)
        sp500_idx  = fdr.DataReader('US500', start_date, end_date)
        benchmarks = pd.DataFrame({
            'KOSPI':  kospi_idx['Close'],
            'KOSDAQ': kosdaq_idx['Close'],
            'SP500':  sp500_idx['Close']
        })
        benchmarks.ffill(inplace=True)
        benchmarks.to_csv("benchmarks.csv")

    print("Data Download Complete.")

# ==============================================================================
# 2. VECTORIZED BACKTESTER (Leakage Free)
# ==============================================================================
class VectorizedBacktester:
    def __init__(self, data_path="data", benchmark_path="benchmarks.csv",
                 sell_cost=0.006, buy_cost=0.00015):
        self.data_path  = data_path
        self.sell_cost  = sell_cost
        self.buy_cost   = buy_cost

        self.benchmarks = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
        self.benchmarks.index = pd.to_datetime(self.benchmarks.index).tz_localize(None)

        self.prices      = {}
        self.fundamentals = {}
        self.volume      = {}
        self.institutions = {}

        print("Loading data into memory...")
        for f in os.listdir(data_path):
            if not f.endswith('.csv'):
                continue
            ticker = f.split('.')[0]
            df = pd.read_csv(os.path.join(data_path, f),
                             index_col='날짜', parse_dates=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if df.empty:
                continue

            df = df.ffill()

            if '종가' in df.columns:
                self.prices[ticker] = df['종가']
            if '거래량' in df.columns:
                self.volume[ticker] = df['거래량']

            fund_cols = [c for c in ['PER', 'PBR', 'EPS', 'BPS', 'DIV', 'DPS']
                         if c in df.columns]
            if fund_cols:
                self.fundamentals[ticker] = df[fund_cols]

            inst_cols = [c for c in ['기관합계', '외국인합계', '개인']
                         if c in df.columns]
            if inst_cols:
                self.institutions[ticker] = df[inst_cols]

        self.prices_df   = pd.DataFrame(self.prices).ffill()
        self.returns_df  = self.prices_df.pct_change()

        self.benchmarks  = self.benchmarks.reindex(self.prices_df.index).ffill()
        self.benchmark_returns = self.benchmarks.pct_change()

    def apply_universe_mask(self, base_weights):
        if not os.path.exists("historical_universe.csv"):
            return base_weights
        univ_df = pd.read_csv("historical_universe.csv", parse_dates=['Date'])
        univ_df['Value'] = True
        # Using pivot_table with aggfunc='first' prevents duplicate issues
        mask = univ_df.pivot_table(index='Date', columns='Ticker',
                                   values='Value', aggfunc='first').fillna(False)
        mask = mask.reindex(index=base_weights.index,
                            columns=base_weights.columns)
        mask = mask.ffill().fillna(False)
        mask.columns       = mask.columns.astype(str)
        base_weights.columns = base_weights.columns.astype(str)
        return base_weights.multiply(mask, axis=0)

    def run_backtest(self, weights_df):
        weights_df = self.apply_universe_mask(weights_df)
        holdings = weights_df.shift(1).fillna(0)

        weights_array = holdings.values
        returns_array = self.returns_df.reindex(
            columns=holdings.columns).fillna(0).values

        port_values    = np.zeros(len(returns_array))
        port_values[0] = 1.0
        current_weights = np.zeros(weights_array.shape[1])

        for i in range(1, len(returns_array)):
            growth   = 1 + returns_array[i]
            prev_val = port_values[i - 1]

            holdings_value        = prev_val * current_weights * growth
            current_val_pre_rebal = (np.sum(holdings_value) +
                                     prev_val * (1 - np.sum(current_weights)))

            target_weights = weights_array[i]

            actual_weights = (holdings_value / current_val_pre_rebal
                              if current_val_pre_rebal > 0
                              else np.zeros_like(current_weights))

            weight_diffs = target_weights - actual_weights
            sells  = np.where(weight_diffs < 0, np.abs(weight_diffs), 0)
            buys   = np.where(weight_diffs > 0, weight_diffs, 0)
            cost   = np.sum(sells) * self.sell_cost + np.sum(buys) * self.buy_cost

            port_values[i] = current_val_pre_rebal * (1 - cost)
            current_weights = target_weights

        port_series = pd.Series(port_values, index=self.prices_df.index)
        return self.calculate_metrics(port_series)

    def calculate_metrics(self, port_series):
        returns = port_series.pct_change().dropna()
        days    = (port_series.index[-1] - port_series.index[0]).days
        if days == 0:
            return {'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'port_series': port_series}

        years = days / 365.25
        cagr  = (port_series.iloc[-1] / port_series.iloc[0]) ** (1 / years) - 1
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)

        roll_max  = port_series.cummax()
        drawdown  = port_series / roll_max - 1.0
        mdd       = drawdown.min()

        def calc_ab(col):
            bm_ret  = self.benchmark_returns[col].loc[returns.index].fillna(0)
            cov_mat = np.cov(returns, bm_ret)
            beta    = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] != 0 else 0
            bm_cagr = (self.benchmarks[col].iloc[-1] /
                       self.benchmarks[col].iloc[0]) ** (1 / years) - 1
            alpha   = cagr - (0.025 + beta * (bm_cagr - 0.025))
            return alpha, beta

        ak, bk = calc_ab('KOSPI')
        aq, bq = calc_ab('KOSDAQ')
        as_, bs = calc_ab('SP500')

        yearly      = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        consistency = (yearly > 0).mean() if len(yearly) > 0 else 0

        return {
            'CAGR': cagr, 'MDD': mdd, 'Sharpe': sharpe,
            'Alpha (KOSPI)': ak,  'Beta (KOSPI)': bk,
            'Alpha (KOSDAQ)': aq, 'Beta (KOSDAQ)': bq,
            'Alpha (SP500)': as_, 'Beta (SP500)': bs,
            'Consistency': consistency,
            'Yearly Returns': yearly,
            'port_series': port_series
        }

# ==============================================================================
# 3. FACTOR LIBRARY
# ==============================================================================
def zscore_cs(df):
    mu  = df.mean(axis=1)
    sig = df.std(axis=1) + 1e-8
    return df.sub(mu, axis=0).div(sig, axis=0)

def rank_cs(df):
    return df.rank(axis=1, pct=True)

def build_factors(bt):
    prices  = bt.prices_df
    returns = prices.pct_change().fillna(0)
    volumes = pd.DataFrame(bt.volume).ffill()

    # 12-1 month momentum
    mom_12_1 = prices.pct_change(252) - prices.pct_change(20)

    # 3-month momentum
    mom_3m = prices.pct_change(60)

    # Inverse realised volatility
    inv_vol = 1.0 / (returns.rolling(21).std() + 1e-8)

    # Foreign net buy
    foreign_flow = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for ticker, inst_df in bt.institutions.items():
        if ticker in foreign_flow.columns and '외국인합계' in inst_df.columns:
            foreign_flow[ticker] = inst_df['외국인합계']
    foreign_flow = foreign_flow.ffill()
    foreign_buy_ratio = (foreign_flow.rolling(20).sum() /
                         (volumes.reindex(columns=foreign_flow.columns)
                          .rolling(20).sum().abs() + 1e-8))

    # Value
    pbr_panel = pd.DataFrame(index=prices.index, columns=prices.columns,
                              dtype=float)
    for ticker, fund_df in bt.fundamentals.items():
        if ticker in pbr_panel.columns and 'PBR' in fund_df.columns:
            pbr = fund_df['PBR'].replace(0, np.nan)
            pbr[pbr < 0] = np.nan
            pbr_panel[ticker] = pbr
    pbr_panel = pbr_panel.ffill()
    inv_pbr   = 1.0 / (pbr_panel + 1e-8)

    factors = {
        'mom_12_1':      (mom_12_1,      0.30),
        'mom_3m':        (mom_3m,        0.20),
        'inv_vol':       (inv_vol,       0.20),
        'foreign_buy':   (foreign_buy_ratio, 0.20),
        'value':         (inv_pbr,       0.10),
    }
    return factors

# ==============================================================================
# 4. VOLATILITY TARGETING
# ==============================================================================
def volatility_target(returns_df, port_weights, target_vol=0.15, window=21):
    port_ret = (port_weights.shift(1) * returns_df.reindex(
        columns=port_weights.columns).fillna(0)).sum(axis=1)
    realised_vol = port_ret.rolling(window).std() * np.sqrt(252)
    realised_vol = realised_vol.replace(0, np.nan).ffill().fillna(target_vol)

    scalar = (target_vol / realised_vol).clip(upper=1.0)
    scaled = port_weights.mul(scalar, axis=0)
    return scaled, scalar

# ==============================================================================
# 5. DYNAMIC STOP-LOSS
# ==============================================================================
def apply_trailing_stop(prices_df, weights_df, returns_df, benchmarks,
                        sl_threshold=0.08, cooldown_period=20,
                        reentry_momentum_days=5, reentry_momentum_threshold=0.02):

    w_array = weights_df.values.copy().astype(float)
    r_array = returns_df.reindex(columns=weights_df.columns).fillna(0).values
    kospi   = benchmarks['KOSPI'].values

    current_val = 1.0
    high_water  = 1.0
    cooldown    = 0
    in_recovery = False

    for i in range(1, len(r_array)):
        # Because w_array holds target weights determined at the close of day (T),
        # these weights are executed at tomorrow's open.
        # Therefore, the return experienced by holding the portfolio from day (T-1) to (T)
        # depends on the weights generated at (T-1) applied to the returns of day (T).

        prev_weights = w_array[i - 1]
        growth_today = 1 + r_array[i]
        port_growth = (np.sum(prev_weights * growth_today) +
                       (1 - np.sum(prev_weights)))
        current_val *= port_growth

        if current_val > high_water:
            high_water = current_val

        drawdown_pct = (high_water - current_val) / (high_water + 1e-12)

        if cooldown > 0:
            w_array[i] = 0.0
            cooldown  -= 1
            if cooldown == 0:
                in_recovery = True
            continue

        if in_recovery:
            start_idx = max(0, i - reentry_momentum_days)
            if i > 0 and kospi[start_idx] > 0:
                kospi_ret = kospi[i - 1] / kospi[start_idx] - 1
            else:
                kospi_ret = 0.0

            if kospi_ret < reentry_momentum_threshold:
                w_array[i] = 0.0
                continue
            else:
                in_recovery = False

        if drawdown_pct > sl_threshold:
            w_array[i]  = 0.0
            current_val *= (1 - 0.006)
            high_water   = current_val
            cooldown     = cooldown_period
            in_recovery  = False

    return pd.DataFrame(w_array, index=weights_df.index,
                        columns=weights_df.columns)

# ==============================================================================
# 6. STRATEGY RUNNER
# ==============================================================================
def run_strategy():
    bt = VectorizedBacktester()

    prices  = bt.prices_df
    returns = prices.pct_change().fillna(0)

    split_date = prices.index[int(len(prices) * 0.60)]
    print(f"Walk-forward split: train up to {split_date.date()}, test from {split_date.date()}")

    print("Building factors...")
    factors = build_factors(bt)

    def zscore_cs(df):
        mu  = df.mean(axis=1)
        sig = df.std(axis=1) + 1e-8
        return df.sub(mu, axis=0).div(sig, axis=0)

    print("Calculating composite signals...")
    w_mom = 3.0
    w_vol = 0.1604
    w_foreign = 2.0
    w_value = 0.3025
    N_STOCKS = 5

    combined_score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    combined_score += w_mom * zscore_cs(factors['mom_12_1'][0].fillna(0))
    combined_score += w_vol * zscore_cs(factors['inv_vol'][0].fillna(0))
    combined_score += w_foreign * zscore_cs(factors['foreign_buy'][0].fillna(0))
    combined_score += w_value * zscore_cs(factors['value'][0].fillna(0))

    ranks     = combined_score.rank(axis=1, ascending=False)
    selected  = (ranks <= N_STOCKS).astype(float)

    print("Applying inverse volatility sizing...")
    inv_vol_panel = (1.0 / (returns.rolling(21).std() + 1e-8))
    inv_vol_selected = inv_vol_panel.where(selected > 0, 0.0)
    row_sum = inv_vol_selected.sum(axis=1).replace(0, np.nan)
    base_weights = inv_vol_selected.div(row_sum, axis=0).fillna(0)

    monthly_dates = base_weights.resample('ME').last().index
    monthly_weights = base_weights.reindex(monthly_dates)
    base_weights = monthly_weights.reindex(base_weights.index).ffill().fillna(0)

    print("Applying KOSDAQ 100-day Soft MA filter...")
    kosdaq_ma = bt.benchmarks['KOSDAQ'].rolling(100).mean()
    market_safe = (bt.benchmarks['KOSDAQ'] > kosdaq_ma).astype(float).rolling(5).min().fillna(0)
    safe_weights = base_weights.multiply(market_safe.replace(0, 0.7), axis=0)

    print("Applying volatility targeting...")
    vol_weights, vol_scalar = volatility_target(
        returns, safe_weights, target_vol=0.70, window=21)

    # Strictly unleveraged.
    vol_weights = vol_weights.clip(lower=0.0, upper=1.0)

    print("Applying trailing stop-loss...")
    final_weights = apply_trailing_stop(
        prices_df      = prices,
        weights_df     = vol_weights,
        returns_df     = returns,
        benchmarks     = bt.benchmarks,
        sl_threshold   = 0.16,
        cooldown_period = 20,
        reentry_momentum_days      = 3,
        reentry_momentum_threshold = 0.01
    )

    print("Running backtest...")
    metrics = bt.run_backtest(final_weights)
    test_series = metrics['port_series'].loc[split_date:]

    days  = (test_series.index[-1] - test_series.index[0]).days
    years = days / 365.25

    test_ret = test_series.pct_change().fillna(0)
    cagr     = (test_series.iloc[-1] / test_series.iloc[0]) ** (1 / years) - 1
    sharpe   = (test_ret.mean() / (test_ret.std() + 1e-9)) * np.sqrt(252)
    roll_max = test_series.cummax()
    mdd      = (test_series / roll_max - 1.0).min()

    def oos_alpha_beta(bm_col):
        bm = bt.benchmarks[bm_col].loc[split_date:]
        bm_ret = bt.benchmark_returns[bm_col].loc[test_series.index].fillna(0)
        cov_mat = np.cov(test_ret, bm_ret)
        beta  = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] != 0 else 0
        bm_cagr = (bm.iloc[-1] / bm.iloc[0]) ** (1 / years) - 1
        alpha = cagr - (0.025 + beta * (bm_cagr - 0.025))
        return alpha, beta

    ak, bk = oos_alpha_beta('KOSPI')
    aq, bq = oos_alpha_beta('KOSDAQ')
    as_, bs = oos_alpha_beta('SP500')

    yearly = test_ret.resample('YE').apply(lambda x: (1 + x).prod() - 1)

    print("============================================================")
    print("OUT-OF-SAMPLE RESULTS (Walk-Forward Validated)")
    print("============================================================")
    print(f"Test Period:        {test_series.index[0].date()} → {test_series.index[-1].date()}")

    print(f"CAGR:               {cagr:.2%}")
    print(f"MDD:                {mdd:.2%}")
    print(f"Sharpe Ratio:       {sharpe:.2f}")
    print(f"Alpha (KOSPI):      {ak:.2%}  |  Beta: {bk:.3f}")
    print(f"Alpha (KOSDAQ):     {aq:.2%}  |  Beta: {bq:.3f}")
    print(f"Alpha (SP500):      {as_:.2%}  |  Beta: {bs:.3f}")
    print("Yearly Returns:")
    for dt, r in yearly.items():
        print(f"  {dt.year}: {r:+.2%}")
    print("============================================================")

    final_weights.to_csv("best_weights.csv")
    metrics['port_series'].to_csv("portfolio_values.csv")

    trades = []
    holdings = final_weights.shift(1).fillna(0)
    for i in range(1, len(final_weights)):
        prev_w = holdings.iloc[i]
        curr_w = final_weights.iloc[i]
        diff   = curr_w - prev_w
        date   = final_weights.index[i]
        for ticker in diff.index:
            if abs(diff[ticker]) > 0.001:
                trades.append({
                    'Date':          date,
                    'Ticker':        ticker,
                    'Weight_Change': diff[ticker],
                    'Action':        'BUY' if diff[ticker] > 0 else 'SELL'
                })

    pd.DataFrame(trades).to_csv("trades.csv", index=False)
    vol_scalar.to_csv("vol_scalar.csv")
    print("Outputs saved: best_weights.csv, portfolio_values.csv, trades.csv, vol_scalar.csv")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantitative Strategy Backtester v2")
    parser.add_argument("--download", action="store_true", help="Download missing historical data")
    parser.add_argument("--optimize", action="store_true", help="Run strategy and backtest")
    args = parser.parse_args()

    if args.download:
        download_data()
    elif args.optimize:
        run_strategy()
    else:
        print("Please specify: --download or --optimize")
