import warnings
warnings.filterwarnings("ignore")

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from pykrx import stock
from pykrx.website.comm import webio
import FinanceDataReader as fdr
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

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
    try:
        P = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
        J = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
        U = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
        A = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        _krx_session.get(P, headers={"User-Agent": A}, timeout=15, verify=False)
        _krx_session.get(J, headers={"User-Agent": A, "Referer": P}, timeout=15, verify=False)
        r = _krx_session.post(U, data={"mbrNm": "", "telNo": "", "di": "", "certType": "", "mbrId": login_id, "pw": login_pw}, headers={"User-Agent": A, "Referer": P}, timeout=15, verify=False)
        return r.json().get("_error_code", "") == "CD001"
    except Exception as e:
        print(f"Login failed: {e}")
        return False

start_date = (datetime.today() - timedelta(days=365*10)).strftime("%Y%m%d")
end_date = datetime.today().strftime("%Y%m%d")

def get_universe(date):
    kospi_caps = stock.get_market_cap(date, market="KOSPI")
    kosdaq_caps = stock.get_market_cap(date, market="KOSDAQ")

    kospi_top = kospi_caps.sort_values("시가총액", ascending=False).head(150).index.tolist()
    kosdaq_top = kosdaq_caps.sort_values("시가총액", ascending=False).head(30).index.tolist()

    return kospi_top + kosdaq_top

def download_ticker_data(ticker):
    file_path = f"data/{ticker}.csv"
    if os.path.exists(file_path):
        return ticker, True
    try:
        df_ohlcv = stock.get_market_ohlcv(start_date, end_date, ticker)
        df_fund = stock.get_market_fundamental(start_date, end_date, ticker)
        df_vol = stock.get_market_trading_volume_by_date(start_date, end_date, ticker)

        if df_ohlcv is not None and not df_ohlcv.empty:
            df = df_ohlcv.join(df_fund, how='left').join(df_vol, how='left')
            df.to_csv(file_path)
            return ticker, True
        return ticker, False
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return ticker, False

if __name__ == "__main__":
    print(f"KRX Login successful: {login_krx(KRX_LOGIN_ID, KRX_LOGIN_PW)}")

    print(f"Data fetching from {start_date} to {end_date}")
    universe = get_universe(end_date)
    os.makedirs("data", exist_ok=True)

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
    print("Benchmarks saved to benchmarks.csv")

    print(f"Downloading {len(universe)} stocks with ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(download_ticker_data, universe), total=len(universe)))
    print("Data download complete.")
