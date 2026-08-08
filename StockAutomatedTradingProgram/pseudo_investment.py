import time
import logging
import requests
import json
import pandas as pd
import numpy as np
import yfinance as yf

# 로깅 설정 (실전 모의투자용 파일 및 콘솔 로그 출력)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# [설정값]
COMMISSION_PCT = 0.0005
BASE_SLIPPAGE_PCT = 0.0005
MAX_SLOTS = 3  

# 한국투자증권 API 연동 정보
CANO = "50201047"          # 종합계좌번호 앞 8자리
ACNT_PRDT_CD = "01"        # 계좌상품코드 2자리
APP_KEY = "PSdNxitKOUCi47IMoJaC1HJBqaQTtHSnefEM"
APP_SECRET = "VQ7Qi75RS8p99iNAsA3chtWO7h8LlEEPxXGss1wSf5VwuV10976zIhhpxfHqBdZP9G/BxXuZYEWck/X5y3y3l/oZmYtUQBOAa9TYW9GLZ7q6YdmTufNCC1yaUTCTHDsKfQ07vhZ1bo7n/HkLGHbaT/F19IL7D2cQlGtAHUwmxtjlJVcZq4Q="
IS_MOCK = True             # 모의투자 여부

# 모의투자 / 실전투자 도메인 설정
if IS_MOCK:
    BASE_URL = "https://openapivts.koreainvestment.com:29443"  # 모의투자 서버
else:
    BASE_URL = "https://openapi.koreainvestment.com:9443"      # 실전투자 서버


def get_access_token():
    """한국투자증권 API 접근 토큰 발급 함수"""
    path = "oauth2/tokenP"
    url = f"{BASE_URL}/{path}"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }
    
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        res_data = res.json()
        access_token = res_data["access_token"]
        logging.info("[API 성공] 한국투자증권 Access Token 발급 완료")
        return access_token
    except Exception as e:
        logging.error(f"[API 오류] Access Token 발급 실패: {e}")
        return None


def get_account_balance(access_token):
    """실전 모의 계좌 잔고 조회 함수 (미국 주식 기준)"""
    path = "uapi/overseas-stock/v1/trading/inquire-balance"
    url = f"{BASE_URL}/{path}"
    
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "VTTS3012R"  # 모의투자 미국 주식 잔고 조회 TR_ID (실전은 TTS3012R)
    }
    
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "OVRS_EXCG_CD": "NASD",  # 나스닥 기준 (통합조회 등 필요에 따라 조정)
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": "",
        "TR_CONT": ""
    }
    
    try:
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        if data["rt_cd"] == "0":
            logging.info("[API 성공] 계좌 잔고 조회 완료")
            return data.get("output1", [])  # 보유 종목 리스트 반환
        else:
            logging.error(f"[API 오류] 잔고 조회 실패: {data.get('msg1')}")
            return []
    except Exception as e:
        logging.error(f"[API 예외] 잔고 조회 중 에러 발생: {e}")
        return []


def get_modern_robust_universe():
    return {
        "AAPL": "Technology",          
        "MSFT": "Technology",          
        "NVDA": "Technology",          
        "GOOGL": "Communication",        
        "META": "Communication",         
        "AMZN": "Consumer Discretionary",
        "TSLA": "Consumer Discretionary",
        "UNH": "Healthcare",           
        "LLY": "Healthcare",           
        "JPM": "Financials",           
        "PG": "Consumer Staples"         
    }


def safe_fetch_data_with_retry(ticker, start_days_ago=300, max_retries=3):
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=start_days_ago)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False, multi_level_index=False)
            if df is not None and not df.empty and len(df) >= 100:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                return df
            raise ValueError("Empty or insufficient data received.")
        except Exception as e:
            logging.warning(f"[API 경고] {ticker} 다운로드 시도 {attempt}/{max_retries} 실패: {e}")
            if attempt == max_retries:
                logging.error(f"[API 오류] {ticker} 최대 재시도 초과. 데이터 수신 포기.")
                return None
            time.sleep(delay)
            delay *= 2


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def add_indicators(df):
    df = df.copy()
    df['Trend_MA'] = df['Close'].rolling(window=200).mean()
    df['Short_MA'] = df['Close'].rolling(window=20).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    bb_std_val = df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2.0 * bb_std_val
    df['RSI'] = calc_rsi(df['Close'], period=14)
    df['ATR'] = calc_atr(df, period=14)
    
    df['Signal_Trend_MA'] = df['Trend_MA'].shift(1)
    df['Signal_Short_MA'] = df['Short_MA'].shift(1)
    df['Signal_RSI'] = df['RSI'].shift(1)
    df['Signal_Lower'] = df['BB_Lower'].shift(1)
    df['Signal_ATR'] = df['ATR'].shift(1)
    df['Signal_Close'] = df['Close'].shift(1)
    return df


def evaluate_today_signals(all_dfs, market_trend, market_mom):
    signals = {}
    is_market_bullish = list(market_trend.values())[-1] if market_trend else True
    is_v_rebound = list(market_mom.values())[-1] if market_mom else False

    for ticker, info in all_dfs.items():
        df = info['df']
        if len(df) < 2:
            continue
            
        idx_pos = len(df) - 2 
        sig_close = df['Signal_Close'].iloc[idx_pos]
        short_ma = df['Signal_Short_MA'].iloc[idx_pos]
        rsi = df['Signal_RSI'].iloc[idx_pos]
        lower = df['Signal_Lower'].iloc[idx_pos]
        atr = df['Signal_ATR'].iloc[idx_pos]

        current_atr_pct = (atr / sig_close) if (pd.notna(atr) and sig_close > 0) else 0.02
        is_valid_volatility = (current_atr_pct >= 0.012)

        if is_market_bullish or is_v_rebound:
            is_buy_condition = ((pd.notna(short_ma) and sig_close > short_ma) or (pd.notna(rsi) and rsi <= 55)) and is_valid_volatility
        else:
            is_buy_condition = ((pd.notna(rsi) and rsi <= 35) or (pd.notna(lower) and sig_close <= lower)) and is_valid_volatility

        signals[ticker] = {
            'sector': info['sector'],
            'buy_signal': is_buy_condition,
            'momentum': df['Close'].iloc[idx_pos] / df['Close'].iloc[idx_pos - 120] - 1 if idx_pos >= 120 else 0
        }
    return signals


def job_daily_trading_execution():
    logging.info("[*] 한국투자증권 모의투자 데일리 리밸런싱 작업 시작")
    
    try:
        # 0. API 접근 토큰 발급
        access_token = get_access_token()
        if not access_token:
            logging.error("[치명적 오류] Access Token 발급 실패로 작업을 중단합니다.")
            return

        # 1. 계좌 잔고 조회 (현재 보유 종목 파악)
        current_holdings = get_account_balance(access_token)
        logging.info(f"[*] 현재 계좌 보유 종목 수: {len(current_holdings)}개")

        # 2. 시장 지표(SPY) 안전 수신
        spy_df = safe_fetch_data_with_retry("SPY", start_days_ago=300)
        if spy_df is None or spy_df.empty:
            logging.error("[치명적 오류] SPY 데이터 수신 실패. 금일 리밸런싱을 중단합니다.")
            return

        spy_df['Market_MA'] = spy_df['Close'].rolling(window=200).mean()
        market_trend = (spy_df['Close'] > spy_df['Market_MA']).shift(1).to_dict()
        market_mom = (spy_df['Close'].pct_change(5) > 0.03).shift(1).to_dict()

        # 3. 유니버스 데이터 수신 및 지표 일괄 계산
        universe_dict = get_modern_robust_universe()
        all_dfs = {}
        for ticker, sector in universe_dict.items():
            df = safe_fetch_data_with_retry(ticker, start_days_ago=300)
            if df is None:
                continue
            df = add_indicators(df)
            all_dfs[ticker] = {'sector': sector, 'df': df}

        if not all_dfs:
            logging.error("[치명적 오류] 유효한 유니버스 데이터가 없습니다. 작업 중단.")
            return

        # 4. 오늘 자 신호 평가
        signals = evaluate_today_signals(all_dfs, market_trend, market_mom)

        logging.info("--- [종목별 시그널 상세 분석 결과] ---")
        for ticker, data in signals.items():
            logging.info(f"종목: {ticker:<6} | 섹터: {data['sector']:<25} | 매수시그널: {str(data['buy_signal']):<5} | 모멘텀: {data['momentum']*100:.2f}%")
        logging.info("----------------------------------------")

        # TODO: 매수/매도 주문 실행 로직 (잔고와 시그널을 바탕으로 실제 주문 API 연동 확장 구간)
        
        logging.info("[*] 한국투자증권 모의투자 데일리 리밸런싱 작업 정상 완료")

    except Exception as e:
        logging.error(f"[치명적 예외 발생] 데일리 루프 실행 중 에러가 발생했습니다: {e}", exc_info=True)


if __name__ == "__main__":
    job_daily_trading_execution()