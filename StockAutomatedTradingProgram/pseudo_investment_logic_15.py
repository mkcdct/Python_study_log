'''
logic_14와의 차이

이전까지는 일봉을 기준으로 검증했다면,
이번에는 분봉을 기준으로 최근 1년간의 주가 데이터로 백테스팅.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# [설정] 분봉 백테스팅 (최근 1년 간의 1시간봉 데이터 활용)
PERIOD = "1y"
INTERVAL = "1h"
COMMISSION_PCT = 0.0005
SLIPPAGE_PCT = 0.0005

def get_sector_universe():
    return {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "ADBE": "Technology",
        "AMZN": "Consumer Cyclical", "TSLA": "Consumer Cyclical", "NKE": "Consumer Cyclical", "HD": "Consumer Cyclical",
        "GOOGL": "Communication Services", "META": "Communication Services", "DIS": "Communication Services",
        "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare",
        "AXP": "Financials", "V": "Financials", "BA": "Industrials", "CAT": "Industrials",
        "XOM": "Energy", "CVX": "Energy", "NEE": "Utilities", "WMT": "Consumer Defensive"
    }

def fetch_intraday_data(ticker, period=PERIOD, interval=INTERVAL):
    try:
        # yfinance를 통한 분봉 데이터 로드 (최대 1년 치 1시간봉)
        df = yf.download(ticker, period=period, interval=interval, progress=False, multi_level_index=False)
        if df.empty or len(df) < 200: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

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

def add_intraday_indicators(df):
    df = df.copy()
    # 분봉(1시간봉) 기준 장기 추세선 (약 30거래일 분량의 시간봉 = 30 * 6.5 = 195 ≒ 200시간봉)
    df['Trend_MA'] = df['Close'].rolling(window=200).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    bb_std_val = df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2.0 * bb_std_val
    df['RSI'] = calc_rsi(df['Close'], period=14)
    df['ATR'] = calc_atr(df, period=14)
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def run_intraday_strategy(df, market_trend_series):
    df = df.copy()
    position = 0
    entry_price = 0.0
    highest_price = 0.0
    holding_bars = 0
    strategy_returns = []

    for i in range(len(df)):
        current_time = df.index[i]
        # 날짜만 추출하여 시장 국면 필터 매칭
        current_date_str = pd.to_datetime(current_time).strftime('%Y-%m-%d')
        
        close = df['Close'].iloc[i]
        trend_ma = df['Trend_MA'].iloc[i]
        rsi = df['RSI'].iloc[i]
        lower = df['BB_Lower'].iloc[i]
        atr = df['ATR'].iloc[i]
        bar_ret = df['Daily_Return'].iloc[i]

        is_market_bullish = True
        if current_date_str in market_trend_series.index:
            is_market_bullish = market_trend_series.loc[current_date_str]

        if position == 0:
            if not is_market_bullish:
                strategy_returns.append(0.0)
                continue

            is_downtrend = pd.notna(trend_ma) and close < trend_ma
            if is_downtrend:
                is_condition = (pd.notna(rsi) and rsi <= 35) or (pd.notna(lower) and close <= lower)
            else:
                is_condition = pd.notna(rsi) and rsi <= 45

            if is_condition:
                position = 1
                entry_price = close * (1 + SLIPPAGE_PCT)
                highest_price = entry_price
                holding_bars = 0
                strategy_returns.append(-COMMISSION_PCT)
            else:
                strategy_returns.append(0.0)
        else:
            holding_bars += 1
            if close > highest_price:
                highest_price = close

            current_atr_pct = (atr / close) if (pd.notna(atr) and close > 0) else 0.01
            # 분봉은 변동 폭이 작으므로 손절/트레일링 폭을 더 민감하게 설정
            dynamic_sl_pct = max(0.015, current_atr_pct * 1.5)
            dynamic_trailing_pct = max(0.025, current_atr_pct * 2.0)

            hit_sl = close <= entry_price * (1 - dynamic_sl_pct)
            hit_trailing = close <= highest_price * (1 - dynamic_trailing_pct)
            # 시간 제한: 1시간봉 기준 40개 바(약 1주일 거래 시간) 경과 시 청산
            hit_time_limit = holding_bars >= 40

            if hit_sl or hit_trailing or hit_time_limit:
                exit_price = close * (1 - SLIPPAGE_PCT)
                trade_ret = (exit_price / entry_price) - 1 - COMMISSION_PCT
                strategy_returns.append(trade_ret)
                position = 0
                entry_price = 0.0
                highest_price = 0.0
                holding_bars = 0
            else:
                strategy_returns.append(bar_ret * position)

    df['Strategy_Return'] = strategy_returns
    return df

if __name__ == "__main__":
    print("[*] 최근 1년 분봉 시장 국면 필터(SPY 1시간봉) 구축 중...")
    spy_df = fetch_intraday_data("SPY", period=PERIOD, interval=INTERVAL)
    if spy_df is not None:
        spy_df['Market_MA'] = spy_df['Close'].rolling(window=200).mean()
        spy_df['Date_Str'] = spy_df.index.strftime('%Y-%m-%d')
        # 일별 기준으로 시장 트렌드 매핑
        market_trend = (spy_df['Close'] > spy_df['Market_MA']).groupby(spy_df['Date_Str']).last()
    else:
        market_trend = pd.Series(True)

    universe_dict = get_sector_universe()
    qualified_stocks = []
    
    print("[*] 종목별 1시간봉 데이터 스크리닝 및 모멘텀 정렬 중...")
    for ticker, sector in universe_dict.items():
        df = fetch_intraday_data(ticker, period=PERIOD, interval=INTERVAL)
        if df is None: continue
        df = add_intraday_indicators(df)
        if len(df) < 100: continue
        
        # 최근 40시간봉(약 1주일) 기준 단기 모멘텀 계산
        recent_momentum = df['Close'].pct_change(40).iloc[-1]
        momentum_val = recent_momentum if pd.notna(recent_momentum) else 0
        
        qualified_stocks.append({
            'ticker': ticker, 'sector': sector, 'df': df, 'momentum': momentum_val
        })

    qualified_stocks = sorted(qualified_stocks, key=lambda x: x['momentum'], reverse=True)

    portfolio_stocks = []
    selected_sectors = set()
    for item in qualified_stocks:
        if item['sector'] not in selected_sectors:
            portfolio_stocks.append(item)
            selected_sectors.add(item['sector'])
        if len(portfolio_stocks) >= 3: break

    print(f"\n[분봉 포트폴리오 구성 완료]")
    for p in portfolio_stocks:
        print(f" - 종목: {p['ticker']} (섹터: {p['sector']}, 1시간모멘텀: {p['momentum']*100:.2f}%)")

    portfolio_results, bnh_results = [], []
    for p in portfolio_stocks:
        opt_df = run_intraday_strategy(p['df'], market_trend)
        portfolio_results.append(opt_df['Strategy_Return'])
        bnh_results.append(p['df']['Daily_Return'])

    port_df = pd.concat(portfolio_results, axis=1).fillna(0)
    port_df['Port_Return'] = port_df.mean(axis=1)
    cum_portfolio = (1 + port_df['Port_Return']).cumprod()
    final_port_return = cum_portfolio.iloc[-1] - 1
    port_mdd = ((cum_portfolio - cum_portfolio.cummax()) / cum_portfolio.cummax()).min()

    bnh_df = pd.concat(bnh_results, axis=1).fillna(0)
    bnh_df['BnH_Return'] = bnh_df.mean(axis=1)
    cum_bnh = (1 + bnh_df['BnH_Return']).cumprod()
    final_bnh_return = cum_bnh.iloc[-1] - 1
    bnh_mdd = ((cum_bnh - cum_bnh.cummax()) / cum_bnh.cummax()).min()

    print(f"\n{'='*55}")
    print(f" 최근 1년 분봉(1시간봉) 하이브리드 전략 최종 성과")
    print(f"{'='*55}")
    print(f" [분봉 동적 하이브리드 전략]")
    print(f"  - 최종 수익률 : {final_port_return * 100:.2f}%")
    print(f"  - 최대낙폭(MDD): {port_mdd * 100:.2f}%")
    print(f"{'-'*55}")
    print(f" [단순보유 (Buy & Hold)]")
    print(f"  - 최종 수익률 : {final_bnh_return * 100:.2f}%")
    print(f"  - 최대낙폭(MDD): {bnh_mdd * 100:.2f}%")
    print(f"{'='*55}")

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(cum_portfolio.index, cum_portfolio, label='Intraday Hybrid Strategy (1h)', color='darkorange', linewidth=2)
    plt.plot(cum_bnh.index, cum_bnh, label='Buy & Hold', color='gray', linestyle='--', linewidth=1.5)
    plt.title('Intraday (1h) Hybrid Strategy vs. Buy & Hold (Last 1 Year)')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date/Time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()