'''
logic_12와의 차이

1. 섹터별 중복 없는 최대 3개 종목 자동 스크리닝
2. 변동성(ATR)에 따라 유동적으로 변하는 동적 트레일링 스탑 로직을 결합함.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

START_DATE = "2020-01-01"
END_DATE = "2026-06-30"
COMMISSION_PCT = 0.0005
SLIPPAGE_PCT = 0.0005

# 1. 섹터 유니버스 풀 설정
def get_sector_universe():
    return {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "ADBE": "Technology",
        "AMZN": "Consumer Cyclical", "TSLA": "Consumer Cyclical", "NKE": "Consumer Cyclical", "HD": "Consumer Cyclical",
        "GOOGL": "Communication Services", "META": "Communication Services", "DIS": "Communication Services",
        "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare",
        "AXP": "Financials", "V": "Financials", "BA": "Industrials", "CAT": "Industrials",
        "XOM": "Energy", "CVX": "Energy", "NEE": "Utilities", "WMT": "Consumer Defensive"
    }

def fetch_data(ticker, start, end, buffer_days=250):
    buffered_start = (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=buffered_start, end=end, interval="1d", progress=False, multi_level_index=False)
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

def add_indicators(df):
    df = df.copy()
    df['Trend_MA'] = df['Close'].rolling(window=200).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    bb_std_val = df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2.0 * bb_std_val
    df['RSI'] = calc_rsi(df['Close'], period=14)
    df['ATR'] = calc_atr(df, period=14)
    df['Daily_Return'] = df['Close'].pct_change()
    return df

# 유연한(동적) 파라미터가 적용된 하이브리드 전략 함수
def run_dynamic_hybrid_strategy(df):
    df = df.copy()
    position = 0
    entry_price = 0.0
    highest_price = 0.0
    holding_days = 0
    strategy_returns = []

    for i in range(len(df)):
        close = df['Close'].iloc[i]
        trend_ma = df['Trend_MA'].iloc[i]
        rsi = df['RSI'].iloc[i]
        lower = df['BB_Lower'].iloc[i]
        atr = df['ATR'].iloc[i]
        daily_ret = df['Daily_Return'].iloc[i]

        if position == 0:
            is_downtrend = pd.notna(trend_ma) and close < trend_ma
            
            if is_downtrend:
                is_condition = (pd.notna(rsi) and rsi <= 35) or (pd.notna(lower) and close <= lower)
            else:
                is_condition = pd.notna(rsi) and rsi <= 45

            if is_condition:
                position = 1
                entry_price = close * (1 + SLIPPAGE_PCT)
                highest_price = entry_price
                holding_days = 0
                strategy_returns.append(-COMMISSION_PCT)
            else:
                strategy_returns.append(0.0)
        else:
            holding_days += 1
            if close > highest_price:
                highest_price = close

            current_atr_pct = (atr / close) if (pd.notna(atr) and close > 0) else 0.02
            dynamic_sl_pct = max(0.02, current_atr_pct * 1.5)
            dynamic_trailing_pct = max(0.03, current_atr_pct * 2.0)

            hit_sl = close <= entry_price * (1 - dynamic_sl_pct)
            hit_trailing = close <= highest_price * (1 - dynamic_trailing_pct)
            hit_time_limit = holding_days >= 20

            if hit_sl or hit_trailing or hit_time_limit:
                exit_price = close * (1 - SLIPPAGE_PCT)
                trade_ret = (exit_price / entry_price) - 1 - COMMISSION_PCT
                strategy_returns.append(trade_ret)
                position = 0
                entry_price = 0.0
                highest_price = 0.0
                holding_days = 0
            else:
                strategy_returns.append(daily_ret * position)

    df['Strategy_Return'] = strategy_returns
    return df

# 메인 실행부 (모멘텀 정렬 랭킹 + 섹터 분산 스크리닝)
if __name__ == "__main__":
    universe_dict = get_sector_universe()
    qualified_stocks = []
    
    print(f"[*] 모멘텀 정렬 및 섹터 분산 스크리닝 중 ({START_DATE} ~ {END_DATE})...")
    for ticker, sector in universe_dict.items():
        df = fetch_data(ticker, START_DATE, END_DATE)
        if df is None: continue
        df = add_indicators(df)
        df_target = df[(df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))].copy()
        if len(df_target) < 100: continue
        
        total_return = (df_target['Close'].iloc[-1] / df_target['Close'].iloc[0]) - 1
        if -0.70 <= total_return < 1.50:
            # 최근 20거래일(약 1개월) 수익률 모멘텀 계산
            recent_momentum = df_target['Close'].pct_change(20).iloc[-1]
            momentum_val = recent_momentum if pd.notna(recent_momentum) else 0
            
            qualified_stocks.append({
                'ticker': ticker, 
                'sector': sector, 
                'df': df_target, 
                'momentum': momentum_val
            })

    # [핵심 수정] 최근 모멘텀이 가장 높은 순서대로 후보군 정렬 (내림차순)
    qualified_stocks = sorted(qualified_stocks, key=lambda x: x['momentum'], reverse=True)

    # 섹터 중복 방지하며 상위 종목부터 최대 3개 선정
    portfolio_stocks = []
    selected_sectors = set()
    for item in qualified_stocks:
        if item['sector'] not in selected_sectors:
            portfolio_stocks.append(item)
            selected_sectors.add(item['sector'])
        if len(portfolio_stocks) >= 3: break

    print(f"\n[포트폴리오 구성 완료 (모멘텀 상위 + 섹터 분산)]")
    for p in portfolio_stocks:
        print(f" - 종목: {p['ticker']} (섹터: {p['sector']}, 최근모멘텀: {p['momentum']*100:.2f}%)")

    portfolio_results, bnh_results = [], []
    for p in portfolio_stocks:
        opt_df = run_dynamic_hybrid_strategy(p['df'])
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
    print(f" 모멘텀 정렬 랭킹 적용 하이브리드 전략 성과")
    print(f"{'='*55}")
    print(f" [모멘텀 기반 동적 하이브리드 전략]")
    print(f"  - 최종 수익률 : {final_port_return * 100:.2f}%")
    print(f"  - 최대낙폭(MDD): {port_mdd * 100:.2f}%")
    print(f"{'-'*55}")
    print(f" [단순보유 (Buy & Hold)]")
    print(f"  - 최종 수익률 : {final_bnh_return * 100:.2f}%")
    print(f"  - 최대낙폭(MDD): {bnh_mdd * 100:.2f}%")
    print(f"{'='*55}")

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(cum_portfolio.index, cum_portfolio, label='Momentum-Ranked Hybrid Strategy', color='purple', linewidth=2)
    plt.plot(cum_bnh.index, cum_bnh, label='Buy & Hold', color='gray', linestyle='--', linewidth=1.5)
    plt.title('Momentum-Ranked Hybrid Strategy vs. Buy & Hold')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()