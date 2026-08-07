import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# 백테스팅 기간 설정 (2020년 ~ 2026년 상반기)
START_DATE = "2020-01-01"
END_DATE = "2026-06-30"

COMMISSION_PCT = 0.0005
BASE_SLIPPAGE_PCT = 0.0005
TARGET_PROFIT_PCT = 0.50  # +50% 도달 시 50% 분할 익절 후 나머지 트레일링

# 파킹통장(CMA 등) 연동 가정: 연 2.5% 금리를 일일 복리로 적용
CASH_ANNUAL_RATE = 0.025
CASH_RETURN_DAILY = (1.0 + CASH_ANNUAL_RATE) ** (1.0 / 252.0) - 1.0

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
    df['Short_MA'] = df['Close'].rolling(window=20).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    bb_std_val = df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2.0 * bb_std_val
    df['RSI'] = calc_rsi(df['Close'], period=14)
    df['ATR'] = calc_atr(df, period=14)
    
    # [룩어헤드 차단] 지표는 전일(shift(1)) 기준 정렬
    df['Signal_Short_MA'] = df['Short_MA'].shift(1)
    df['Signal_RSI'] = df['RSI'].shift(1)
    df['Signal_Lower'] = df['BB_Lower'].shift(1)
    df['Signal_ATR'] = df['ATR'].shift(1)
    df['Signal_Close'] = df['Close'].shift(1)
    
    return df

def run_dynamic_strategy_single_stock(df, market_trend_series, market_mom_series):
    df = df.copy()
    position = 0.0          
    entry_price = 0.0
    highest_price = 0.0
    holding_days = 0
    strategy_returns = []
    exit_flags = []

    for i in range(len(df)):
        current_date = df.index[i]
        open_p = df['Open'].iloc[i]
        close_p = df['Close'].iloc[i]
        
        sig_close = df['Signal_Close'].iloc[i]
        short_ma = df['Signal_Short_MA'].iloc[i]
        rsi = df['Signal_RSI'].iloc[i]
        lower = df['Signal_Lower'].iloc[i]
        atr = df['Signal_ATR'].iloc[i]

        is_market_bullish = market_trend_series.get(current_date, True)
        is_v_rebound = market_mom_series.get(current_date, False)

        current_atr_pct = (atr / sig_close) if (pd.notna(atr) and sig_close > 0) else 0.02
        dynamic_slippage = BASE_SLIPPAGE_PCT + max(0.0, current_atr_pct * 0.2)

        if position == 0.0:
            strategy_returns.append(0.0)
            exit_flags.append(False)

            if pd.isna(sig_close):
                continue

            if is_market_bullish or is_v_rebound:
                is_condition = (pd.notna(short_ma) and sig_close > short_ma) or (pd.notna(rsi) and rsi <= 55)
            else:
                is_condition = (pd.notna(rsi) and rsi <= 35) or (pd.notna(lower) and sig_close <= lower)

            if is_condition:
                position = 1.0
                entry_price = open_p * (1 + dynamic_slippage)
                highest_price = open_p
                holding_days = 0
                day_ret = (close_p / open_p) - 1 if open_p > 0 else 0
                net_day_ret = day_ret - (COMMISSION_PCT + dynamic_slippage)
                strategy_returns[-1] = net_day_ret
        else:
            holding_days += 1
            if close_p > highest_price:
                highest_price = close_p

            prev_close = df['Close'].iloc[i-1] if i > 0 else close_p
            daily_ret = (close_p / prev_close) - 1 if prev_close > 0 else 0

            hit_target_profit = (position == 1.0) and (sig_close >= entry_price * (1 + TARGET_PROFIT_PCT))

            if hit_target_profit:
                half_profit_ret = ((entry_price * (1 + TARGET_PROFIT_PCT)) / prev_close - 1) * 0.5
                net_day_ret = half_profit_ret + (daily_ret * 0.5) - COMMISSION_PCT
                strategy_returns.append(net_day_ret)
                exit_flags.append(False) 
                position = 0.5  
            else:
                if is_market_bullish or is_v_rebound:
                    dynamic_trailing_pct = max(0.06, current_atr_pct * 3.0)
                    hit_exit = (sig_close <= highest_price * (1 - dynamic_trailing_pct))
                else:
                    dynamic_sl_pct = max(0.02, current_atr_pct * 1.5)
                    dynamic_trailing_pct = max(0.03, current_atr_pct * 2.0)
                    hit_sl = sig_close <= entry_price * (1 - dynamic_sl_pct)
                    hit_trailing = sig_close <= highest_price * (1 - dynamic_trailing_pct)
                    hit_exit = hit_sl or hit_trailing

                if hit_exit:
                    net_day_ret = (daily_ret * position) - COMMISSION_PCT
                    strategy_returns.append(net_day_ret)
                    exit_flags.append(True) 

                    position = 0.0
                    entry_price = 0.0
                    highest_price = 0.0
                    holding_days = 0
                else:
                    strategy_returns.append(daily_ret * position)
                    exit_flags.append(False)

    df['Strategy_Return'] = strategy_returns
    df['Exit_Flag'] = exit_flags
    return df

if __name__ == "__main__":
    print("[*] 시장 국면 분석용 SPY 데이터 로딩 중...")
    spy_df = fetch_data("SPY", START_DATE, END_DATE)
    if spy_df is not None:
        spy_df['Market_MA'] = spy_df['Close'].rolling(window=200).mean()
        market_trend = (spy_df['Close'] > spy_df['Market_MA']).shift(1).to_dict()
        market_mom = (spy_df['Close'].pct_change(5) > 0.03).shift(1).to_dict()
    else:
        market_trend, market_mom = {}, {}

    universe_dict = get_modern_robust_universe()
    all_dfs = {}

    print("[*] 유니버스 데이터 준비 및 백테스트 실행 중...")
    for ticker, sector in universe_dict.items():
        df = fetch_data(ticker, START_DATE, END_DATE)
        if df is None: continue
        df = add_indicators(df)
        df_target = df[(df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))].copy()
        if len(df_target) >= 100:
            all_dfs[ticker] = {'sector': sector, 'df': df_target}

    sample_ticker = list(all_dfs.keys())[0]
    dates = all_dfs[sample_ticker]['df'].index

    processed_dfs = {}
    for t, info in all_dfs.items():
        processed_dfs[t] = run_dynamic_strategy_single_stock(info['df'], market_trend, market_mom)

    portfolio_returns = []
    current_portfolio = {}
    MAX_SLOTS = 3  

    for current_date in dates:
        tickers_to_remove = []
        for ticker in list(current_portfolio.keys()):
            p_df = processed_dfs[ticker]
            if current_date in p_df.index:
                if p_df.loc[current_date, 'Exit_Flag']:
                    tickers_to_remove.append(ticker)

        for t in tickers_to_remove:
            del current_portfolio[t]

        if len(current_portfolio) < MAX_SLOTS:
            candidate_scores = []
            for ticker, info in all_dfs.items():
                if ticker in current_portfolio: continue
                if info['sector'] in [current_portfolio[x]['sector'] for x in current_portfolio]: continue

                sub_df = info['df']
                if current_date in sub_df.index:
                    idx_pos = sub_df.index.get_loc(current_date)
                    if idx_pos >= 120:
                        mom_val = sub_df['Close'].iloc[idx_pos - 1] / sub_df['Close'].iloc[idx_pos - 121] - 1
                        candidate_scores.append((ticker, info['sector'], mom_val))

            candidate_scores = sorted(candidate_scores, key=lambda x: x[2], reverse=True)

            for cand in candidate_scores:
                if len(current_portfolio) >= MAX_SLOTS: break
                t, s, _ = cand
                current_portfolio[t] = {'sector': s}

        day_port_ret = 0.0
        active_tickers = list(current_portfolio.keys())
        active_count = len(active_tickers)
        
        if active_count > 0:
            port_vals = []
            for t in active_tickers:
                p_df = processed_dfs[t]
                if current_date in p_df.index:
                    port_vals.append(p_df.loc[current_date, 'Strategy_Return'])
            
            if port_vals:
                stock_weight = active_count / MAX_SLOTS
                cash_weight = 1.0 - stock_weight
                weighted_stock_ret = np.mean(port_vals) * stock_weight
                weighted_cash_ret = CASH_RETURN_DAILY * cash_weight
                day_port_ret = weighted_stock_ret + weighted_cash_ret
        else:
            day_port_ret = CASH_RETURN_DAILY

        portfolio_returns.append(day_port_ret)

    port_ret_series = pd.Series(portfolio_returns, index=dates)

    # ==========================================
    # 🎲 몬테카를로 시뮬레이션 실행 (1,000회 셔플)
    # ==========================================
    print("[*] 몬테카를로 시뮬레이션(1,000회 재샘플링) 실행 중...")
    NUM_SIMULATIONS = 1000
    sim_returns_matrix = np.zeros((len(port_ret_series), NUM_SIMULATIONS))

    returns_array = port_ret_series.values

    for sim in range(NUM_SIMULATIONS):
        # 일일 수익률 배열을 무작위로 섞음 (Bootstrapping / Shuffling)
        shuffled_returns = np.random.choice(returns_array, size=len(returns_array), replace=True)
        cum_sim = np.cumprod(1 + shuffled_returns)
        sim_returns_matrix[:, sim] = cum_sim

    # 통계치 계산
    final_sim_returns = sim_returns_matrix[-1, :] - 1
    mean_final_return = np.mean(final_sim_returns)
    percentile_5_return = np.percentile(final_sim_returns, 5)   # 최악의 경우 (하위 5%)
    percentile_95_return = np.percentile(final_sim_returns, 95) # 행운의 경우 (상위 5%)

    # 가상 시뮬레이션별 MDD 계산
    sim_mdds = []
    for sim in range(NUM_SIMULATIONS):
        curve = sim_returns_matrix[:, sim]
        mdd = np.min((curve - np.maximum.accumulate(curve)) / np.maximum.accumulate(curve))
        sim_mdds.append(mdd)
    
    mean_mdd = np.mean(sim_mdds)
    worst_mdd_5pct = np.percentile(sim_mdds, 5) # 최악의 MDD (하위 5% 위험선)

    print(f"\n{'='*55}")
    print(f" 🎲 몬테카를로 시뮬레이션 결과 (1,000회 가상 검증)")
    print(f"{'='*55}")
    print(f" [최종 누적 수익률 분포]")
    print(f"  - 평균 예측 수익률 : {mean_final_return * 100:.2f}%")
    print(f"  - 하위 5% (최악)   : {percentile_5_return * 100:.2f}%")
    print(f"  - 상위 5% (최고)   : {percentile_95_return * 100:.2f}%")
    print(f"{'-'*55}")
    print(f" [최대 낙폭(MDD) 리스크 분포]")
    print(f"  - 평균 예상 MDD    : {mean_mdd * 100:.2f}%")
    print(f"  - 하위 5% (최악 MDD): {worst_mdd_5pct * 100:.2f}%")
    print(f"{'='*55}")

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(dates, sim_returns_matrix, color='lightgray', alpha=0.05) # 1000개 선
    plt.plot(dates, np.mean(sim_returns_matrix, axis=1), color='blue', linewidth=2, label='Mean Simulation')
    plt.plot(dates, np.percentile(sim_returns_matrix, 5, axis=1), color='red', linestyle='--', linewidth=1.5, label='Worst 5% (VaR)')
    plt.title('Monte Carlo Simulation of Optimized Strategy (1,000 Paths)')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()