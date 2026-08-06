import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

START_DATE = "2020-01-01"
END_DATE = "2026-06-30"
COMMISSION_PCT = 0.0005
SLIPPAGE_PCT = 0.0005
TARGET_PROFIT_PCT = 0.35  # +35% 도달 시 익절 후 종목 교체

def get_historical_robust_universe():
    return {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Communication Services",
        "AMZN": "Consumer Cyclical", "META": "Communication Services", "UNH": "Healthcare", "JNJ": "Healthcare",
        "JPM": "Financials", "V": "Financials", "XOM": "Energy", "WMT": "Consumer Defensive", "KO": "Consumer Defensive"
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
    # NOTE: Trend_MA(200일)는 현재 전략 로직 어디에도 쓰이지 않는 미사용 컬럼입니다.
    # buffer_days=250(달력일 ≈ 영업일 178일)로는 200일 워밍업에 살짝 부족해서
    # 타겟구간 초반 며칠은 NaN이 나올 수 있지만, 실제 진입/청산 판단에는 영향 없습니다.
    df['Trend_MA'] = df['Close'].rolling(window=200).mean()
    df['Short_MA'] = df['Close'].rolling(window=20).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    bb_std_val = df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2.0 * bb_std_val
    df['RSI'] = calc_rsi(df['Close'], period=14)
    df['ATR'] = calc_atr(df, period=14)
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def run_dynamic_strategy_single_stock(df, market_trend_series, market_mom_series):
    """
    개별 종목 기준: ATR 트레일링 스톱 + +35% 목표 익절 로직이 포함된 시뮬레이션
    청산(익절 또는 손절/트레일링)이 발생한 날짜 인덱스를 반환하여 포트폴리오 리밸런싱에 활용

    [수정사항] 기존 코드는 entry_price/exit_price에 SLIPPAGE_PCT를 반영해 계산해놓고도
    실제 손익(daily_ret, net_day_ret)에는 반영하지 않아 슬리피지 비용이 전혀 차감되지
    않고 있었습니다. 진입일/청산일 모두 (COMMISSION_PCT + SLIPPAGE_PCT)를 비용으로
    차감하도록 수정했습니다.
    """
    df = df.copy()
    position = 0
    entry_price = 0.0
    highest_price = 0.0
    holding_days = 0
    strategy_returns = []
    exit_flags = [] # 청산 발생 여부 기록

    for i in range(len(df)):
        current_date = df.index[i]
        close = df['Close'].iloc[i]
        prev_close = df['Close'].iloc[i-1] if i > 0 else close
        short_ma = df['Short_MA'].iloc[i]
        rsi = df['RSI'].iloc[i]
        lower = df['BB_Lower'].iloc[i]
        atr = df['ATR'].iloc[i]

        is_market_bullish = market_trend_series.get(current_date, True)
        is_v_rebound = market_mom_series.get(current_date, False)

        if position == 0:
            strategy_returns.append(0.0)
            exit_flags.append(False)

            if is_market_bullish or is_v_rebound:
                is_condition = (pd.notna(short_ma) and close > short_ma) or (pd.notna(rsi) and rsi <= 55)
            else:
                is_condition = (pd.notna(rsi) and rsi <= 35) or (pd.notna(lower) and close <= lower)

            if is_condition:
                position = 1
                entry_price = close * (1 + SLIPPAGE_PCT)
                highest_price = entry_price
                holding_days = 0
                # [수정] 진입 비용 = 수수료 + 슬리피지 (기존: 수수료만 차감)
                strategy_returns[-1] = -(COMMISSION_PCT + SLIPPAGE_PCT)
        else:
            holding_days += 1
            if close > highest_price:
                highest_price = close

            current_atr_pct = (atr / close) if (pd.notna(atr) and close > 0) else 0.02

            # 1. 목표 익절 체크 (+35% 도달 시 강제 청산)
            hit_target_profit = close >= entry_price * (1 + TARGET_PROFIT_PCT)

            # 2. 트레일링 스톱 및 손절 체크
            if is_market_bullish or is_v_rebound:
                dynamic_trailing_pct = max(0.06, current_atr_pct * 3.0)
                hit_exit = (close <= highest_price * (1 - dynamic_trailing_pct)) or hit_target_profit
            else:
                dynamic_sl_pct = max(0.02, current_atr_pct * 1.5)
                dynamic_trailing_pct = max(0.03, current_atr_pct * 2.0)
                hit_sl = close <= entry_price * (1 - dynamic_sl_pct)
                hit_trailing = close <= highest_price * (1 - dynamic_trailing_pct)
                hit_exit = hit_sl or hit_trailing or holding_days >= 20 or hit_target_profit

            if hit_exit:
                daily_ret = (close / prev_close) - 1 if prev_close > 0 else 0
                # [수정] 청산 비용 = 수수료 + 슬리피지 (기존: 수수료만 차감, exit_price 미사용 변수였음)
                net_day_ret = (daily_ret * position) - (COMMISSION_PCT + SLIPPAGE_PCT)
                strategy_returns.append(net_day_ret)
                exit_flags.append(True) # 청산 발생!

                position = 0
                entry_price = 0.0
                highest_price = 0.0
                holding_days = 0
            else:
                daily_ret = (close / prev_close) - 1 if prev_close > 0 else 0
                strategy_returns.append(daily_ret * position)
                exit_flags.append(False)

    df['Strategy_Return'] = strategy_returns
    df['Exit_Flag'] = exit_flags
    return df

if __name__ == "__main__":
    print("[*] 시장 국면 분석용 SPY 데이터 로딩 중 (2020~2026)...")
    spy_df = fetch_data("SPY", START_DATE, END_DATE)
    if spy_df is not None:
        spy_df['Market_MA'] = spy_df['Close'].rolling(window=200).mean()
        market_trend = (spy_df['Close'] > spy_df['Market_MA']).to_dict()
        market_mom = (spy_df['Close'].pct_change(5) > 0.03).to_dict()
    else:
        market_trend, market_mom = {}, {}

    universe_dict = get_historical_robust_universe()
    all_dfs = {}

    print("[*] 유니버스 전체 데이터 준비 중...")
    for ticker, sector in universe_dict.items():
        df = fetch_data(ticker, START_DATE, END_DATE)
        if df is None: continue
        df = add_indicators(df)
        df_target = df[(df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))].copy()
        if len(df_target) >= 100:
            all_dfs[ticker] = {'sector': sector, 'df': df_target}

    # 타겟 기간 날짜 인덱스 생성
    sample_ticker = list(all_dfs.keys())[0]
    dates = all_dfs[sample_ticker]['df'].index

    # 동적 포트폴리오 시뮬레이션 (3개 종목 슬롯 분산)
    print("[*] 동적 리밸런싱 포트폴리오 시뮬레이션 실행 중...")

    # 각 종목별 전략 적용 데이터 사전 계산
    processed_dfs = {}
    for t, info in all_dfs.items():
        processed_dfs[t] = run_dynamic_strategy_single_stock(info['df'], market_trend, market_mom)

    portfolio_returns = []
    bnh_returns = []

    # 현재 포트폴리오에 담긴 종목 관리 (최대 3개, 섹터 중복 방지)
    current_portfolio = {} # {ticker: {'sector': s, 'entry_date': d}}

    for current_date in dates:
        # 1. 기존 포트폴리오 종목 중 청산(익절/손절)된 종목이 있는지 확인하고 제거
        tickers_to_remove = []
        for ticker in list(current_portfolio.keys()):
            p_df = processed_dfs[ticker]
            if current_date in p_df.index:
                if p_df.loc[current_date, 'Exit_Flag']:
                    tickers_to_remove.append(ticker)

        for t in tickers_to_remove:
            del current_portfolio[t]

        # 2. 슬롯이 3개 미만이면, 현재 시점 기준으로 모멘텀(120일 수익률)이 가장 높은 종목을 스크리닝하여 충원
        if len(current_portfolio) < 3:
            candidate_scores = []
            for ticker, info in all_dfs.items():
                if ticker in current_portfolio: continue
                # 섹터 중복 방지
                if info['sector'] in [current_portfolio[x]['sector'] for x in current_portfolio]: continue

                sub_df = info['df']
                if current_date in sub_df.index:
                    idx_pos = sub_df.index.get_loc(current_date)
                    if idx_pos >= 120:
                        mom_val = sub_df['Close'].iloc[idx_pos] / sub_df['Close'].iloc[idx_pos - 120] - 1
                        candidate_scores.append((ticker, info['sector'], mom_val))

            candidate_scores = sorted(candidate_scores, key=lambda x: x[2], reverse=True)

            for cand in candidate_scores:
                if len(current_portfolio) >= 3: break
                t, s, _ = cand
                current_portfolio[t] = {'sector': s}

        # 3. 당일 포트폴리오 구성 종목들의 전략 수익률 평균 산출
        day_port_ret = 0.0
        day_bnh_ret = 0.0

        active_tickers = list(current_portfolio.keys())
        if len(active_tickers) > 0:
            port_vals, bnh_vals = [], []
            for t in active_tickers:
                p_df = processed_dfs[t]
                if current_date in p_df.index:
                    port_vals.append(p_df.loc[current_date, 'Strategy_Return'])
                    bnh_vals.append(p_df.loc[current_date, 'Daily_Return'])
            if port_vals:
                day_port_ret = np.mean(port_vals)
                day_bnh_ret = np.mean(bnh_vals)

        portfolio_returns.append(day_port_ret)
        bnh_returns.append(day_bnh_ret)

    port_ret_series = pd.Series(portfolio_returns, index=dates)
    bnh_ret_series = pd.Series(bnh_returns, index=dates)

    cum_portfolio = (1 + port_ret_series).cumprod()
    final_port_return = cum_portfolio.iloc[-1] - 1
    port_mdd = ((cum_portfolio - cum_portfolio.cummax()) / cum_portfolio.cummax()).min()

    cum_bnh = (1 + bnh_ret_series).cumprod()
    final_bnh_return = cum_bnh.iloc[-1] - 1
    bnh_mdd = ((cum_bnh - cum_bnh.cummax()) / cum_bnh.cummax()).min()

    # 위험조정수익률(샤프비율, 무위험수익률 0% 가정, 연 252거래일)
    port_daily_std = port_ret_series.std()
    port_sharpe = (port_ret_series.mean() / port_daily_std) * np.sqrt(252) if port_daily_std > 0 else np.nan
    bnh_daily_std = bnh_ret_series.std()
    bnh_sharpe = (bnh_ret_series.mean() / bnh_daily_std) * np.sqrt(252) if bnh_daily_std > 0 else np.nan

    print(f"\n{'='*55}")
    print(f" 동적 리밸런싱 전략 최종 성과 (2020~2026 상반기)")
    print(f" [슬리피지 반영 수정판]")
    print(f"{'='*55}")
    print(f" [동적 리밸런싱 전략 (+35% 익절 + 트레일링)]")
    print(f"  - 최종 수익률 : {final_port_return * 100:.2f}%")
    print(f"  - 최대낙폭(MDD): {port_mdd * 100:.2f}%")
    print(f"  - 샤프비율     : {port_sharpe:.2f}")
    print(f"{'-'*55}")
    print(f" [단순보유 (Buy & Hold)]")
    print(f"  - 최종 수익률 : {final_bnh_return * 100:.2f}%")
    print(f"  - 최대낙폭(MDD): {bnh_mdd * 100:.2f}%")
    print(f"  - 샤프비율     : {bnh_sharpe:.2f}")
    print(f"{'='*55}")

    plt.figure(figsize=(12, 6))
    plt.plot(cum_portfolio.index, cum_portfolio, label='Dynamic Rebalancing Strategy (+35% TP, slippage-fixed)', color='purple', linewidth=2)
    plt.plot(cum_bnh.index, cum_bnh, label='Buy & Hold', color='gray', linestyle='--', linewidth=1.5)
    plt.title('Dynamic Rebalancing Strategy vs. Buy & Hold (2020-2026)')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()