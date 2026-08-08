'''
최근 3년간 하락 추세인 종목 화이자, 인텔, 보잉을 대상으로
하락 추세 속에서의 짧은 반등 구간에서 수익 실현 전략 확인.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# 여러 종목 설정 (리스트 형태)
TICKERS = ["INTC", "BA", "PFE"]
START_DATE = "2020-01-01"
END_DATE = "2026-12-31"

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.03


# ------------------------------
# 1. 데이터 다운로드 (단일 종목씩 안전하게 다운로드)
# ------------------------------
def fetch_data(ticker, start, end, buffer_days=250):
    buffered_start = (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    # 단일 종목 문자열을 넘겨 MultiIndex 문제를 원천 차단
    df = yf.download(ticker, start=buffered_start, end=end, interval="1d", progress=False, multi_level_index=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ------------------------------
# 2. 기술적 지표 계산
# ------------------------------
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def add_indicators(df, bb_window=20, bb_std=2.0, rsi_period=14, trend_ma_window=200):
    df = df.copy()
    
    df['Trend_MA'] = df['Close'].rolling(window=trend_ma_window).mean()
    df['BB_Mid'] = df['Close'].rolling(window=bb_window).mean()
    bb_std_val = df['Close'].rolling(window=bb_window).std()
    df['BB_Upper'] = df['BB_Mid'] + bb_std * bb_std_val
    df['BB_Lower'] = df['BB_Mid'] - bb_std * bb_std_val
    df['RSI'] = calc_rsi(df['Close'], period=rsi_period)
    df['Daily_Return'] = df['Close'].pct_change()
    
    return df


# ------------------------------
# 3. 하락 추세 단기 반등 전략 함수
# ------------------------------
def run_downtrend_bounce_strategy(df, rsi_oversold=35, take_profit_pct=0.04, stop_loss_pct=0.03, max_holding_days=10):
    df = df.copy()
    
    position = 0
    entry_price = 0.0
    holding_days = 0
    filtered_signal = []

    for i in range(len(df)):
        close = df['Close'].iloc[i]
        trend_ma = df['Trend_MA'].iloc[i]
        rsi = df['RSI'].iloc[i]
        lower = df['BB_Lower'].iloc[i]
        mid = df['BB_Mid'].iloc[i]

        if position == 0:
            is_downtrend = pd.notna(trend_ma) and close < trend_ma
            is_oversold = (pd.notna(rsi) and rsi <= rsi_oversold) or (pd.notna(lower) and close <= lower)
            
            if is_downtrend and is_oversold:
                position = 1
                entry_price = close
                holding_days = 0
        else:
            holding_days += 1

            hit_tp = close >= entry_price * (1 + take_profit_pct)
            hit_sl = close <= entry_price * (1 - stop_loss_pct)
            hit_time_limit = holding_days >= max_holding_days
            hit_mid_target = pd.notna(mid) and close >= mid

            if hit_tp or hit_sl or hit_time_limit or hit_mid_target:
                position = 0
                entry_price = 0.0
                holding_days = 0

        filtered_signal.append(position)

    df['Signal'] = filtered_signal
    df['Position'] = df['Signal'].shift(1)
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    return df


# ------------------------------
# 4. 성과 분석 지표 함수
# ------------------------------
def calc_sharpe_ratio(daily_returns, risk_free_rate=RISK_FREE_RATE):
    daily_returns = daily_returns.dropna()
    excess_return = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    if excess_return.std() == 0:
        return np.nan
    return (excess_return.mean() / excess_return.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def calc_mdd(cumulative_series):
    peak = cumulative_series.cummax()
    drawdown = (cumulative_series - peak) / peak
    return drawdown.min()


def summarize(returns_series):
    cumulative = (1 + returns_series.fillna(0)).cumprod()
    final_return = cumulative.iloc[-1] - 1
    mdd = calc_mdd(cumulative)
    sharpe = calc_sharpe_ratio(returns_series)
    return cumulative, final_return, mdd, sharpe


# ------------------------------
# 5. 백테스팅 실행 및 결과 출력
# ------------------------------
if __name__ == "__main__":
    all_results = []
    
    for ticker in TICKERS:
        print(f"[{ticker}] 백테스팅 분석 중...")
        raw_df = fetch_data(ticker, start=START_DATE, end=END_DATE)
        df = add_indicators(raw_df)
        
        # 기간 필터링
        df = df[(df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))].copy()
        df = run_downtrend_bounce_strategy(df)
        
        _, ret_market, mdd_market, sharpe_market = summarize(df['Daily_Return'])
        _, ret_strat, mdd_strat, sharpe_strat = summarize(df['Strategy_Return'])
        
        all_results.append({
            'ticker': ticker,
            'df': df,
            'ret_market': ret_market, 'mdd_market': mdd_market, 'sharpe_market': sharpe_market,
            'ret_strat': ret_strat, 'mdd_strat': mdd_strat, 'sharpe_strat': sharpe_strat
        })

    print(f"\n{'='*65}")
    print(f"하락 추세 단기 반등 전략 다중 종목 검증 결과 ({START_DATE[:4]} ~ {END_DATE[:4]})")
    print(f"{'='*65}")
    print(f"{'종목':6s} | {'전략 구분':15s} | {'최종수익률':10s} | {'MDD':8s} | {'샤프비율':8s}")
    print("-" * 58)
    
    for r in all_results:
        t = r['ticker']
        print(f"{t:6s} | {'단순보유 (Buy&Hold)':15s} | {r['ret_market']*100:9.2f}% | {r['mdd_market']*100:7.2f}% | {r['sharpe_market']:8.2f}")
        print(f"{t:6s} | {'하락추세 반등전략':15s} | {r['ret_strat']*100:9.2f}% | {r['mdd_strat']*100:7.2f}% | {r['sharpe_strat']:8.2f}")
        print("-" * 58)

    # ------------------------------
    # 6. 시각화 (종목별 서브플롯)
    # ------------------------------
    fig, axes = plt.subplots(len(TICKERS), 1, figsize=(10, 4 * len(TICKERS)))
    if len(TICKERS) == 1:
        axes = [axes]

    for ax, r in zip(axes, all_results):
        d = r['df']
        t = r['ticker']
        ax.plot(d.index, (1 + d['Daily_Return'].fillna(0)).cumprod(), label=f'{t} Buy & Hold', color='gray', linestyle='--')
        ax.plot(d.index, (1 + d['Strategy_Return'].fillna(0)).cumprod(), label=f'{t} Downtrend Bounce Strategy', color='crimson')
        ax.set_title(f"[{t}] Downtrend Bounce Strategy Validation")
        ax.set_ylabel('Cumulative Return')
        ax.legend(loc='upper left')
        ax.grid(True)

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()