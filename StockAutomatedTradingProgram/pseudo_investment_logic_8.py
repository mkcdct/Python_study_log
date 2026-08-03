'''
logic_7과의 차이
종목을 조금 더 광범위한 S&P500 ETF로 변경
이동평균선 추세 필터 & 트레일링 스탑 전략 반영
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# 종목 설정 (S&P 500 ETF)
TICKER = "SPY"
START_DATE = "2020-01-01"
END_DATE = "2026-12-31"

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.03


# ------------------------------
# 1. 데이터 다운로드
# ------------------------------
def fetch_data(ticker, start, end, buffer_days=250):
    buffered_start = (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=buffered_start, end=end, interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ------------------------------
# 2. 지표 계산 (장기 이평선 등)
# ------------------------------
def add_indicators(df, trend_ma_window=200):
    df = df.copy()
    df['Trend_MA'] = df['Close'].rolling(window=trend_ma_window).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    return df


# ------------------------------
# 3. 추세 필터 + 트레일링 스탑 전략 함수
# ------------------------------
def run_trend_trailing_strategy(df, trailing_stop_pct=0.08, trend_ma_window=200):
    """
    - 진입 조건: 종가가 장기 이평선(Trend_MA) 위에 위치할 때만 매수 허용 (상승 추세장)
    - 청산 조건: 보유 중 최고가 대비 설정된 비율(trailing_stop_pct) 이상 하락하거나,
                 종가가 장기 이평선 아래로 떨어질 때 전량 매도
    """
    df = df.copy()
    
    position = 0
    highest_price = 0.0
    filtered_signal = []

    for i in range(len(df)):
        close = df['Close'].iloc[i]
        trend_ma = df['Trend_MA'].iloc[i]

        if position == 0:
            # 진입 조건: 장기 이평선 위 (상승 추세)
            if pd.notna(trend_ma) and close > trend_ma:
                position = 1
                highest_price = close  # 진입 시점부터 최고가 추적 시작
        else:
            # 보유 중 최고가 갱신
            if close > highest_price:
                highest_price = close

            # 청산 조건 ①: 트레일링 스탑 (최고가 대비 일정 비율 하락 시 이탈)
            hit_trailing_stop = close <= highest_price * (1 - trailing_stop_pct)
            
            # 청산 조건 ②: 추세 이탈 (장기 이평선 아래로 하향 이탈)
            hit_trend_break = pd.notna(trend_ma) and close < trend_ma

            if hit_trailing_stop or hit_trend_break:
                position = 0
                highest_price = 0.0

        filtered_signal.append(position)

    df['Signal'] = filtered_signal
    df['Position'] = df['Signal'].shift(1)  # 룩어헤드 방지
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
# 5. 백테스팅 실행 및 출력
# ------------------------------
if __name__ == "__main__":
    raw_df = fetch_data(TICKER, start=START_DATE, end=END_DATE)
    df = add_indicators(raw_df)
    
    # 지정한 백테스팅 기간 자르기
    df = df[(df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))].copy()
    
    # 전략 적용 (트레일링 스탑 8% 기준)
    df = run_trend_trailing_strategy(df, trailing_stop_pct=0.08)
    
    cum_market, ret_market, mdd_market, sharpe_market = summarize(df['Daily_Return'])
    cum_strat, ret_strat, mdd_strat, sharpe_strat = summarize(df['Strategy_Return'])

    print(f"\n{'='*55}")
    print(f"[{TICKER}] S&P 500 추세 추종 + 트레일링 스탑 전략 성과 비교 ({START_DATE[:4]} ~ {END_DATE[:4]})")
    print(f"{'='*55}")
    print(f"{'전략 구분':15s} | {'최종수익률':10s} | {'MDD':8s} | {'샤프비율':8s}")
    print("-" * 50)
    print(f"{'단순보유 (Buy&Hold)':15s} | {ret_market*100:9.2f}% | {mdd_market*100:7.2f}% | {sharpe_market:8.2f}")
    print(f"{'추세+트레일링스탑':15s} | {ret_strat*100:9.2f}% | {mdd_strat*100:7.2f}% | {sharpe_strat:8.2f}")
    print("-" * 50)

    # ------------------------------
    # 6. 시각화
    # ------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, (1 + df['Daily_Return'].fillna(0)).cumprod(), label=f'{TICKER} Buy & Hold', color='gray', linestyle='--')
    plt.plot(df.index, (1 + df['Strategy_Return'].fillna(0)).cumprod(), label=f'{TICKER} Trend + Trailing Stop', color='crimson')
    plt.title(f"S&P 500 Trend Following & Trailing Stop Strategy ({START_DATE[:4]} - {END_DATE[:4]})")
    plt.ylabel('Cumulative Return')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # insight : 지수나 우상향 대형주는 단순 보유 전략이 가장 강력함.
    # 물론, 수익률이 조금 낮더라도 하락장의 공포를 피하고 싶다면 방어 전략/트레일링 스탑 전략이 의미있는 선택이 될 수도.
    