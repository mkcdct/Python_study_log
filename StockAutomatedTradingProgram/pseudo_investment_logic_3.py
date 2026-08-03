# Backtesting logic for the investment strategy
# golden cross/death cross 전략의 샤프비율 분석 & 2023년 횡보장에서의 유효성 확인
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKER = "NVDA"
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  # 무위험수익률. 필요하면 연 3~4% 정도로 바꿔서 비교해도 됨


# ------------------------------
# 1. 데이터 다운로드
# ------------------------------
def fetch_data(ticker, start=None, end=None, period=None):
    if period:
        df = yf.download(ticker, period=period, interval="1d")
    else:
        df = yf.download(ticker, start=start, end=end, interval="1d")
    df.columns = df.columns.get_level_values(0)  # 멀티인덱스 제거
    return df


# ------------------------------
# 2. 골든크로스 + 거래량 필터 백테스트
# ------------------------------
def run_golden_cross_backtest(df, ma_short=20, ma_long=60, volume_window=3):
    df = df.copy()

    df['MA_short'] = df['Close'].rolling(window=ma_short).mean()
    df['MA_long'] = df['Close'].rolling(window=ma_long).mean()

    df['Signal'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1

    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Confirm'] = df['Volume'] > df['Volume_MA20']

    df['Golden_Cross'] = (df['Signal'] == 1) & (df['Signal'].shift(1) == 0)
    df['Death_Cross'] = (df['Signal'] == 0) & (df['Signal'].shift(1) == 1)

    df['Volume_Confirm_Nearby'] = (
        df['Volume_Confirm']
        .rolling(window=2 * volume_window + 1, center=True, min_periods=1)
        .max()
        .astype(bool)
    )

    df['Filtered_Signal'] = 0
    position = 0
    for i in range(len(df)):
        # 버그 수정: Volume_Confirm이 아니라 Volume_Confirm_Nearby를 써야 함
        if df['Golden_Cross'].iloc[i] and df['Volume_Confirm_Nearby'].iloc[i]:
            position = 1
        elif df['Death_Cross'].iloc[i]:
            position = 0
        df.iloc[i, df.columns.get_loc('Filtered_Signal')] = position

    df['Position'] = df['Filtered_Signal'].shift(1)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']

    df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

    return df


# ------------------------------
# 3. 샤프비율 & MDD 계산
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


# ------------------------------
# 4. 한 기간에 대한 전체 분석 실행
# ------------------------------
def analyze_period(label, ticker=TICKER, start=None, end=None, period=None):
    print(f"\n{'='*50}")
    print(f"[{label}] 분석 시작")
    print(f"{'='*50}")

    raw_df = fetch_data(ticker, start=start, end=end, period=period)
    df = run_golden_cross_backtest(raw_df)

    final_market_return = df['Cumulative_Market_Return'].iloc[-1] - 1
    final_strategy_return = df['Cumulative_Strategy'].iloc[-1] - 1

    mdd_market = calc_mdd(df['Cumulative_Market_Return'])
    mdd_strategy = calc_mdd(df['Cumulative_Strategy'])

    sharpe_market = calc_sharpe_ratio(df['Daily_Return'])
    sharpe_strategy = calc_sharpe_ratio(df['Strategy_Return'])

    print(f"[단순 보유] 최종수익률: {final_market_return*100:6.2f}%  "
          f"MDD: {mdd_market*100:6.2f}%  샤프비율: {sharpe_market:5.2f}")
    print(f"[골든크로스] 최종수익률: {final_strategy_return*100:6.2f}%  "
          f"MDD: {mdd_strategy*100:6.2f}%  샤프비율: {sharpe_strategy:5.2f}")

    return {
        'label': label,
        'df': df,
        'market_return': final_market_return,
        'strategy_return': final_strategy_return,
        'mdd_market': mdd_market,
        'mdd_strategy': mdd_strategy,
        'sharpe_market': sharpe_market,
        'sharpe_strategy': sharpe_strategy,
    }


# ------------------------------
# 5. 실행: 여러 기간 비교 (개수 자유롭게 늘리거나 줄이면 됨)
# ------------------------------
PERIODS = [
    {'label': '최근 1년', 'period': '1y'},
    {'label': '2023년', 'start': '2023-01-01', 'end': '2023-12-31'},
    {'label': '2022년', 'start': '2022-01-01', 'end': '2022-12-31'},
    {'label': '2020년', 'start': '2020-01-01', 'end': '2020-12-31'},
]

if __name__ == "__main__":
    results = [analyze_period(**p) for p in PERIODS]

    # 요약 비교 표
    print(f"\n{'='*50}")
    print("요약 비교")
    print(f"{'='*50}")

    rows = ['최종수익률(단순보유)', '최종수익률(골든크로스)',
            'MDD(단순보유)', 'MDD(골든크로스)',
            '샤프비율(단순보유)', '샤프비율(골든크로스)']

    summary_data = {'구분': rows}
    for r in results:
        summary_data[r['label']] = [
            f"{r['market_return']*100:.2f}%",
            f"{r['strategy_return']*100:.2f}%",
            f"{r['mdd_market']*100:.2f}%",
            f"{r['mdd_strategy']*100:.2f}%",
            f"{r['sharpe_market']:.2f}",
            f"{r['sharpe_strategy']:.2f}",
        ]
    summary = pd.DataFrame(summary_data)
    print(summary.to_string(index=False))

    # 시각화 (기간 개수만큼 자동으로 나란히 배치)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]  # subplot이 1개일 때도 반복문이 동작하도록 리스트로 감싸줌

    for ax, result in zip(axes, results):
        d = result['df']
        ax.plot(d.index, d['Cumulative_Market_Return'], label='Buy & Hold')
        ax.plot(d.index, d['Cumulative_Strategy'], label='Golden Cross Strategy')
        ax.set_title(f"NVDA: {result['label']}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    '''
    하락/급변 구간(2020, 2022)에서는 해당 전략이 원래 설계 목적인
    손실 방어, 변동성 완화 달성에는 작동함을 확인했음.
    이 전략을 상승장에서 굳이 쓸 이유는 없지만, 하락/급변 국면의 장세에서는 쓸 만함.
    '''