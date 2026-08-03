"""
골든크로스 전략 + RSI 눌림목 타이밍 조합 백테스팅 + 샤프비율 분석 + 기간별 비교

전략 3종 비교:
1. 단순보유 (Buy & Hold)
2. 골든크로스 (추세 필터 단독, 교차 시점에만 진입/청산)
3. 골든크로스 + RSI (골든크로스 국면 안에서 RSI 눌림목 반등 시점에 진입,
   국면이 끝나면(데드크로스) 청산 — 추세는 필터로, 타이밍은 RSI로)
"""

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
# 2. 지표 계산 (이평선, 거래량 필터, RSI)
# ------------------------------
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df, ma_short=20, ma_long=60, volume_window=3, rsi_period=14):
    df = df.copy()

    df['MA_short'] = df['Close'].rolling(window=ma_short).mean()
    df['MA_long'] = df['Close'].rolling(window=ma_long).mean()

    df['Signal'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1  # 골든크로스 국면 여부

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

    df['RSI'] = calc_rsi(df['Close'], period=rsi_period)

    df['Daily_Return'] = df['Close'].pct_change()
    return df


# ------------------------------
# 3. 전략 1: 골든크로스 단독 (교차 시점에만 진입/청산)
# ------------------------------
def run_golden_cross_only(df):
    df = df.copy()
    position = 0
    filtered_signal = []

    for i in range(len(df)):
        if df['Golden_Cross'].iloc[i] and df['Volume_Confirm_Nearby'].iloc[i]:
            position = 1
        elif df['Death_Cross'].iloc[i]:
            position = 0
        filtered_signal.append(position)

    df['Filtered_Signal'] = filtered_signal
    df['Position'] = df['Filtered_Signal'].shift(1)
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    return df


# ------------------------------
# 4. 전략 2: 골든크로스(추세 필터) + RSI 눌림목(타이밍)
#    - 골든크로스 국면(Signal==1) 안에서 RSI가 과매도 구간(rsi_oversold 이하)에
#      갔다가 다시 그 위로 올라오는 반등 시점에 진입
#    - 국면이 끝나면(데드크로스) 보유 여부와 상관없이 청산
# ------------------------------
def run_golden_cross_rsi(df, rsi_oversold=35, rsi_overbought=70, exit_on_overbought=False):
    df = df.copy()

    rsi_prev = df['RSI'].shift(1)
    df['RSI_Bounce'] = (rsi_prev <= rsi_oversold) & (df['RSI'] > rsi_oversold)

    position = 0
    filtered_signal = []

    for i in range(len(df)):
        regime_on = df['Signal'].iloc[i] == 1

        if position == 0:
            # 국면 안에서 RSI 눌림목 반등이 나오면 진입
            if regime_on and df['RSI_Bounce'].iloc[i]:
                position = 1
        else:
            # 국면이 끝나면(데드크로스) 무조건 청산
            if df['Death_Cross'].iloc[i]:
                position = 0
            # (선택) RSI 과매수 구간 진입 시에도 익절 청산하고 싶으면 아래 조건 사용
            elif exit_on_overbought and df['RSI'].iloc[i] >= rsi_overbought:
                position = 0

        filtered_signal.append(position)

    df['Filtered_Signal_RSI'] = filtered_signal
    df['Position_RSI'] = df['Filtered_Signal_RSI'].shift(1)
    df['Strategy_RSI_Return'] = df['Daily_Return'] * df['Position_RSI']
    return df


# ------------------------------
# 5. 샤프비율 & MDD
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
# 6. 한 기간에 대한 3개 전략 분석
# ------------------------------
def analyze_period(label, ticker=TICKER, start=None, end=None, period=None):
    print(f"\n{'='*60}")
    print(f"[{label}] 분석 시작")
    print(f"{'='*60}")

    raw_df = fetch_data(ticker, start=start, end=end, period=period)
    df = add_indicators(raw_df)
    df = run_golden_cross_only(df)
    df = run_golden_cross_rsi(df)

    cum_market, ret_market, mdd_market, sharpe_market = summarize(df['Daily_Return'])
    cum_gc, ret_gc, mdd_gc, sharpe_gc = summarize(df['Strategy_Return'])
    cum_rsi, ret_rsi, mdd_rsi, sharpe_rsi = summarize(df['Strategy_RSI_Return'])

    df['Cumulative_Market_Return'] = cum_market
    df['Cumulative_Strategy'] = cum_gc
    df['Cumulative_Strategy_RSI'] = cum_rsi

    print(f"{'전략':15s} {'최종수익률':>12s} {'MDD':>10s} {'샤프비율':>8s}")
    print(f"{'단순보유':15s} {ret_market*100:11.2f}% {mdd_market*100:9.2f}% {sharpe_market:8.2f}")
    print(f"{'골든크로스':15s} {ret_gc*100:11.2f}% {mdd_gc*100:9.2f}% {sharpe_gc:8.2f}")
    print(f"{'골든크로스+RSI':15s} {ret_rsi*100:11.2f}% {mdd_rsi*100:9.2f}% {sharpe_rsi:8.2f}")

    return {
        'label': label,
        'df': df,
        'market_return': ret_market, 'mdd_market': mdd_market, 'sharpe_market': sharpe_market,
        'gc_return': ret_gc, 'mdd_gc': mdd_gc, 'sharpe_gc': sharpe_gc,
        'rsi_return': ret_rsi, 'mdd_rsi': mdd_rsi, 'sharpe_rsi': sharpe_rsi,
    }


# ------------------------------
# 7. 실행: 여러 기간 비교
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
    print(f"\n{'='*60}")
    print("요약 비교")
    print(f"{'='*60}")

    rows = [
        '최종수익률(단순보유)', '최종수익률(골든크로스)', '최종수익률(골든크로스+RSI)',
        'MDD(단순보유)', 'MDD(골든크로스)', 'MDD(골든크로스+RSI)',
        '샤프비율(단순보유)', '샤프비율(골든크로스)', '샤프비율(골든크로스+RSI)',
    ]

    summary_data = {'구분': rows}
    for r in results:
        summary_data[r['label']] = [
            f"{r['market_return']*100:.2f}%",
            f"{r['gc_return']*100:.2f}%",
            f"{r['rsi_return']*100:.2f}%",
            f"{r['mdd_market']*100:.2f}%",
            f"{r['mdd_gc']*100:.2f}%",
            f"{r['mdd_rsi']*100:.2f}%",
            f"{r['sharpe_market']:.2f}",
            f"{r['sharpe_gc']:.2f}",
            f"{r['sharpe_rsi']:.2f}",
        ]
    summary = pd.DataFrame(summary_data)
    print(summary.to_string(index=False))

    # 시각화
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        d = result['df']
        ax.plot(d.index, d['Cumulative_Market_Return'], label='Buy & Hold')
        ax.plot(d.index, d['Cumulative_Strategy'], label='Golden Cross')
        ax.plot(d.index, d['Cumulative_Strategy_RSI'], label='Golden Cross + RSI')
        ax.set_title(f"NVDA: {result['label']}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()


# 상대적으로 변동성이 작은 코카콜라(KO) 종목으로도 동일 전략 적용
TICKER = "KO"  # 코카콜라 - 사회 이슈/트렌드에 상대적으로 둔감한 종목으로 비교
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
# 2. 지표 계산 (이평선, 거래량 필터, RSI)
# ------------------------------
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df, ma_short=20, ma_long=60, volume_window=3, rsi_period=14):
    df = df.copy()

    df['MA_short'] = df['Close'].rolling(window=ma_short).mean()
    df['MA_long'] = df['Close'].rolling(window=ma_long).mean()

    df['Signal'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1  # 골든크로스 국면 여부

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

    df['RSI'] = calc_rsi(df['Close'], period=rsi_period)

    df['Daily_Return'] = df['Close'].pct_change()
    return df


# ------------------------------
# 3. 전략 1: 골든크로스 단독 (교차 시점에만 진입/청산)
# ------------------------------
def run_golden_cross_only(df):
    df = df.copy()
    position = 0
    filtered_signal = []

    for i in range(len(df)):
        if df['Golden_Cross'].iloc[i] and df['Volume_Confirm_Nearby'].iloc[i]:
            position = 1
        elif df['Death_Cross'].iloc[i]:
            position = 0
        filtered_signal.append(position)

    df['Filtered_Signal'] = filtered_signal
    df['Position'] = df['Filtered_Signal'].shift(1)
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    return df


# ------------------------------
# 4. 전략 2: 골든크로스(추세 필터) + RSI 눌림목(추가 진입 신호)
#    - 진입 신호 ①: 골든크로스 발생 + 거래량 확인 (기존 전략과 동일한 기본 진입)
#    - 진입 신호 ②: 국면(Signal==1)이 켜져 있는 동안 RSI가 과매도(rsi_oversold 이하)
#      에서 반등할 때 (추가 진입 — 골든크로스 이후 눌림목 재진입 기회를 놓치지 않기 위함)
#    - 청산: 국면이 끝나면(데드크로스) 무조건 청산
# ------------------------------
def run_golden_cross_rsi(df, rsi_oversold=35, rsi_overbought=70, exit_on_overbought=False):
    df = df.copy()

    rsi_prev = df['RSI'].shift(1)
    df['RSI_Bounce'] = (rsi_prev <= rsi_oversold) & (df['RSI'] > rsi_oversold)

    position = 0
    filtered_signal = []

    for i in range(len(df)):
        regime_on = df['Signal'].iloc[i] == 1
        golden_cross_entry = df['Golden_Cross'].iloc[i] and df['Volume_Confirm_Nearby'].iloc[i]
        rsi_bounce_entry = regime_on and df['RSI_Bounce'].iloc[i]

        if position == 0:
            # 골든크로스 발생 시점 진입 OR 국면 중 RSI 눌림목 반등 시점 진입
            if golden_cross_entry or rsi_bounce_entry:
                position = 1
        else:
            # 국면이 끝나면(데드크로스) 무조건 청산
            if df['Death_Cross'].iloc[i]:
                position = 0
            # (선택) RSI 과매수 구간 진입 시에도 익절 청산하고 싶으면 아래 조건 사용
            elif exit_on_overbought and df['RSI'].iloc[i] >= rsi_overbought:
                position = 0

        filtered_signal.append(position)

    df['Filtered_Signal_RSI'] = filtered_signal
    df['Position_RSI'] = df['Filtered_Signal_RSI'].shift(1)
    df['Strategy_RSI_Return'] = df['Daily_Return'] * df['Position_RSI']
    return df


# ------------------------------
# 5. 샤프비율 & MDD
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
# 6. 한 기간에 대한 3개 전략 분석
# ------------------------------
def analyze_period(label, ticker=TICKER, start=None, end=None, period=None):
    print(f"\n{'='*60}")
    print(f"[{label}] 분석 시작")
    print(f"{'='*60}")

    raw_df = fetch_data(ticker, start=start, end=end, period=period)
    df = add_indicators(raw_df)
    df = run_golden_cross_only(df)
    df = run_golden_cross_rsi(df)

    cum_market, ret_market, mdd_market, sharpe_market = summarize(df['Daily_Return'])
    cum_gc, ret_gc, mdd_gc, sharpe_gc = summarize(df['Strategy_Return'])
    cum_rsi, ret_rsi, mdd_rsi, sharpe_rsi = summarize(df['Strategy_RSI_Return'])

    df['Cumulative_Market_Return'] = cum_market
    df['Cumulative_Strategy'] = cum_gc
    df['Cumulative_Strategy_RSI'] = cum_rsi

    print(f"{'전략':15s} {'최종수익률':>12s} {'MDD':>10s} {'샤프비율':>8s}")
    print(f"{'단순보유':15s} {ret_market*100:11.2f}% {mdd_market*100:9.2f}% {sharpe_market:8.2f}")
    print(f"{'골든크로스':15s} {ret_gc*100:11.2f}% {mdd_gc*100:9.2f}% {sharpe_gc:8.2f}")
    print(f"{'골든크로스+RSI':15s} {ret_rsi*100:11.2f}% {mdd_rsi*100:9.2f}% {sharpe_rsi:8.2f}")

    return {
        'label': label,
        'df': df,
        'market_return': ret_market, 'mdd_market': mdd_market, 'sharpe_market': sharpe_market,
        'gc_return': ret_gc, 'mdd_gc': mdd_gc, 'sharpe_gc': sharpe_gc,
        'rsi_return': ret_rsi, 'mdd_rsi': mdd_rsi, 'sharpe_rsi': sharpe_rsi,
    }


# ------------------------------
# 7. 실행: 여러 기간 비교
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
    print(f"\n{'='*60}")
    print("요약 비교")
    print(f"{'='*60}")

    rows = [
        '최종수익률(단순보유)', '최종수익률(골든크로스)', '최종수익률(골든크로스+RSI)',
        'MDD(단순보유)', 'MDD(골든크로스)', 'MDD(골든크로스+RSI)',
        '샤프비율(단순보유)', '샤프비율(골든크로스)', '샤프비율(골든크로스+RSI)',
    ]

    summary_data = {'구분': rows}
    for r in results:
        summary_data[r['label']] = [
            f"{r['market_return']*100:.2f}%",
            f"{r['gc_return']*100:.2f}%",
            f"{r['rsi_return']*100:.2f}%",
            f"{r['mdd_market']*100:.2f}%",
            f"{r['mdd_gc']*100:.2f}%",
            f"{r['mdd_rsi']*100:.2f}%",
            f"{r['sharpe_market']:.2f}",
            f"{r['sharpe_gc']:.2f}",
            f"{r['sharpe_rsi']:.2f}",
        ]
    summary = pd.DataFrame(summary_data)
    print(summary.to_string(index=False))

    # 시각화
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        d = result['df']
        ax.plot(d.index, d['Cumulative_Market_Return'], label='Buy & Hold')
        ax.plot(d.index, d['Cumulative_Strategy'], label='Golden Cross')
        ax.plot(d.index, d['Cumulative_Strategy_RSI'], label='Golden Cross + RSI')
        ax.set_title(f"NVDA: {result['label']}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

'''
코카콜라와 같이 변동성이 작은 종목은 추세추종/모멘텀 계열 지표가 잘 작동하기 힘듦.
따라서 이런 종목에 대해서는 평균 회귀에 베팅하는 전략이 적절함.
'''