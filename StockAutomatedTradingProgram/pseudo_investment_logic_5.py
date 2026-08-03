"""
백테스팅: 골든크로스 / 골든크로스+RSI / 볼린저밴드 평균회귀 비교
+ 샤프비율 분석 + 기간별 비교 (기본 종목: 코카콜라 KO)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKER = "KO"  # 코카콜라 - 저변동성/박스권 성격의 종목
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  # 무위험수익률. 필요하면 연 3~4% 정도로 바꿔서 비교해도 됨


# ------------------------------
# 1. 데이터 다운로드
#    - buffer_days: 지표 워밍업(특히 150일 장기추세 이평선)을 위해
#      분석 시작일보다 앞서서 추가로 가져오는 과거 데이터 일수
# ------------------------------
def fetch_data(ticker, start=None, end=None, period=None, buffer_days=250):
    if period:
        # period 지정 시에는(예: "1y") 지표 워밍업을 위해 더 길게 받아온 뒤
        # analyze_period에서 필요한 만큼만 뒤에서 잘라 씀
        fetch_period = "2y" if period == "1y" else period
        df = yf.download(ticker, period=fetch_period, interval="1d")
    else:
        buffered_start = (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=buffered_start, end=end, interval="1d")
    df.columns = df.columns.get_level_values(0)  # 멀티인덱스 제거
    return df


# ------------------------------
# 2. 지표 계산 (이평선, 거래량 필터, RSI, 볼린저밴드)
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


def add_indicators(df, ma_short=20, ma_long=60, volume_window=3, rsi_period=14,
                    bb_window=20, bb_std=2, trend_ma_window=150):
    df = df.copy()

    # 이동평균 / 골든크로스
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
        .rolling(window=2 * volume_window + 1, center=False, min_periods=1)
        .max()
        .astype(bool)
    )

    # RSI
    df['RSI'] = calc_rsi(df['Close'], period=rsi_period)

    # 볼린저 밴드
    df['BB_Mid'] = df['Close'].rolling(window=bb_window).mean()
    bb_std_val = df['Close'].rolling(window=bb_window).std()
    df['BB_Upper'] = df['BB_Mid'] + bb_std * bb_std_val
    df['BB_Lower'] = df['BB_Mid'] - bb_std * bb_std_val

    # 장기 추세 필터용 이동평균 (기본 150일) - 볼린저 평균회귀가
    # 진짜 하락추세에서 저가매수를 시도하는 것을 막기 위함
    df['MA_trend'] = df['Close'].rolling(window=trend_ma_window).mean()

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
#    - 진입 신호 ②: 국면(Signal==1)이 켜져 있는 동안 RSI가 과매도에서 반등할 때
#      (추가 진입). use_trend_filter=True면 종가가 장기추세선(MA_trend) 위에
#      있을 때만 이 추가 진입을 허용 — 국면이 막 켜졌지만 아직 장기추세로는
#      불안정한 구간(예: 급락 초입의 일시적 반등)에서 성급하게 진입하는 걸 방지
# ------------------------------
def run_golden_cross_rsi(df, rsi_oversold=35, rsi_overbought=70,
                          exit_on_overbought=False, use_trend_filter=True):
    df = df.copy()

    rsi_prev = df['RSI'].shift(1)
    df['RSI_Bounce'] = (rsi_prev <= rsi_oversold) & (df['RSI'] > rsi_oversold)

    position = 0
    filtered_signal = []

    for i in range(len(df)):
        regime_on = df['Signal'].iloc[i] == 1
        golden_cross_entry = df['Golden_Cross'].iloc[i] and df['Volume_Confirm_Nearby'].iloc[i]

        trend_ma = df['MA_trend'].iloc[i]
        close = df['Close'].iloc[i]
        trend_ok = (not use_trend_filter) or (pd.notna(trend_ma) and close > trend_ma)
        rsi_bounce_entry = regime_on and df['RSI_Bounce'].iloc[i] and trend_ok

        if position == 0:
            if golden_cross_entry or rsi_bounce_entry:
                position = 1
        else:
            if df['Death_Cross'].iloc[i]:
                position = 0
            elif exit_on_overbought and df['RSI'].iloc[i] >= rsi_overbought:
                position = 0

        filtered_signal.append(position)

    df['Filtered_Signal_RSI'] = filtered_signal
    df['Position_RSI'] = df['Filtered_Signal_RSI'].shift(1)
    df['Strategy_RSI_Return'] = df['Daily_Return'] * df['Position_RSI']
    return df


# ------------------------------
# 5. 전략 3: 볼린저 밴드 평균회귀 (박스권/저변동성 종목용)
#    - 진입: 종가가 하단 밴드 이하로 내려갈 때 (과매도로 판단, 저가 매수)
#      단, use_trend_filter=True면 종가가 장기추세선(MA_trend) 위에 있을 때만 진입 허용
#      → "진짜 하락추세 중의 반등 시도"를 걸러내기 위함
#    - 청산 ①: 종가가 중심선(20일 이평) 이상으로 회복할 때 (평균으로 되돌아옴)
#    - 청산 ②(손절): 진입가 대비 stop_loss_pct 이상 추가 하락하면 손절
#    - exit_at == 'upper'로 바꾸면 상단 밴드까지 들고 가는 공격적 버전도 가능
# ------------------------------
def run_bollinger_reversion(df, exit_at='mid', stop_loss_pct=0.05, use_trend_filter=True):
    df = df.copy()

    position = 0
    entry_price = None
    filtered_signal = []
    stop_loss_hit = []

    for i in range(len(df)):
        close = df['Close'].iloc[i]
        lower = df['BB_Lower'].iloc[i]
        mid = df['BB_Mid'].iloc[i]
        upper = df['BB_Upper'].iloc[i]
        trend_ma = df['MA_trend'].iloc[i]

        hit_stop = False

        if position == 0:
            trend_ok = (not use_trend_filter) or (pd.notna(trend_ma) and close > trend_ma)
            if pd.notna(lower) and close <= lower and trend_ok:
                position = 1
                entry_price = close
        else:
            exit_level = mid if exit_at == 'mid' else upper

            # 손절 조건을 먼저 확인 (평균회귀를 기다리다 손실이 더 커지는 걸 방지)
            if stop_loss_pct is not None and close <= entry_price * (1 - stop_loss_pct):
                position = 0
                entry_price = None
                hit_stop = True
            elif pd.notna(exit_level) and close >= exit_level:
                position = 0
                entry_price = None

        filtered_signal.append(position)
        stop_loss_hit.append(hit_stop)

    df['Filtered_Signal_BB'] = filtered_signal
    df['Stop_Loss_Hit'] = stop_loss_hit
    df['Position_BB'] = df['Filtered_Signal_BB'].shift(1)
    df['Strategy_BB_Return'] = df['Daily_Return'] * df['Position_BB']
    return df


# ------------------------------
# 6. 샤프비율 & MDD
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
# 7. 한 기간에 대한 4개 전략 분석
# ------------------------------
def analyze_period(label, ticker=TICKER, start=None, end=None, period=None):
    print(f"\n{'='*65}")
    print(f"[{label}] 분석 시작")
    print(f"{'='*65}")

    raw_df = fetch_data(ticker, start=start, end=end, period=period)
    df = add_indicators(raw_df)

    # 지표 워밍업용으로 여유있게 받아온 부분을 잘라내고, 실제 분석 구간만 남김
    if period == "1y":
        df = df.tail(252).copy()  # 최근 약 1년(거래일 기준)만 남김
    elif start is not None:
        df = df[df.index >= pd.to_datetime(start)].copy()

    df = run_golden_cross_only(df)
    df = run_golden_cross_rsi(df)
    df = run_bollinger_reversion(df)

    cum_market, ret_market, mdd_market, sharpe_market = summarize(df['Daily_Return'])
    cum_gc, ret_gc, mdd_gc, sharpe_gc = summarize(df['Strategy_Return'])
    cum_rsi, ret_rsi, mdd_rsi, sharpe_rsi = summarize(df['Strategy_RSI_Return'])
    cum_bb, ret_bb, mdd_bb, sharpe_bb = summarize(df['Strategy_BB_Return'])

    df['Cumulative_Market_Return'] = cum_market
    df['Cumulative_Strategy'] = cum_gc
    df['Cumulative_Strategy_RSI'] = cum_rsi
    df['Cumulative_Strategy_BB'] = cum_bb

    print(f"{'전략':18s} {'최종수익률':>12s} {'MDD':>10s} {'샤프비율':>8s}")
    print(f"{'단순보유':18s} {ret_market*100:11.2f}% {mdd_market*100:9.2f}% {sharpe_market:8.2f}")
    print(f"{'골든크로스':18s} {ret_gc*100:11.2f}% {mdd_gc*100:9.2f}% {sharpe_gc:8.2f}")
    print(f"{'골든크로스+RSI':18s} {ret_rsi*100:11.2f}% {mdd_rsi*100:9.2f}% {sharpe_rsi:8.2f}")
    print(f"{'볼린저밴드 평균회귀':18s} {ret_bb*100:11.2f}% {mdd_bb*100:9.2f}% {sharpe_bb:8.2f}")

    return {
        'label': label,
        'df': df,
        'market_return': ret_market, 'mdd_market': mdd_market, 'sharpe_market': sharpe_market,
        'gc_return': ret_gc, 'mdd_gc': mdd_gc, 'sharpe_gc': sharpe_gc,
        'rsi_return': ret_rsi, 'mdd_rsi': mdd_rsi, 'sharpe_rsi': sharpe_rsi,
        'bb_return': ret_bb, 'mdd_bb': mdd_bb, 'sharpe_bb': sharpe_bb,
    }


# ------------------------------
# 8. 실행: 여러 기간 비교
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
    print(f"\n{'='*65}")
    print("요약 비교")
    print(f"{'='*65}")

    rows = [
        '최종수익률(단순보유)', '최종수익률(골든크로스)', '최종수익률(골든크로스+RSI)', '최종수익률(볼린저밴드)',
        'MDD(단순보유)', 'MDD(골든크로스)', 'MDD(골든크로스+RSI)', 'MDD(볼린저밴드)',
        '샤프비율(단순보유)', '샤프비율(골든크로스)', '샤프비율(골든크로스+RSI)', '샤프비율(볼린저밴드)',
    ]

    summary_data = {'구분': rows}
    for r in results:
        summary_data[r['label']] = [
            f"{r['market_return']*100:.2f}%",
            f"{r['gc_return']*100:.2f}%",
            f"{r['rsi_return']*100:.2f}%",
            f"{r['bb_return']*100:.2f}%",
            f"{r['mdd_market']*100:.2f}%",
            f"{r['mdd_gc']*100:.2f}%",
            f"{r['mdd_rsi']*100:.2f}%",
            f"{r['mdd_bb']*100:.2f}%",
            f"{r['sharpe_market']:.2f}",
            f"{r['sharpe_gc']:.2f}",
            f"{r['sharpe_rsi']:.2f}",
            f"{r['sharpe_bb']:.2f}",
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
        ax.plot(d.index, d['Cumulative_Strategy_BB'], label='Bollinger Reversion')
        ax.set_title(f"{TICKER}: {result['label']}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()
