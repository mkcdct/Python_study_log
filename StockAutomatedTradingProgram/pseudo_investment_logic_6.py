'''
logic_5와의 차이

이동평균선 단축: 코카콜라처럼 추세 전환이 완만한 종목은 기존 20/60일선이 너무 느리게 반응하므로, 10일/40일선으로 단축하여 추세 변화를 더 빠르게 포착하도록 조정했습니다.

볼린저 밴드 표준편차 완화: 기본 표준편차 배수를 2.0에서 1.8로 낮춰 박스권 안에서 밴드 하단 터치 기회를 소폭 늘려 수익 기회를 개선했습니다.

RSI 기준 조정: 과매도 기준을 기존 35에서 40으로 상향하여 보수적인 진입 타이밍을 조금 더 유연하게 가져갔습니다
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf  # yfinance 임포트 위치 수정

TICKER = "KO"  # 코카콜라 - 저변동성/박스권 성격의 종목
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.03  # 최근 금리 환경을 반영하여 무위험수익률 3% 설정


# ------------------------------
# 1. 데이터 다운로드
# ------------------------------
def fetch_data(ticker, start=None, end=None, period=None, buffer_days=250):
    if period:
        fetch_period = "3y" if period == "1y" else period
        df = yf.download(ticker, period=fetch_period, interval="1d", progress=False)
    else:
        buffered_start = (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=buffered_start, end=end, interval="1d", progress=False)
    
    # 멀티인덱스 컬럼 안전하게 제거
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ------------------------------
# 2. 지표 계산 (코카콜라 맞춤형 파라미터 적용)
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


def add_indicators(df, ma_short=10, ma_long=40, volume_window=3, rsi_period=14,
                   bb_window=20, bb_std=1.8, trend_ma_window=100):
    df = df.copy()

    # 코카콜라는 변동성이 낮고 추세 전환이 느리므로 단기/장기 이평선 기간을 축소하여 반응 속도 개선
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
        .rolling(window=2 * volume_window + 1, center=False, min_periods=1)
        .max()
        .astype(bool)
    )

    df['RSI'] = calc_rsi(df['Close'], period=rsi_period)

    # 볼린저 밴드: 저변동성 종목 특성을 감안해 std를 2.0 -> 1.8로 낮추어 밴드 터치 빈도와 기회 확보
    df['BB_Mid'] = df['Close'].rolling(window=bb_window).mean()
    bb_std_val = df['Close'].rolling(window=bb_window).std()
    df['BB_Upper'] = df['BB_Mid'] + bb_std * bb_std_val
    df['BB_Lower'] = df['BB_Mid'] - bb_std * bb_std_val

    df['MA_trend'] = df['Close'].rolling(window=trend_ma_window).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    return df


# ------------------------------
# 3. 전략 1: 골든크로스 단독
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
# 4. 전략 2: 골든크로스 + RSI 눌림목
# ------------------------------
def run_golden_cross_rsi(df, rsi_oversold=40, rsi_overbought=70,
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
# 5. 전략 3: 볼린저 밴드 평균회귀 (룩어헤드 바이어스 방지: 익일 시가 진입 반영)
# ------------------------------
def run_bollinger_reversion(df, exit_at='mid', stop_loss_pct=0.04, use_trend_filter=True):
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
            # 당일 종가 조건 만족 시 다음날부터 포지션 유지하도록 설계 (Look-ahead 방지 관점)
            if pd.notna(lower) and close <= lower and trend_ok:
                position = 1
                entry_price = close
        else:
            exit_level = mid if exit_at == 'mid' else upper

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
    # 시프트 적용하여 당일 시그널로 익일 매매 반영
    df['Position_BB'] = df['Filtered_Signal_BB'].shift(1)
    df['Strategy_BB_Return'] = df['Daily_Return'] * df['Position_BB']
    return df


# ------------------------------
# 6. 샤프비율 & MDD 계산
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
# 7. 기간별 분석 함수
# ------------------------------
def analyze_period(label, ticker=TICKER, start=None, end=None, period=None):
    print(f"\n{'='*65}")
    print(f"[{label}] 분석 시작")
    print(f"{'='*65}")

    raw_df = fetch_data(ticker, start=start, end=end, period=period)
    df = add_indicators(raw_df)

    if period == "1y":
        df = df.tail(252).copy()
    elif start is not None:
        df = df[df.index >= pd.to_datetime(start)].copy()
        if end is not None:
            df = df[df.index <= pd.to_datetime(end)].copy()

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
# 8. 실행: 2020년부터 2026년까지 연도별 확장 검증
# ------------------------------
PERIODS = [
    {'label': '2020년', 'start': '2020-01-01', 'end': '2020-12-31'},
    {'label': '2021년', 'start': '2021-01-01', 'end': '2021-12-31'},
    {'label': '2022년', 'start': '2022-01-01', 'end': '2022-12-31'},
    {'label': '2023년', 'start': '2023-01-01', 'end': '2023-12-31'},
    {'label': '2024년', 'start': '2024-01-01', 'end': '2024-12-31'},
    {'label': '2025년', 'start': '2025-01-01', 'end': '2025-12-31'},
    {'label': '2026년(현재)', 'start': '2026-01-01', 'end': '2026-12-31'},
]

if __name__ == "__main__":
    results = [analyze_period(**p) for p in PERIODS]

    # 요약 비교 표 출력
    print(f"\n{'='*65}")
    print("연도별 요약 비교 (2020 ~ 2026)")
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

    # 시각화 (서브플롯 행렬 구성)
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        d = result['df']
        if not d.empty:
            ax.plot(d.index, d['Cumulative_Market_Return'], label='Buy & Hold')
            ax.plot(d.index, d['Cumulative_Strategy'], label='Golden Cross')
            ax.plot(d.index, d['Cumulative_Strategy_RSI'], label='Golden Cross + RSI')
            ax.plot(d.index, d['Cumulative_Strategy_BB'], label='Bollinger Reversion')
            ax.set_title(f"{TICKER}: {result['label']}")
            ax.set_ylabel('Cumulative Return')
            ax.legend(loc='upper left')
            ax.grid(True)
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()