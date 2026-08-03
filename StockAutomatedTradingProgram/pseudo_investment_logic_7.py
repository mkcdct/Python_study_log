'''
사회적 노이즈에 크게 영향을 받지 않고 박스권 내에서 주가가 움직이는 종목을 대상으로
경기방어, 박스권 내 변동성/평균 회귀 투자 철학을 반영하여
기존 코드를 '프록터 앤 갬블(PG), 필수소비재 ETF(XLP)까지 확장하고
ATR 변동성 필터와 시간 제한 청산 로직을 추가함.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# 복수 종목 설정 (코카콜라, 프록터 앤 갬블, 필수소비재 ETF)
TICKERS = ["KO", "PG", "XLP"]  
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.03  # 무위험수익률 3% 반영


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
# 2. 박스권 및 변동성 지표 계산 (ATR, 볼린저 밴드, ATR 필터)
# ------------------------------
def calc_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def add_box_indicators(df, bb_window=20, bb_std=1.8, atr_period=14, trend_ma_window=100):
    df = df.copy()

    # 볼린저 밴드 (박스권 중심 및 상하단 기준)
    df['BB_Mid'] = df['Close'].rolling(window=bb_window).mean()
    bb_std_val = df['Close'].rolling(window=bb_window).std()
    df['BB_Upper'] = df['BB_Mid'] + bb_std * bb_std_val
    df['BB_Lower'] = df['BB_Mid'] - bb_std * bb_std_val

    # ATR (변동성 필터용) 및 ATR 이평선 (변동성 정체/폭발 국면 확인)
    df['ATR'] = calc_atr(df, period=atr_period)
    df['ATR_MA'] = df['ATR'].rolling(window=20).mean()

    # 장기 추세선 (대세 하락장 필터링용)
    df['MA_trend'] = df['Close'].rolling(window=trend_ma_window).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    return df


# ------------------------------
# 3. 박스권 변동성 회귀 전략 (ATR 필터 + 손절 + 시간 제한 청산)
# ------------------------------
def run_box_reversion_strategy(df, stop_loss_pct=0.04, max_holding_days=20, use_trend_filter=True):
    df = df.copy()

    position = 0
    entry_price = None
    holding_days = 0
    filtered_signal = []
    stop_loss_hit = []

    for i in range(len(df)):
        close = df['Close'].iloc[i]
        lower = df['BB_Lower'].iloc[i]
        mid = df['BB_Mid'].iloc[i]
        trend_ma = df['MA_trend'].iloc[i]
        atr = df['ATR'].iloc[i]
        atr_ma = df['ATR_MA'].iloc[i]

        hit_stop = False

        if position == 0:
            # 조건 1: 대세 상승/완만 국면 (종가가 장기 이평선 위)
            trend_ok = (not use_trend_filter) or (pd.notna(trend_ma) and close > trend_ma)
            
            # 조건 2: 변동성 필터 (ATR이 너무 죽어있지 않고, 폭발하지도 않는 적정 박스권 변동성 구간)
            volatility_ok = pd.notna(atr) and pd.notna(atr_ma) and (atr >= atr_ma * 0.7)
            
            # 진입 시그널: 볼린저 밴드 하단 이탈 + 필터 만족
            if pd.notna(lower) and close <= lower and trend_ok and volatility_ok:
                position = 1
                entry_price = close
                holding_days = 0
        else:
            holding_days += 1

            # 청산 조건 ①: 고정 손절 비율 도달
            if stop_loss_pct is not None and close <= entry_price * (1 - stop_loss_pct):
                position = 0
                entry_price = None
                holding_days = 0
                hit_stop = True
            
            # 청산 조건 ②: 시간 제한 청산 (기회비용 절감 - 일정 기간 내 중심선 회복 못하면 탈출)
            elif holding_days >= max_holding_days:
                position = 0
                entry_price = None
                holding_days = 0

            # 청산 조건 ③: 평균 회귀 완료 (중심선 도달 시 익절)
            elif pd.notna(mid) and close >= mid:
                position = 0
                entry_price = None
                holding_days = 0

        filtered_signal.append(position)
        stop_loss_hit.append(hit_stop)

    df['Filtered_Signal'] = filtered_signal
    df['Stop_Loss_Hit'] = stop_loss_hit
    df['Position'] = df['Filtered_Signal'].shift(1)  # 룩어헤드 방지 (익일 시가/종가 매매 반영)
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    return df


# ------------------------------
# 4. 샤프비율 및 MDD 계산 함수
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
# 5. 종목별 백테스팅 실행 및 비교
# ------------------------------
def backtest_ticker(ticker, start_date, end_date):
    raw_df = fetch_data(ticker, start=start_date, end=end_date)
    df = add_box_indicators(raw_df)
    
    # 분석 구간 자르기
    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))].copy()
    
    df = run_box_reversion_strategy(df)
    
    cum_market, ret_market, mdd_market, sharpe_market = summarize(df['Daily_Return'])
    cum_strat, ret_strat, mdd_strat, sharpe_strat = summarize(df['Strategy_Return'])
    
    df['Cumulative_Market'] = cum_market
    df['Cumulative_Strategy'] = cum_strat
    
    return {
        'ticker': ticker,
        'df': df,
        'market_return': ret_market, 'mdd_market': mdd_market, 'sharpe_market': sharpe_market,
        'strat_return': ret_strat, 'mdd_strat': mdd_strat, 'sharpe_strat': sharpe_strat,
    }


if __name__ == "__main__":
    START_DATE = "2020-01-01"
    END_DATE = "2026-12-31"  # 또는 현재일 기준
    
    all_results = []
    for ticker in TICKERS:
        print(f"[{ticker}] 백테스팅 진행 중 ({START_DATE} ~ {END_DATE})...")
        res = backtest_ticker(ticker, START_DATE, END_DATE)
        all_results.append(res)

    # ------------------------------
    # 6. 결과 요약 출력
    # ------------------------------
    print(f"\n{'='*65}")
    print(f"종목별 박스권 전략 vs 단순보유 성과 비교 ({START_DATE[:4]} ~ {END_DATE[:4]})")
    print(f"{'='*65}")
    print(f"{'종목':6s} | {'구분':10s} | {'최종수익률':10s} | {'MDD':8s} | {'샤프비율':8s}")
    print("-" * 55)
    
    for r in all_results:
        t = r['ticker']
        print(f"{t:6s} | {'단순보유':10s} | {r['market_return']*100:9.2f}% | {r['mdd_market']*100:7.2f}% | {r['sharpe_market']:8.2f}")
        print(f"{t:6s} | {'박스권전략':10s} | {r['strat_return']*100:9.2f}% | {r['mdd_strat']*100:7.2f}% | {r['sharpe_strat']:8.2f}")
        print("-" * 55)

    # ------------------------------
    # 7. 시각화
    # ------------------------------
    fig, axes = plt.subplots(len(TICKERS), 1, figsize=(10, 4 * len(TICKERS)))
    if len(TICKERS) == 1:
        axes = [axes]

    for ax, r in zip(axes, all_results):
        d = r['df']
        t = r['ticker']
        ax.plot(d.index, d['Cumulative_Market'], label=f'{t} Buy & Hold', color='gray', linestyle='--')
        ax.plot(d.index, d['Cumulative_Strategy'], label=f'{t} Box Reversion Strategy', color='royalblue')
        ax.set_title(f"{t} Strategy Performance ({START_DATE[:4]} - {END_DATE[:4]})")
        ax.set_ylabel('Cumulative Return')
        ax.legend(loc='upper left')
        ax.grid(True)

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()


# insight : 수익률을 극대화하려면 단순보유가 정답이지만, 
# 하락장에서 계좌가 녹는 꼴을 절대 못 보겠고, 심리적 안정감을 최우선으로 하겠다"면 박스권 변동성 제어 전략이 훌륭한 방패가 될 수 있음.
# 즉, 박스권 내 움직임이 주가 되는 종목은 단순보유전략이 우월함.
