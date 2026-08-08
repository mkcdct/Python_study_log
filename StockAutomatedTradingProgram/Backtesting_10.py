'''
하락 추세 종목을 파이썬으로 발굴하여
해당 종목을 대상으로 전략을 검증.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

START_DATE = "2023-01-01"
END_DATE = "2026-12-31"

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.03


# ------------------------------
# 1. 안정적이고 넓은 대형주 유니버스 풀 설정 (에러 원천 차단)
# ------------------------------
def get_extended_universe():
    print("[*] 확장된 미국 대형주 및 섹터 대표 유니버스 로드 중...")
    universe = [
        "INTC", "BA", "PFE", "NKE", "PYPL", "DIS", "BAX", "MMM", "CVS", "UPS",
        "VZ", "T", "JNJ", "MDT", "XOM", "CVX", "AAPL", "MSFT", "AMZN", "GOOGL",
        "META", "TSLA", "NVDA", "AMD", "QCOM", "IBM", "ORCL", "NFLX", "ADBE",
        "CRM", "WMT", "HD", "MCD", "KO", "PEP", "COST", "SBUX", "GM", "F", "CAT",
        "GE", "HON", "UNH", "LLY", "ABBV", "MRK", "TMO", "DHR", "NEE", "DUK",
        "SO", "BMY", "AMGN", "GILD", "SBUX", "TGT", "LOW", "BKNG", "AXP", "V"
    ]
    return list(set(universe))


# ------------------------------
# 2. 안전한 데이터 다운로드
# ------------------------------
def fetch_data(ticker, start, end, buffer_days=250):
    buffered_start = (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=buffered_start, end=end, interval="1d", progress=False, multi_level_index=False)
        if df.empty or len(df) < 200:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


# ------------------------------
# 3. 대폭 완화된 스크리닝 조건 적용
# ------------------------------
def scan_qualified_downtrend_stocks(universe, start, end):
    qualified_list = []
    print(f"[*] 총 {len(universe)}개 종목 대상 완화된 조건으로 스크리닝 중...")
    
    for idx, ticker in enumerate(universe):
        df = fetch_data(ticker, start, end)
        if df is None:
            continue
            
        df['Trend_MA'] = df['Close'].rolling(window=200).mean()
        df_target = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))].copy()
        
        if len(df_target) < 100:
            continue
            
        below_ma_ratio = (df_target['Close'] < df_target['Trend_MA']).mean()
        total_return = (df_target['Close'].iloc[-1] / df_target['Close'].iloc[0]) - 1
        
        # [조건 완화] 이평선 하단 30% 이상, 수익률 -60% ~ +25% 사이
        if below_ma_ratio >= 0.30 and -0.60 <= total_return < 0.25:
            df_target['Rolling_Low'] = df_target['Close'].rolling(window=20).min()
            is_local_low = df_target['Close'] == df_target['Rolling_Low']
            
            low_indices = np.where(is_local_low)[0]
            if len(low_indices) < 3:
                continue
                
            bounce_success_count = 0
            valid_lows_count = 0
            
            for l_idx in low_indices:
                if l_idx + 5 < len(df_target):
                    valid_lows_count += 1
                    low_price = df_target['Close'].iloc[l_idx]
                    future_max = df_target['Close'].iloc[l_idx+1 : l_idx+6].max()
                    if (future_max / low_price) - 1 >= 0.035:  # 3.5% 이상 반등으로 완화
                        bounce_success_count += 1
            
            if valid_lows_count > 0:
                bounce_ratio = bounce_success_count / valid_lows_count
                if bounce_ratio >= 0.25:  # 반등 성공률 25% 이상으로 완화
                    qualified_list.append(ticker)
                    print(f"  [발굴 성공] {ticker} (3년수익률: {total_return*100:.1f}%, 반등성공률: {bounce_ratio*100:.1f}%)")
                    
    return qualified_list


# ------------------------------
# 4. 기술적 지표 및 전략 함수
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


def calc_mdd(cumulative_series):
    peak = cumulative_series.cummax()
    drawdown = (cumulative_series - peak) / peak
    return drawdown.min()


def summarize(returns_series):
    cumulative = (1 + returns_series.fillna(0)).cumprod()
    final_return = cumulative.iloc[-1] - 1
    mdd = calc_mdd(cumulative)
    return cumulative, final_return, mdd


# ------------------------------
# 5. 메인 실행부
# ------------------------------
if __name__ == "__main__":
    universe = get_extended_universe()
    target_tickers = scan_qualified_downtrend_stocks(universe, START_DATE, END_DATE)
    
    # 발굴된 종목 중 최대 5개만 샘플 테스트
    target_tickers = target_tickers[:5]
    
    if not target_tickers:
        print("\n조건에 부합하는 종목이 없습니다.")
    else:
        print(f"\n[발굴 완료] 대표 발굴 종목 {len(target_tickers)}개 대상 백테스팅 시작\n")
        all_results = []
        
        for ticker in target_tickers:
            raw_df = fetch_data(ticker, start=START_DATE, end=END_DATE)
            if raw_df is None:
                continue
            df = add_indicators(raw_df)
            df = df[(df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))].copy()
            
            df = run_downtrend_bounce_strategy(df)
            
            _, ret_market, mdd_market = summarize(df['Daily_Return'])
            _, ret_strat, mdd_strat = summarize(df['Strategy_Return'])
            
            all_results.append({
                'ticker': ticker, 'df': df,
                'ret_market': ret_market, 'mdd_market': mdd_market,
                'ret_strat': ret_strat, 'mdd_strat': mdd_strat
            })

        print(f"{'='*65}")
        print(f"완화된 조건 발굴 종목 반등 전략 성과 ({START_DATE[:4]} ~ {END_DATE[:4]})")
        print(f"{'='*65}")
        print(f"{'종목':6s} | {'전략 구분':15s} | {'최종수익률':10s} | {'MDD':8s}")
        print("-" * 50)
        
        for r in all_results:
            t = r['ticker']
            print(f"{t:6s} | {'단순보유 (Buy&Hold)':15s} | {r['ret_market']*100:9.2f}% | {r['mdd_market']*100:7.2f}%")
            print(f"{t:6s} | {'하락추세 반등전략':15s} | {r['ret_strat']*100:9.2f}% | {r['mdd_strat']*100:7.2f}%")
            print("-" * 50)

        # 시각화
        fig, axes = plt.subplots(len(all_results), 1, figsize=(10, 4 * len(all_results)))
        if len(all_results) == 1:
            axes = [axes]

        for ax, r in zip(axes, all_results):
            d = r['df']
            t = r['ticker']
            ax.plot(d.index, (1 + d['Daily_Return'].fillna(0)).cumprod(), label=f'{t} Buy & Hold', color='gray', linestyle='--')
            ax.plot(d.index, (1 + d['Strategy_Return'].fillna(0)).cumprod(), label=f'{t} Bounce Strategy', color='royalblue')
            ax.set_title(f"Relaxed Condition Strategy: [{t}]")
            ax.set_ylabel('Cumulative Return')
            ax.legend(loc='upper left')
            ax.grid(True)

        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()

        # 주가가 횡보하거나 완만한 상승세일 때 불필요한 매매를 줄이기 위해, RSI 침체 구간이나 볼린저밴드 하단 이탈 조건에 '거래량 폭증'이나 '이동평균선 기울기' 같은 필터를 추가하면 완성도가 더욱 높아질 수 있습니다.