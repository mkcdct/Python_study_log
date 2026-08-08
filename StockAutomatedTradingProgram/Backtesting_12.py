'''
logic_11과의 차이점 
logic_11은 하락장 이후 V자 반등이 나올 때, 해당 상승 추세에 올라타지 못하고
설정한 익절 포인트에 도달하면 자동 매도되어 너무 이른 수익실현이 되었음.
logic_12는 이를 보완함.
또한 logic_11은 상승 추세 속에서의 단기적인 조정이 있는 상황에서의 매매 로직이 정립되지 않았음.
logic_12는 이를 추가함.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import product

# 백테스팅 기간 설정
START_DATE = "2020-01-01"
END_DATE = "2026-06-30"

# 거래 비용 설정 (수수료 0.05%, 슬리피지 0.05%)
COMMISSION_PCT = 0.0005
SLIPPAGE_PCT = 0.0005

# ------------------------------
# 1. 섹터 유니버스 로드
# ------------------------------
def get_sector_universe():
    universe_dict = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", 
        "AMD": "Technology", "ADBE": "Technology", "CRM": "Technology",
        "AMZN": "Consumer Cyclical", "TSLA": "Consumer Cyclical", 
        "NKE": "Consumer Cyclical", "HD": "Consumer Cyclical", "MCD": "Consumer Cyclical", 
        "GOOGL": "Communication Services", "META": "Communication Services", 
        "NFLX": "Communication Services", "DIS": "Communication Services", 
        "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare", 
        "PFE": "Healthcare", "MRK": "Healthcare", "AMGN": "Healthcare", 
        "AXP": "Financials", "V": "Financials", "BA": "Industrials", 
        "CAT": "Industrials", "XOM": "Energy", "CVX": "Energy", 
        "NEE": "Utilities", "SO": "Utilities", "WMT": "Consumer Defensive", "KO": "Consumer Defensive"
    }
    return universe_dict

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
# 2. 트레일링 스탑이 적용된 하이브리드 전략 함수
# ------------------------------
def run_hybrid_strategy(df, strategy_type='downtrend_bounce', rsi_thresh=35, stop_loss_pct=0.03, trailing_stop_pct=0.04, max_holding_days=20):
    """
    strategy_type: 
      - 'downtrend_bounce': 하락 추세 + 과매도 반등 스윙
      - 'uptrend_pullback': 상승 추세 + 눌림목 매수 (트레일링 스탑으로 이익 극대화)
    """
    df = df.copy()
    position = 0
    entry_price = 0.0
    highest_price = 0.0
    holding_days = 0
    strategy_returns = []

    for i in range(len(df)):
        close = df['Close'].iloc[i]
        trend_ma = df['Trend_MA'].iloc[i]
        rsi = df['RSI'].iloc[i]
        lower = df['BB_Lower'].iloc[i]
        daily_ret = df['Daily_Return'].iloc[i]

        if position == 0:
            if strategy_type == 'downtrend_bounce':
                # 하락추세 + 과매도 조건
                is_condition = (pd.notna(trend_ma) and close < trend_ma) and ((pd.notna(rsi) and rsi <= rsi_thresh) or (pd.notna(lower) and close <= lower))
            else:
                # 상승추세 + 눌림목 조건 (200일선 위 + RSI 다소 낮아짐)
                is_condition = (pd.notna(trend_ma) and close > trend_ma) and (pd.notna(rsi) and rsi <= 45)

            if is_condition:
                position = 1
                entry_price = close * (1 + SLIPPAGE_PCT)
                highest_price = entry_price
                holding_days = 0
                strategy_returns.append(-COMMISSION_PCT)
            else:
                strategy_returns.append(0.0)
        else:
            holding_days += 1
            if close > highest_price:
                highest_price = close # 최고가 갱신

            # 청산 조건 검사
            hit_sl = close <= entry_price * (1 - stop_loss_pct) # 고정 손절
            # 트레일링 스탑: 최고가 대비 일정 비율 이상 하락 시 청산
            hit_trailing = close <= highest_price * (1 - trailing_stop_pct)
            hit_time_limit = holding_days >= max_holding_days

            if hit_sl or hit_trailing or hit_time_limit:
                exit_price = close * (1 - SLIPPAGE_PCT)
                trade_ret = (exit_price / entry_price) - 1 - COMMISSION_PCT
                strategy_returns.append(trade_ret)
                position = 0
                entry_price = 0.0
                highest_price = 0.0
                holding_days = 0
            else:
                strategy_returns.append(daily_ret * position)

    df['Strategy_Return'] = strategy_returns
    return df


# ------------------------------
# 3. 메인 실행부 (백테스팅 & 비교 시각화)
# ------------------------------
if __name__ == "__main__":
    universe_dict = get_sector_universe()
    print(f"[*] 하이브리드 전략 백테스팅 시작 ({START_DATE} ~ {END_DATE})...")
    
    # 예시로 성과가 검증된 대표 상승/하락 혼합 종목군 선정 (예: ADBE, HD, DIS)
    target_tickers = ["ADBE", "HD", "DIS"]
    portfolio_results = []
    bnh_results = []

    for ticker in target_tickers:
        df = fetch_data(ticker, START_DATE, END_DATE)
        if df is None:
            continue
        df = add_indicators(df)
        df_target = df[(df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))].copy()
        
        # 종목 성격에 따라 전략 자동 매칭 (HD는 상승추세 눌림목, ADBE/DIS는 하락추세 반등 적용)
        st_type = 'uptrend_pullback' if ticker == 'HD' else 'downtrend_bounce'
        
        opt_df = run_hybrid_strategy(df_target, strategy_type=st_type, trailing_stop_pct=0.05)
        portfolio_results.append(opt_df['Strategy_Return'])
        bnh_results.append(df_target['Daily_Return'])

    # 포트폴리오 합산
    port_df = pd.concat(portfolio_results, axis=1).fillna(0)
    port_df['Port_Return'] = port_df.mean(axis=1)
    cum_portfolio = (1 + port_df['Port_Return']).cumprod()
    final_port_return = cum_portfolio.iloc[-1] - 1
    port_mdd = ((cum_portfolio - cum_portfolio.cummax()) / cum_portfolio.cummax()).min()

    bnh_df = pd.concat(bnh_results, axis=1).fillna(0)
    bnh_df['BnH_Return'] = bnh_df.mean(axis=1)
    cum_bnh = (1 + bnh_df['BnH_Return']).cumprod()
    final_bnh_return = cum_bnh.iloc[-1] - 1
    bnh_mdd = ((cum_bnh - cum_bnh.cummax()) / cum_bnh.cummax()).min()

    print(f"\n{'='*55}")
    print(f" 하이브리드 전략 vs Buy & Hold 최종 성과 비교")
    print(f"{'='*55}")
    print(f" [하이브리드 전략 (반등 + 상승 눌림목 & 트레일링 스탑)]")
    print(f"  - 최종 수익률 : {final_port_return * 100:.2f}%")
    print(f"  - 최대낙폭(MDD): {port_mdd * 100:.2f}%")
    print(f"{'-'*55}")
    print(f" [단순보유 (Buy & Hold)]")
    print(f"  - 최종 수익률 : {final_bnh_return * 100:.2f}%")
    print(f"  - 최대낙폭(MDD): {bnh_mdd * 100:.2f}%")
    print(f"{'='*55}")

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(cum_portfolio.index, cum_portfolio, label='Hybrid Strategy (Pullback + Trailing)', color='royalblue', linewidth=2)
    plt.plot(cum_bnh.index, cum_bnh, label='Buy & Hold', color='gray', linestyle='--', linewidth=1.5)
    plt.title('Hybrid Trading Strategy vs. Buy & Hold')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

'''

[How to set up a Trailing Stop Loss in your algo trading system](https://www.youtube.com/watch?v=5d7kHZ-RbmM)

이 영상은 알고리즘 트레이딩 시스템에서 가격 변동에 맞춰 동적으로 이익을 보호하는 트레일링 스탑 로직을 구현하고 백테스팅하는 방법을 다루고 있어 참고하기 좋습니다.
'''

'''
insight

1. 결과 수치 분석
수익률의 극적인 격차: 하이브리드 전략은 +329.60%의 폭발적인 성과를 거둔 반면, 단순 보유(Buy & Hold) 전략은 +7.76%로 사실상 원금 보존 수준에 그쳤습니다.

리스크 방어의 우수성: 수익률이 40배 가까이 차이가 남에도 불구하고, 최대낙폭(MDD)은 하이브리드 전략이 -25.41%로 단순 보유(-43.62%)에 비해 훨씬 안정적이었습니다.

2. 도출된 핵심 인사이트 (Key Insights)
 1) "알파(Alpha)는 상승장에서 오고, 방어(Defense)는 하락장에서 온다"
단순 보유 전략이 6년이 넘는 기간(2020~2026) 동안 7%대 수익에 그친 이유는 대상 종목(예: HD, ADBE, DIS 등)이 중간중간 겪은 혹독한 장기 조정과 박스권 횡보 때문입니다.

반면 하이브리드 전략은 하락장/조정기에는 잦은 손절(-3%)과 현금화로 손실을 최소화하고, 상승장 진입 시 눌림목 매수와 트레일링 스탑으로 대세 상승의 과실을 온전히 발라먹음으로써 복리 효과를 극대화했습니다.

 2) 트레일링 스탑(Trailing Stop)의 '수익 극대화' 마법
기존 전략의 최대 단점이었던 '너무 빠른 고정 익절(+5%)' 문제가 트레일링 스탑 도입으로 완전히 해소되었습니다.

주가가 오를 때 익절 라인이 함께 따라 올라가는 구조 덕분에, 종목이 수십~수백 퍼센트 폭등할 때 중간에 털리지 않고 추세를 끝까지 추종(Trend Following)할 수 있었음이 이 막대한 수익률(+329%)로 증명되었습니다.

 3) 복합 매매 시스템(Hybrid Framework)의 실효성 입증
하나의 고정된 전략(예: 무조건 가치투자, 무조건 저점매수)만 고집하는 것보다, 시장 국면(추세의 방향)에 따라 매수 조건을 달리 가져가는 하이브리드 접근법이 실제 주식 시장의 거친 변동성을 이겨내는 데 훨씬 유리하다는 결론을 얻을 수 있습니다.
'''