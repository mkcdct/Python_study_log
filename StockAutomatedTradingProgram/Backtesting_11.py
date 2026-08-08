'''
1. 유니버스 풀 및 기간 설정 조건
대상 유니버스: 기술, 경기소비재, 통신서비스, 헬스케어, 금융, 산업재, 에너지, 유틸리티, 경기방어주 등 9개 GICS 섹터에 속한 미국 대형주 56개 종목

백테스팅 기간: 2020년 1월 1일부터 2026년 6월 30일까지 (코로나 폭락장 및 급등락 시기 포함)

2. 하락 추세 종목 스크리닝 조건 (발굴 조건)
각 종목별 과거 데이터를 바탕으로 아래 두 가지 조건을 모두 충족하는 종목만 후보로 선정합니다.

장기 추세 조건: 전체 기간 중 200일 이동평균선 아래에 머문 날의 비율이 5% 이상인 종목

수익률 범위 조건: 기간 내 총 수익률이 -70% 이상 ~ +100% 미만 사이에 해당하는 종목

3. 섹터 분산 포트폴리오 구성 조건
선정 개수: 스크리닝된 종목 중 최대 3개 종목 선정

섹터 중복 방지: 동일 섹터의 종목이 겹치지 않도록 서로 다른 섹터에 속한 종목 우선 배분

4. 개별 종목 파라미터 최적화 조건 (Grid Search)
선정된 각 종목별로 가장 유리한 성과를 내기 위해 아래 파라미터 조합을 시뮬레이션하여 최적값을 탐색합니다.

RSI 침체 기준: 30, 35, 40 중 택일

익절 목표치 (Take Profit): +3% 또는 +5% 중 택일

손절 기준 (Stop Loss): -2% 또는 -4% 중 택일

최적화 기준: 기간 내 누적 수익률이 가장 높았던 조합 선택

5. 전략 매수(진입) 시그널 조건
포지션이 없는 상태에서 아래의 두 가지 조건을 동시에 만족할 때 매수 진입합니다.

당일 종가가 200일 이동평균선보다 아래에 위치 (하락 추세)

14일 RSI가 최적화된 침체 기준(예: 30 이하)이거나, 볼린저밴드 하단 이하로 가격이 이탈 (과매도)

6. 전략 매도(청산) 시그널 조건
포지션을 보유한 상태에서 아래의 4가지 청산 조건 중 하나라도 먼저 충족되면 매도합니다.

익절 (Take Profit): 진입 가격 대비 최적화된 목표 수익률 도달 시

손절 (Stop Loss): 진입 가격 대비 설정된 손실률 도달 시

기간 제한 (Time Limit): 최대 보유 기간(10거래일) 경과 시

중단선 도달 (Mid Target): 주가가 볼린저밴드 중심선(Mid) 이상으로 회복 시

7. 거래 비용 반영 조건
수수료 및 슬리피지: 진입 시와 청산 시 각각 0.05%의 거래 수수료와 슬리피지를 복합 차감하여 현실적인 수익률 계산

8. 포트폴리오 성과 비교 조건
전략 포트폴리오: 최종 선정된 3개 종목의 일별 전략 수익률을 동일 가중치(평균)로 합산하여 전체 전략의 누적 수익률 및 최대낙폭(MDD) 산출

단순보유(Buy & Hold) 비교군: 동일한 3개 종목을 기간 동안 단순 보유했을 때의 일별 수익률 평균을 산출하여 전략 성과와 비교 및 시각화
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import product

# 백테스팅 기간 설정 (코로나 폭락장 포함)
START_DATE = "2020-01-01"
END_DATE = "2026-06-30"

# 거래 비용 설정 (수수료 0.05%, 슬리피지 0.05%)
COMMISSION_PCT = 0.0005
SLIPPAGE_PCT = 0.0005

# ------------------------------
# 1. 섹터 정보가 포함된 확장된 대형주 유니버스 풀 설정
# ------------------------------
def get_sector_universe():
    print("[*] 섹터 정보가 포함된 미국 대형주 유니버스 로드 중...")
    universe_dict = {
        # Tech
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", 
        "AMD": "Technology", "QCOM": "Technology", "IBM": "Technology", 
        "ORCL": "Technology", "ADBE": "Technology", "CRM": "Technology",
        # Consumer Cyclical
        "AMZN": "Consumer Cyclical", "TSLA": "Consumer Cyclical", 
        "NKE": "Consumer Cyclical", "HD": "Consumer Cyclical", 
        "MCD": "Consumer Cyclical", "SBUX": "Consumer Cyclical", 
        "GM": "Consumer Cyclical", "F": "Consumer Cyclical", "BKNG": "Consumer Cyclical",
        # Communication Services
        "GOOGL": "Communication Services", "META": "Communication Services", 
        "NFLX": "Communication Services", "DIS": "Communication Services", 
        "VZ": "Communication Services", "T": "Communication Services",
        # Healthcare
        "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare", 
        "PFE": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare", 
        "TMO": "Healthcare", "DHR": "Healthcare", "BMY": "Healthcare", 
        "AMGN": "Healthcare", "GILD": "Healthcare", "MDT": "Healthcare", 
        "CVS": "Healthcare", "BAX": "Healthcare",
        # Financials
        "AXP": "Financials", "V": "Financials",
        # Industrials
        "BA": "Industrials", "UPS": "Industrials", "CAT": "Industrials", 
        "GE": "Industrials", "HON": "Industrials", "MMM": "Industrials",
        # Energy & Utilities
        "XOM": "Energy", "CVX": "Energy", "NEE": "Utilities", 
        "DUK": "Utilities", "SO": "Utilities",
        # Consumer Defensive
        "WMT": "Consumer Defensive", "KO": "Consumer Defensive", 
        "PEP": "Consumer Defensive", "COST": "Consumer Defensive", 
        "TGT": "Consumer Defensive"
    }
    return universe_dict


# ------------------------------
# 2. 데이터 다운로드
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
# 3. 기술적 지표 및 전략 실행 함수 (수수료/슬리피지 반영)
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


def run_strategy(df, rsi_oversold=35, take_profit_pct=0.04, stop_loss_pct=0.03, max_holding_days=10):
    df = df.copy()
    position = 0
    entry_price = 0.0
    holding_days = 0
    strategy_returns = []

    for i in range(len(df)):
        close = df['Close'].iloc[i]
        trend_ma = df['Trend_MA'].iloc[i]
        rsi = df['RSI'].iloc[i]
        lower = df['BB_Lower'].iloc[i]
        mid = df['BB_Mid'].iloc[i]
        daily_ret = df['Daily_Return'].iloc[i]

        if position == 0:
            is_downtrend = pd.notna(trend_ma) and close < trend_ma
            is_oversold = (pd.notna(rsi) and rsi <= rsi_oversold) or (pd.notna(lower) and close <= lower)
            
            if is_downtrend and is_oversold:
                position = 1
                entry_price = close * (1 + SLIPPAGE_PCT) # 매수 시 슬리피지 발생
                holding_days = 0
                strategy_returns.append(-COMMISSION_PCT) # 진입 수수료 차감
            else:
                strategy_returns.append(0.0)
        else:
            holding_days += 1
            # 청산 조건 검사
            hit_tp = close >= entry_price * (1 + take_profit_pct)
            hit_sl = close <= entry_price * (1 - stop_loss_pct)
            hit_time_limit = holding_days >= max_holding_days
            hit_mid_target = pd.notna(mid) and close >= mid

            if hit_tp or hit_sl or hit_time_limit or hit_mid_target:
                exit_price = close * (1 - SLIPPAGE_PCT) # 매도 시 슬리피지 발생
                trade_ret = (exit_price / entry_price) - 1 - COMMISSION_PCT # 청산 수수료 차감
                strategy_returns.append(trade_ret)
                position = 0
                entry_price = 0.0
                holding_days = 0
            else:
                # 보유 중 일일 수익률 반영
                strategy_returns.append(daily_ret * position)

    df['Strategy_Return'] = strategy_returns
    return df


# ------------------------------
# 4. 단일 종목 파라미터 최적화 (Grid Search)
# ------------------------------
def optimize_parameters(df):
    rsi_list = [30, 35, 40]
    tp_list = [0.03, 0.05]
    sl_list = [0.02, 0.04]
    
    best_score = -999
    best_params = (35, 0.04, 0.03)
    
    for rsi_v, tp_v, sl_v in product(rsi_list, tp_list, sl_list):
        temp_df = run_strategy(df, rsi_oversold=rsi_v, take_profit_pct=tp_v, stop_loss_pct=sl_v)
        cum_ret = (1 + temp_df['Strategy_Return'].fillna(0)).prod() - 1
        
        if cum_ret > best_score:
            best_score = cum_ret
            best_params = (rsi_v, tp_v, sl_v)
            
    return best_params


# ------------------------------
# 5. 메인 실행부 (스크리닝, 포트폴리오 구성, 전략 vs BnH 비교)
# ------------------------------
if __name__ == "__main__":
    universe_dict = get_sector_universe()
    qualified_stocks = []
    
    print(f"\n[*] 총 {len(universe_dict)}개 종목 대상 하락추세 반등 스크리닝 중 ({START_DATE} ~ {END_DATE})...")
    
    for ticker, sector in universe_dict.items():
        df = fetch_data(ticker, START_DATE, END_DATE)
        if df is None:
            continue
        
        df = add_indicators(df)
        df_target = df[(df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))].copy()
        
        if len(df_target) < 100:
            continue
            
        below_ma_ratio = (df_target['Close'] < df_target['Trend_MA']).mean()
        total_return = (df_target['Close'].iloc[-1] / df_target['Close'].iloc[0]) - 1
        
        # [조건 완화] 이평선 하단 5% 이상, 전체 기간 수익률 -70% ~ +100% 사이로 확장
        if below_ma_ratio >= 0.05 and -0.70 <= total_return < 1.00:
            qualified_stocks.append({'ticker': ticker, 'sector': sector, 'df': df_target})
            print(f"  [후보 발굴] {ticker} ({sector}, 기간수익률: {total_return*100:.1f}%, 하단체류비율: {below_ma_ratio*100:.1f}%)")

    # 최대 3종목 선정 (서로 다른 섹터 우선 배분)
    portfolio_stocks = []
    selected_sectors = set()
    
    for item in qualified_stocks:
        if item['sector'] not in selected_sectors:
            portfolio_stocks.append(item)
            selected_sectors.add(item['sector'])
        if len(portfolio_stocks) >= 3:
            break
            
    if not portfolio_stocks:
        print("\n조건에 부합하는 종목이 없습니다.")
    else:
        print(f"\n[포트폴리오 구성 완료] 총 {len(portfolio_stocks)}개 종목 선정 (섹터 분산 적용)")
        for p in portfolio_stocks:
            print(f" - 종목: {p['ticker']} (섹터: {p['sector']})")
            
        # 1) 전략 포트폴리오 성과 산출
        portfolio_results = []
        for p in portfolio_stocks:
            ticker = p['ticker']
            raw_df = p['df']
            
            print(f"\n[*] {ticker} 최적 파라미터 탐색(Grid Search) 중...")
            best_rsi, best_tp, best_sl = optimize_parameters(raw_df)
            print(f" -> {ticker} 최적 설정: RSI <= {best_rsi}, TP: {best_tp*100}%, SL: {best_sl*100}%")
            
            optimized_df = run_strategy(raw_df, rsi_oversold=best_rsi, take_profit_pct=best_tp, stop_loss_pct=best_sl)
            portfolio_results.append(optimized_df['Strategy_Return'])
            
        portfolio_df = pd.concat(portfolio_results, axis=1).fillna(0)
        portfolio_df['Port_Return'] = portfolio_df.mean(axis=1)
        
        cum_portfolio = (1 + portfolio_df['Port_Return']).cumprod()
        final_port_return = cum_portfolio.iloc[-1] - 1
        port_peak = cum_portfolio.cummax()
        port_mdd = ((cum_portfolio - port_peak) / port_peak).min()

        # 2) 단순 보유(Buy & Hold) 포트폴리오 성과 산출
        bnh_results = []
        for p in portfolio_stocks:
            bnh_results.append(p['df']['Daily_Return'])
            
        bnh_df = pd.concat(bnh_results, axis=1).fillna(0)
        bnh_df['BnH_Port_Return'] = bnh_df.mean(axis=1)
        
        cum_bnh = (1 + bnh_df['BnH_Port_Return']).cumprod()
        final_bnh_return = cum_bnh.iloc[-1] - 1
        bnh_peak = cum_bnh.cummax()
        bnh_mdd = ((cum_bnh - bnh_peak) / bnh_peak).min()
        
        # 3) 결과 출력
        print(f"\n{'='*55}")
        print(f" 포트폴리오 최종 백테스팅 비교 성과 ({START_DATE[:4]} ~ {END_DATE[:4]})")
        print(f"{'='*55}")
        print(f" [하락추세 반등 전략 포트폴리오]")
        print(f"  - 최종 수익률 : {final_port_return * 100:.2f}%")
        print(f"  - 최대낙폭(MDD): {port_mdd * 100:.2f}%")
        print(f"{'-'*55}")
        print(f" [단순보유 (Buy & Hold) 포트폴리오]")
        print(f"  - 최종 수익률 : {final_bnh_return * 100:.2f}%")
        print(f"  - 최대낙폭(MDD): {bnh_mdd * 100:.2f}%")
        print(f"{'='*55}")

        # 4) 시각화 비교
        plt.figure(figsize=(10, 5))
        plt.plot(cum_portfolio.index, cum_portfolio, label='Strategy Portfolio (Bounce)', color='darkorange', linewidth=2)
        plt.plot(cum_bnh.index, cum_bnh, label='Buy & Hold Portfolio', color='gray', linestyle='--', linewidth=1.5)
        plt.title('Portfolio Strategy vs. Buy & Hold Comparison')
        plt.ylabel('Cumulative Return')
        plt.xlabel('Date')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        '''
        insight 
        지나친 저점 매수(과매도) 필터의 한계 : 코로나 폭락장은 급락 후 곧바로 v자 반등이 나타났음.
        위 전략은 RSI30 이하 또는 볼린저밴드 하단 이탈이라는 엄격한 조건이 필요함.
        코로나 시기의 장세는 낙폭이 워낙 가파르다보니 바닥을 잡으려다 타이밍을 놓쳤거나,
        진입하자마자 급반등하면서 짧은 익절 구간에 걸려 상승장의 수익을 온전히 챙기지 못 함.
        
        추세 추종 전략의 딜레마 : 위 전략의 기본 전제는 200일선 아래 즉, 하락 추세임.
        코로나 직후 엄청난 유동성 유입으로 200일선을 강하게 돌파하며 장세가 우상향할 때, 
        위 전략은 여전히 하락 추세 종목의 틀에 갇혀 있거나, 뒤늦게 진입하여 상승 초입의 대시세를 놓쳤을 수 있음.
        
        정리 : 위 전략(역발상 단기 스윙)은 지속적인 하락 추세 또는 박스권 종목에 대해서는 유의미하지만,
        급락 후 강한 v자 반등 종목에 대해서는 단순 보유 전략보다 효과성이 떨어짐.
        '''