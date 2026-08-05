'''
logic_17과의 차이

돌파 매수(추세추종) 진입조건 추가

돌파 매수 시, 진입 유형별 분리된 청산 파라미터를 수정

진입 유형(돌파매수, 눌림목 진입)별 성과 분리 집계
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# [설정] 분봉 백테스팅 (최근 1년 간의 1시간봉 데이터 활용)
PERIOD = "1y"
INTERVAL = "1h"
COMMISSION_PCT = 0.0005
SLIPPAGE_PCT = 0.0005

# 눌림목(평균회귀) 손절폭 설정 - 손절 위주 조기청산 문제 완화를 위해 완화
# (기존: 최소 1.5%, ATR*1.5배 -> 완화: 최소 2.5%, ATR*2.0배)
# 트레일링/시간제한은 그대로 둬서 이번엔 손절폭 하나만 바꾼 효과를 격리해서 확인
PULLBACK_SL_MIN_PCT = 0.025
PULLBACK_SL_ATR_MULT = 2.0

def get_sector_universe():
    return {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "ADBE": "Technology",
        "AMZN": "Consumer Cyclical", "TSLA": "Consumer Cyclical", "NKE": "Consumer Cyclical", "HD": "Consumer Cyclical",
        "GOOGL": "Communication Services", "META": "Communication Services", "DIS": "Communication Services",
        "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare",
        "AXP": "Financials", "V": "Financials", "BA": "Industrials", "CAT": "Industrials",
        "XOM": "Energy", "CVX": "Energy", "NEE": "Utilities", "WMT": "Consumer Defensive"
    }

def fetch_intraday_data(ticker, period=PERIOD, interval=INTERVAL):
    try:
        # yfinance를 통한 분봉 데이터 로드 (최대 1년 치 1시간봉)
        df = yf.download(ticker, period=period, interval=interval, progress=False, multi_level_index=False)
        if df.empty or len(df) < 200: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def add_intraday_indicators(df, breakout_lookback=20):
    df = df.copy()
    # 분봉(1시간봉) 기준 장기 추세선 (약 30거래일 분량의 시간봉 = 30 * 6.5 = 195 ≒ 200시간봉)
    df['Trend_MA'] = df['Close'].rolling(window=200).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    bb_std_val = df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2.0 * bb_std_val
    df['RSI'] = calc_rsi(df['Close'], period=14)
    df['ATR'] = calc_atr(df, period=14)
    df['Daily_Return'] = df['Close'].pct_change()
    # 신규: 돌파 매수 판단용 최근 N바 고점 (전일까지 값만 사용하도록 1바 shift, 미래참조 방지)
    df['Rolling_High'] = df['Close'].rolling(window=breakout_lookback).max().shift(1)
    return df

def run_intraday_strategy(df, market_trend_series,
                           breakout_rsi_min=50, breakout_rsi_max=80):
    """
    변경점 (이번 버전):
    - 기존 눌림목(평균회귀) 진입 조건에 더해, 눌림목 조건이 안 나오는
      강한 단방향 상승추세를 놓치지 않도록 '돌파 매수(추세추종)' 진입 조건을 보조로 추가.
      -> 눌림목 조건이 우선 적용되고, 안 맞을 때만 돌파 조건을 확인 (상호배타적).
    - 돌파 매수 트레이드는 눌림목 트레이드와 손절/트레일링/시간제한 파라미터를 분리해서 사용
      (평균회귀용으로 튜닝된 타이트한 손절을 추세추종에 그대로 쓰면 노이즈에 바로 털리기 때문).
    - trade_log에 entry_type('pullback' / 'breakout')을 함께 기록해서
      두 진입 유형의 성과를 나중에 따로 집계할 수 있도록 함.
    """
    df = df.copy()
    position = 0
    entry_price = 0.0
    entry_time = None
    entry_type = None
    highest_price = 0.0
    holding_bars = 0
    strategy_returns = []
    trade_log = []

    for i in range(len(df)):
        current_time = df.index[i]
        # 날짜만 추출하여 시장 국면 필터 매칭
        current_date_str = pd.to_datetime(current_time).strftime('%Y-%m-%d')

        close = df['Close'].iloc[i]
        trend_ma = df['Trend_MA'].iloc[i]
        rsi = df['RSI'].iloc[i]
        lower = df['BB_Lower'].iloc[i]
        atr = df['ATR'].iloc[i]
        bar_ret = df['Daily_Return'].iloc[i]
        rolling_high = df['Rolling_High'].iloc[i]

        is_market_bullish = True
        if current_date_str in market_trend_series.index:
            is_market_bullish = market_trend_series.loc[current_date_str]

        if position == 0:
            if not is_market_bullish:
                strategy_returns.append(0.0)
                continue

            is_downtrend = pd.notna(trend_ma) and close < trend_ma
            if is_downtrend:
                is_pullback_condition = (pd.notna(rsi) and rsi <= 35) or (pd.notna(lower) and close <= lower)
            else:
                is_pullback_condition = pd.notna(rsi) and rsi <= 45

            # 신규: 눌림목 조건 미충족 시에만 돌파 매수 조건 확인
            # (상승추세 + 최근 N바 고점 돌파 + RSI가 과매도도 과열 극단도 아닌 강세 구간)
            is_uptrend = pd.notna(trend_ma) and close > trend_ma
            is_breakout_condition = (
                not is_pullback_condition and is_uptrend
                and pd.notna(rolling_high) and close > rolling_high
                and pd.notna(rsi) and breakout_rsi_min <= rsi <= breakout_rsi_max
            )

            if is_pullback_condition:
                entry_type = 'pullback'
            elif is_breakout_condition:
                entry_type = 'breakout'
            else:
                entry_type = None

            if entry_type is not None:
                position = 1
                entry_price = close * (1 + SLIPPAGE_PCT)
                entry_time = current_time
                highest_price = entry_price
                holding_bars = 0
                strategy_returns.append(-COMMISSION_PCT)
            else:
                strategy_returns.append(0.0)
        else:
            holding_bars += 1
            if close > highest_price:
                highest_price = close

            current_atr_pct = (atr / close) if (pd.notna(atr) and close > 0) else 0.01

            if entry_type == 'breakout':
                # 추세추종용 파라미터: 노이즈에 바로 털리지 않도록 손절/트레일링 폭을 더 넓게,
                # 추세가 더 오래 이어질 수 있으므로 시간제한도 더 길게 (약 2주)
                dynamic_sl_pct = max(0.025, current_atr_pct * 2.5)
                dynamic_trailing_pct = max(0.04, current_atr_pct * 3.0)
                time_limit_bars = 80
            else:
                # 기존 눌림목(평균회귀)용 파라미터 - 손절폭만 완화 (트레일링/시간제한은 유지해 효과 격리)
                dynamic_sl_pct = max(PULLBACK_SL_MIN_PCT, current_atr_pct * PULLBACK_SL_ATR_MULT)
                dynamic_trailing_pct = max(0.025, current_atr_pct * 2.0)
                time_limit_bars = 40

            hit_sl = close <= entry_price * (1 - dynamic_sl_pct)
            hit_trailing = close <= highest_price * (1 - dynamic_trailing_pct)
            hit_time_limit = holding_bars >= time_limit_bars

            if hit_sl or hit_trailing or hit_time_limit:
                exit_price = close * (1 - SLIPPAGE_PCT)
                # trade_ret_total: 트레이드 리포팅(trade_log, 승률/평균수익률 집계)용 총 손익률.
                # 예전 버전에서는 이 값을 그대로 equity curve(strategy_returns)에도 넣었는데,
                # 그러면 보유 중 매 바마다 이미 누적해온 일별 등락률과 겹쳐서
                # 해당 트레이드의 가격 변동분이 이중으로 반영되는 버그가 있었음.
                trade_ret_total = (exit_price / entry_price) - 1 - COMMISSION_PCT

                # equity curve에는 "이 바의 등락률 + 청산 비용"만 반영 (마크투마켓, 이중계산 방지)
                exit_bar_return = bar_ret * position - SLIPPAGE_PCT - COMMISSION_PCT
                strategy_returns.append(exit_bar_return)

                exit_reason = "손절" if hit_sl else ("트레일링" if hit_trailing else "시간제한")
                trade_log.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "entry_type": entry_type,
                    "holding_bars": holding_bars,
                    "trade_return": trade_ret_total,
                    "exit_reason": exit_reason,
                })

                position = 0
                entry_price = 0.0
                entry_time = None
                entry_type = None
                highest_price = 0.0
                holding_bars = 0
            else:
                strategy_returns.append(bar_ret * position)

    df['Strategy_Return'] = strategy_returns
    return df, trade_log


def compute_perf(return_series):
    """수익률 시리즈(단순 등락률, 각 바의 소수점 수익률) -> (최종수익률, MDD) 계산"""
    if len(return_series) == 0:
        return 0.0, 0.0
    cum = (1 + return_series).cumprod()
    final_return = cum.iloc[-1] - 1
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return final_return, mdd


def summarize_trades(trade_log, interval_hours=1, entry_type_filter=None):
    """거래 횟수 / 평균 보유기간(바, 시간, 거래일 환산) / 승률 / 청산사유 분포 요약
    entry_type_filter='pullback' 또는 'breakout'을 넘기면 해당 유형만 집계."""
    if entry_type_filter is not None:
        trade_log = [t for t in trade_log if t.get("entry_type") == entry_type_filter]

    n_trades = len(trade_log)
    if n_trades == 0:
        return {
            "n_trades": 0, "avg_holding_bars": 0.0, "avg_holding_hours": 0.0,
            "avg_holding_trading_days": 0.0, "win_rate": 0.0, "avg_return": 0.0,
            "total_return_sum": 0.0, "exit_reason_counts": {}
        }
    holding_bars_list = [t["holding_bars"] for t in trade_log]
    returns_list = [t["trade_return"] for t in trade_log]
    avg_holding_bars = float(np.mean(holding_bars_list))
    avg_holding_hours = avg_holding_bars * interval_hours
    avg_holding_trading_days = avg_holding_hours / 6.5  # 미국장 하루 약 6.5시간 기준
    win_rate = float(np.mean([r > 0 for r in returns_list])) * 100
    avg_return = float(np.mean(returns_list)) * 100  # 거래 1건당 평균 수익률(%)

    exit_reason_counts = {}
    for t in trade_log:
        exit_reason_counts[t["exit_reason"]] = exit_reason_counts.get(t["exit_reason"], 0) + 1

    return {
        "n_trades": n_trades,
        "avg_holding_bars": avg_holding_bars,
        "avg_holding_hours": avg_holding_hours,
        "avg_holding_trading_days": avg_holding_trading_days,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "exit_reason_counts": exit_reason_counts,
    }


if __name__ == "__main__":
    print("[*] 최근 1년 분봉 시장 국면 필터(SPY 1시간봉) 구축 중...")
    spy_df = fetch_intraday_data("SPY", period=PERIOD, interval=INTERVAL)
    if spy_df is not None:
        spy_df['Market_MA'] = spy_df['Close'].rolling(window=200).mean()
        spy_df['Date_Str'] = spy_df.index.strftime('%Y-%m-%d')
        # 일별 기준으로 시장 트렌드 매핑
        market_trend = (spy_df['Close'] > spy_df['Market_MA']).groupby(spy_df['Date_Str']).last()
    else:
        market_trend = pd.Series(True)

    universe_dict = get_sector_universe()
    qualified_stocks = []

    print("[*] 종목별 1시간봉 데이터 스크리닝 및 모멘텀 정렬 중...")
    for ticker, sector in universe_dict.items():
        df = fetch_intraday_data(ticker, period=PERIOD, interval=INTERVAL)
        if df is None: continue
        df = add_intraday_indicators(df)
        if len(df) < 100: continue

        # 최근 40시간봉(약 1주일) 기준 단기 모멘텀 계산
        recent_momentum = df['Close'].pct_change(40).iloc[-1]
        momentum_val = recent_momentum if pd.notna(recent_momentum) else 0

        qualified_stocks.append({
            'ticker': ticker, 'sector': sector, 'df': df, 'momentum': momentum_val
        })

    qualified_stocks = sorted(qualified_stocks, key=lambda x: x['momentum'], reverse=True)

    portfolio_stocks = []
    selected_sectors = set()
    for item in qualified_stocks:
        if item['sector'] not in selected_sectors:
            portfolio_stocks.append(item)
            selected_sectors.add(item['sector'])
        if len(portfolio_stocks) >= 3: break

    print(f"\n[분봉 포트폴리오 구성 완료]")
    for p in portfolio_stocks:
        print(f" - 종목: {p['ticker']} (섹터: {p['sector']}, 1시간모멘텀: {p['momentum']*100:.2f}%)")

    portfolio_results, bnh_results = [], []
    all_trade_logs = {}      # 종목별 거래 로그 저장
    all_strategy_returns = {}  # 종목별 전략 수익률 시리즈 (개별/구간별 분석용)
    all_bnh_returns = {}       # 종목별 단순보유 수익률 시리즈

    for p in portfolio_stocks:
        opt_df, trade_log = run_intraday_strategy(p['df'], market_trend)
        portfolio_results.append(opt_df['Strategy_Return'])
        bnh_results.append(p['df']['Daily_Return'])
        all_trade_logs[p['ticker']] = trade_log
        all_strategy_returns[p['ticker']] = opt_df['Strategy_Return']
        all_bnh_returns[p['ticker']] = p['df']['Daily_Return']

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
    print(f" 최근 1년 분봉(1시간봉) 하이브리드 전략 최종 성과")
    print(f"{'='*55}")
    print(f" [분봉 동적 하이브리드 전략]")
    print(f"  - 최종 수익률 : {final_port_return * 100:.2f}%")
    print(f"  - 최대낙폭(MDD): {port_mdd * 100:.2f}%")
    print(f"{'-'*55}")
    print(f" [단순보유 (Buy & Hold)]")
    print(f"  - 최종 수익률 : {final_bnh_return * 100:.2f}%")
    print(f"  - 최대낙폭(MDD): {bnh_mdd * 100:.2f}%")
    print(f"{'='*55}")

    # ---- 신규: 종목별 거래 횟수 / 평균 보유기간 / 승률 / 청산사유 ----
    print(f"\n{'='*55}")
    print(f" 종목별 거래 통계 (거래횟수 / 평균 보유기간 / 승률)")
    print(f"{'='*55}")

    total_trades = 0
    total_holding_bars_weighted = 0

    for p in portfolio_stocks:
        ticker = p['ticker']
        stats = summarize_trades(all_trade_logs[ticker], interval_hours=1)
        total_trades += stats['n_trades']
        total_holding_bars_weighted += stats['avg_holding_bars'] * stats['n_trades']

        print(f"\n [{ticker}]")
        print(f"  - 거래 횟수        : {stats['n_trades']}건")
        print(f"  - 평균 보유기간    : {stats['avg_holding_bars']:.1f}바 "
              f"(≈ {stats['avg_holding_hours']:.1f}시간, ≈ {stats['avg_holding_trading_days']:.1f}거래일)")
        print(f"  - 승률             : {stats['win_rate']:.1f}%")
        if stats['exit_reason_counts']:
            reason_str = ", ".join([f"{k} {v}건" for k, v in stats['exit_reason_counts'].items()])
            print(f"  - 청산 사유 분포   : {reason_str}")

    print(f"\n{'-'*55}")
    if total_trades > 0:
        portfolio_avg_holding_bars = total_holding_bars_weighted / total_trades
        print(f" [포트폴리오 합산]")
        print(f"  - 총 거래 횟수     : {total_trades}건")
        print(f"  - 평균 보유기간(가중평균) : {portfolio_avg_holding_bars:.1f}바 "
              f"(≈ {portfolio_avg_holding_bars:.1f}시간, ≈ {portfolio_avg_holding_bars/6.5:.1f}거래일)")
        # 왕복 비용 총량(대략치): 거래당 진입+청산 각각 commission+slippage 발생
        round_trip_cost_pct = (COMMISSION_PCT + SLIPPAGE_PCT) * 2
        est_total_cost_pct = total_trades * round_trip_cost_pct * 100
        print(f"  - 거래당 왕복 비용(수수료+슬리피지): {round_trip_cost_pct*100:.2f}% "
              f"-> 전체 거래 비용 합계 대략 {est_total_cost_pct:.2f}%p (단순 합산치, 복리효과 미반영)")
    print(f"{'='*55}")

    # ---- 신규: 진입 유형별(눌림목 vs 돌파) 성과 비교 ----
    # -> 이번에 추가한 돌파 매수가 실제로 도움이 됐는지, 아니면 특정 종목(GOOGL 등) 하나 때문에
    #    우연히 좋아 보이는 건지 구분하기 위해 진입 유형 단위로 따로 집계
    print(f"\n{'='*55}")
    print(f" 진입 유형별 성과 비교 (눌림목 vs 돌파매수)")
    print(f"{'='*55}")
    for p in portfolio_stocks:
        ticker = p['ticker']
        log = all_trade_logs[ticker]
        pullback_stats = summarize_trades(log, entry_type_filter='pullback')
        breakout_stats = summarize_trades(log, entry_type_filter='breakout')
        print(f"\n [{ticker}]")
        print(f"  - 눌림목: {pullback_stats['n_trades']}건 | 승률 {pullback_stats['win_rate']:.1f}% | "
              f"건당 평균수익률 {pullback_stats['avg_return']:+.2f}% | "
              f"평균보유 {pullback_stats['avg_holding_trading_days']:.1f}거래일")
        print(f"  - 돌파  : {breakout_stats['n_trades']}건 | 승률 {breakout_stats['win_rate']:.1f}% | "
              f"건당 평균수익률 {breakout_stats['avg_return']:+.2f}% | "
              f"평균보유 {breakout_stats['avg_holding_trading_days']:.1f}거래일")
    print(f"{'='*55}")

    # ---- 종목별 개별 성과 (전략 vs 단순보유) ----
    # -> 포트폴리오 평균이 아니라 종목 하나하나가 각각 얼마를 벌었는지 봐서
    #    특정 종목이 전체 성과를 끌어내렸는지 확인
    print(f"\n{'='*55}")
    print(f" 종목별 개별 성과 (최근 1년 전체, 전략 vs 단순보유)")
    print(f"{'='*55}")
    for p in portfolio_stocks:
        ticker = p['ticker']
        strat_ret, strat_mdd = compute_perf(all_strategy_returns[ticker].fillna(0))
        bnh_ret, bnh_mdd_each = compute_perf(all_bnh_returns[ticker].fillna(0))
        gap = strat_ret - bnh_ret
        print(f"\n [{ticker}]")
        print(f"  - 전략   : 수익률 {strat_ret*100:6.2f}%  |  MDD {strat_mdd*100:6.2f}%")
        print(f"  - 단순보유: 수익률 {bnh_ret*100:6.2f}%  |  MDD {bnh_mdd_each*100:6.2f}%")
        print(f"  - 전략-단순보유 격차: {gap*100:+.2f}%p {'(전략 열세)' if gap < 0 else '(전략 우세)'}")

    # ---- 신규 2: 상반기/하반기 국면별 성과 (포트폴리오 + 종목별) ----
    # -> 전체 구간을 절반으로 나눠 각 구간에서 따로 누적수익률/MDD를 재계산.
    #    특정 시기(예: 강한 상승장 구간)에서만 전략이 유독 부진한지 확인.
    #    yfinance 1시간봉은 대개 최근 730일 정도만 제공되어 일봉처럼
    #    2020/2022/2023년 등 서로 다른 '해'로 나눠 검증하기는 어려우므로,
    #    현재 확보된 1년 구간 내에서 상반기/하반기로 나누는 방식으로 대체.
    print(f"\n{'='*55}")
    print(f" 상반기 vs 하반기 국면별 성과")
    print(f"{'='*55}")

    full_start = port_df.index.min()
    full_end = port_df.index.max()
    mid_date = full_start + (full_end - full_start) / 2
    print(f" (분기 기준일: {mid_date} / 전체구간: {full_start} ~ {full_end})")

    def split_and_perf(return_series, mid):
        s = return_series.fillna(0)
        first_half = s[s.index <= mid]
        second_half = s[s.index > mid]
        return compute_perf(first_half), compute_perf(second_half)

    # 포트폴리오 단위
    (port_h1_ret, port_h1_mdd), (port_h2_ret, port_h2_mdd) = split_and_perf(port_df['Port_Return'], mid_date)
    (bnh_h1_ret, bnh_h1_mdd), (bnh_h2_ret, bnh_h2_mdd) = split_and_perf(bnh_df['BnH_Return'], mid_date)

    print(f"\n [포트폴리오 합산]")
    print(f"  - 상반기: 전략 {port_h1_ret*100:6.2f}%(MDD {port_h1_mdd*100:6.2f}%)  |  "
          f"단순보유 {bnh_h1_ret*100:6.2f}%(MDD {bnh_h1_mdd*100:6.2f}%)")
    print(f"  - 하반기: 전략 {port_h2_ret*100:6.2f}%(MDD {port_h2_mdd*100:6.2f}%)  |  "
          f"단순보유 {bnh_h2_ret*100:6.2f}%(MDD {bnh_h2_mdd*100:6.2f}%)")

    # 종목별
    for p in portfolio_stocks:
        ticker = p['ticker']
        (s_h1_ret, s_h1_mdd), (s_h2_ret, s_h2_mdd) = split_and_perf(all_strategy_returns[ticker], mid_date)
        (b_h1_ret, b_h1_mdd), (b_h2_ret, b_h2_mdd) = split_and_perf(all_bnh_returns[ticker], mid_date)
        print(f"\n [{ticker}]")
        print(f"  - 상반기: 전략 {s_h1_ret*100:6.2f}%(MDD {s_h1_mdd*100:6.2f}%)  |  "
              f"단순보유 {b_h1_ret*100:6.2f}%(MDD {b_h1_mdd*100:6.2f}%)")
        print(f"  - 하반기: 전략 {s_h2_ret*100:6.2f}%(MDD {s_h2_mdd*100:6.2f}%)  |  "
              f"단순보유 {b_h2_ret*100:6.2f}%(MDD {b_h2_mdd*100:6.2f}%)")
    print(f"\n{'='*55}")

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(cum_portfolio.index, cum_portfolio, label='Intraday Hybrid Strategy (1h)', color='darkorange', linewidth=2)
    plt.plot(cum_bnh.index, cum_bnh, label='Buy & Hold', color='gray', linestyle='--', linewidth=1.5)
    plt.title('Intraday (1h) Hybrid Strategy vs. Buy & Hold (Last 1 Year)')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date/Time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('intraday_backtest_result.png', dpi=120)  # 스크립트 실행 위치(현재 작업 디렉토리) 기준 상대경로로 저장
    print("\n[*] 그래프 저장 완료: intraday_backtest_result.png")
    plt.show()
