# Backtesting logic for the investment strategy

import yfinance as yf

# 종목 코드 : 엔비디아
ticker = "NVDA"

# 최근 1년치 일별 데이터 가져오기 (시가/고가/저가/종가/거래량)
df = yf.download(ticker, period="1y", interval="1d")

# multindex 제거
df.columns = df.columns.get_level_values(0)

# 데이터가 잘 호출됐는지 확인
print(df.head())
print("...")
print(df.tail())
print(f"\n총 {len(df)}개의 거래일 데이터를 가져왔습니다.")

# CSV로 저장(다음에 API 재호출 없이 바로 사용가능)
df.to_csv("nvda_1y.csv")
print("nvda_1y.csv 로 저장 완료")

# 이동평균선 이용한 golden cross/death cross 전략.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("nvda_1y.csv", index_col = 0, parse_dates = True)

# ------------------------------
# 1. 이동평균선 계산
# ------------------------------

df['MA20'] = df['Close'].rolling(window=20).mean() # 20일 이동평균선
df['MA60'] = df['Close'].rolling(window=60).mean() # 60일 이동평균선

# ------------------------------
# 2. 매매 시그널 생성
# ------------------------------
# 20일선이 60일선 위에 있으면 1(매수/보유), 아래면 0(현금/미보유)

df['Signal'] = 0
df.loc[df['MA20'] > df['MA60'], 'Signal'] = 1

df['Position'] = df['Signal'].shift(1) # 실제 매매는 다음날 체결

# ------------------------------
# 3. 수익률 계산
# ------------------------------

df['Daily_Return'] = df['Close'].pct_change()

# 전략 수익률 : 보유 중일 때만 시장 수익률을 따라감
df['Strategy_Return'] = df['Daily_Return'] * df['Position']

# ------------------------------
# 4. 누적 수익률 계산
# ------------------------------

df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod()
df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

# ------------------------------
# 5. 결과 시각화
# ------------------------------
print(df[['Close', 'MA20', 'MA60', 'Signal', 'Position', 'Strategy_Return']].tail(15))

final_market_return = df['Cumulative_Market_Return'].iloc[-1] - 1
final_strategy_return = df['Cumulative_Strategy'].iloc[-1] - 1

print(f"\n[단순 보유(Buy & Hold)] 최종 수익률: {final_market_return*100:.2f}%")
print(f"[골든크로스 전략]      최종 수익률: {final_strategy_return*100:.2f}%")

# ---------------------------
# 6. 시각화
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Cumulative_Market_Return'], label='Buy & Hold')
plt.plot(df.index, df['Cumulative_Strategy'], label='Golden Cross Strategy')
plt.title('NVDA: Buy & Hold vs Golden Cross Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

'''
시장이 하락한 날들 중에서, 해당 전략이 실제로 얼마나 손실을 
피했는지 계산 : MDD(최대낙폭)을 둘 다 계산하여 비교
'''

def calc_mdd(cumulative_series):
    peak = cumulative_series.cummax()
    drawdown = (cumulative_series - peak) / peak
    return drawdown.min()

mdd_market = calc_mdd(df['Cumulative_Market_Return'])
mdd_strategy = calc_mdd(df['Cumulative_Strategy'])

print(f"Buy & Hold 최대낙폭: {mdd_market*100:.2f}%")
print(f"골든크로스 전략 최대낙폭: {mdd_strategy*100:.2f}%")

'''
golden cross 전략은 이번 테스트에서 하락 방어력이 MDD 기준으로 약간 있었지만,
그 대가로 상승장 수익을 크게 희생했다. 
즉, NVDA 종목에 대해서는 단순 보유 전략이 더 나은 선택지였다.
'''

