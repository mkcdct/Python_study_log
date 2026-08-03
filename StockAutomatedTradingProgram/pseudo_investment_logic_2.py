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

# 이동평균선 이용한 golden cross/death cross 전략 & 거래량 필터 조합

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

# 거래량 필터: 최근 20일 평균 거래량 대비 오늘 거래량이 높은지
df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
df['Volume_Confirm'] = df['Volume'] > df['Volume_MA20']

# 골든크로스가 "발생한 날" 잡아내기 (0->1로 바뀐 순간)
df['Golden_Cross'] = (df['Signal'] == 1) & (df['Signal'].shift(1) == 0)
df['Death_Cross'] = (df['Signal'] == 0) & (df['Signal'].shift(1) == 1)

# 교차 전후 3일 이내에 거래량 확인이 있었는지 체크
# rolling window로 "앞뒤 3일 중 하나라도 거래량 조건 만족"을 판단
window = 3
df['Volume_Confirm_Nearby'] = (
    df['Volume_Confirm']
    .rolling(window=2*window+1, center=True, min_periods=1)
    .max()
    .astype(bool)
)

# 필터링된 최종 신호 만들기
# - 골든크로스가 뜨고 + 거래량도 확인되면 -> 매수 시작
# - 데드크로스가 뜨면 -> 무조건 매도 (청산은 거래량과 무관하게 실행)
df['Filtered_Signal'] = 0
position = 0

for i in range(len(df)):
    if df['Golden_Cross'].iloc[i] and df['Volume_Confirm_Nearby'].iloc[i]:
        position = 1
    elif df['Death_Cross'].iloc[i]:
        position = 0
    df.iloc[i, df.columns.get_loc('Filtered_Signal')] = position

df['Position'] = df['Filtered_Signal'].shift(1)

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
NVDA 종목 특성상 지난 1년동안 강한 상승과 잦은 조정이 반복됐기 때문에,
박스권/하락장에서 빛을 발하는 골든크로스 전략이 단순 보유 전략보다 더 낮은 수익률을 기록했음.
pseudo_investment_logic_3에서는 우선 전략의 수익률 자체보다 샤프비율(위험 대비 수익)을 단순 보유 전략과
비교해보고, 다른 년도 중 박스권/하락장이 있었던 년도에서도 동일한 전략을 적용해
해당 전략의 유효성을 검증해볼 것.
'''