# Backtesting logic for the investment strategy

import yfinance as yf

# 종목 코드 : 엔비디아
ticker = "NVDA"

# 최근 1년치 일별 데이터 가져오기 (시가/고가/저가/종가/거래량)
df = yf.download(ticker, period="1y", interval="1d")

# 데이터가 잘 호출됐는지 확인
print(df.head())
print("...")
print(df.tail())
print(f"\n총 {len(df)}개의 거래일 데이터를 가져왔습니다.")

# CSV로 저장(다음에 API 재호출 없이 바로 사용가능)
df.to_csv("nvda_1y.csv")
print("nvda_1y.csv 로 저장 완료")

