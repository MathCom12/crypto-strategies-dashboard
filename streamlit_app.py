import streamlit as st
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

# Streamlit 페이지 구성
st.title("캔들차트 예측 게임")
st.write("평균(mean)과 표준편차(std)를 입력하여 랜덤 수익률을 생성하고, 이를 통해 가격 데이터를 생성하여 캔들차트를 시각화합니다.")

# 사용자 입력
mean = st.number_input("평균 수익률 (mean)", value=0.0, step=0.001, format="%.3f")
std = st.number_input("표준편차 (std)", value=0.01, step=0.001, format="%.3f")
days = st.number_input("기간 (days)", value=20, min_value=5, max_value=365, step=1)

# 랜덤 수익률 생성
def generate_random_returns(mean, std, periods):
    """mean과 std를 기반으로 랜덤 수익률 생성"""
    returns = np.random.normal(loc=mean, scale=std, size=periods)
    return returns

# 수익률을 이용해 가격 생성
def generate_prices_from_returns(initial_price, returns):
    """수익률로부터 가격을 계산"""
    prices = [initial_price]
    for r in returns:
        new_price = prices[-1] * (1 + r)
        prices.append(new_price)
    return np.array(prices)

# 초기 가격 설정
initial_price = 100.0

# 세션 상태를 사용하여 데이터 보존
if 'ohlc_df' not in st.session_state or 'correct_answer' not in st.session_state or st.session_state.get('new_problem', False):
    # 새로운 데이터를 생성하여 세션 상태에 저장
    returns = generate_random_returns(mean, std, days * 10)
    prices = generate_prices_from_returns(initial_price, returns)
    
    # 가격 데이터 프레임 생성
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    ohlc_data = []

    # OHLC 데이터 생성
    for i in range(days):
        period_prices = prices[i*10:(i+1)*10]  # 각 기간의 10개 가격 데이터
        open_price = period_prices[0]
        high_price = np.max(period_prices)
        low_price = np.min(period_prices)
        close_price = period_prices[-1]
        ohlc_data.append([open_price, high_price, low_price, close_price])

    # 데이터프레임으로 변환하여 세션 상태에 저장
    ohlc_df = pd.DataFrame(ohlc_data, columns=['Open', 'High', 'Low', 'Close'], index=dates)
    st.session_state['ohlc_df'] = ohlc_df

    # 정답 생성
    price_diff = ohlc_df['Close'].iloc[-1] - ohlc_df['Close'].iloc[-2]
    if price_diff > 0:
        st.session_state['correct_answer'] = '상승'
    elif price_diff < 0:
        st.session_state['correct_answer'] = '하락'
    else:
        st.session_state['correct_answer'] = '변동 없음'
    
    # 새로운 문제가 생성되었으므로 False로 설정
    st.session_state['new_problem'] = False

# 세션 상태에서 데이터 가져오기
ohlc_df = st.session_state['ohlc_df']
correct_answer = st.session_state['correct_answer']

# 마지막 바를 숨기기 위한 조작
hidden_data = ohlc_df[:-1]  # 마지막 바를 숨김
visible_data = ohlc_df  # 전체 데이터

# 캔들차트 그리기 (마지막 바는 ?로 표시)
st.subheader("랜덤 캔들차트 (마지막 바는 예측을 위해 숨겨짐)")

fig, ax = plt.subplots()
mpf.plot(hidden_data, type='candle', style='charles', volume=False, ax=ax)

# 마지막 바를 물음표로 표시
last_date = ohlc_df.index[-1]
ax.annotate('?', xy=(last_date, ohlc_df['Close'].iloc[-2]), xytext=(last_date, ohlc_df['Close'].iloc[-2] + 5),
            fontsize=15, color='red', ha='center')

st.pyplot(fig)

# 다음 캔들 예측
st.subheader("다음 캔들 예측")
choices = ['상승', '하락', '변동 없음']
prediction = st.radio("다음 캔들의 변화를 예측하세요:", choices)

# 예측 결과 확인 버튼 클릭 시 동작
if st.button("예측 결과 확인"):
    st.write(f"정답: {correct_answer}")

    # 전체 데이터로 차트 업데이트
    fig, ax = plt.subplots()
    mpf.plot(visible_data, type='candle', style='charles', volume=False, ax=ax)
    st.pyplot(fig)

    # 예측 결과 확인
    if prediction == correct_answer:
        st.success("정답입니다!")
    else:
        st.error("오답입니다. 다시 시도해보세요.")
    
    # 예측 결과 확인 후에 "다음 문제" 버튼 표시
    if st.button("다음 문제"):
        # 세션 상태를 갱신하여 새로운 문제를 생성하도록 설정
        st.session_state['new_problem'] = True
        # 리로드하여 새로운 문제 생성
        st.experimental_rerun()

