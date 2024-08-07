import streamlit as st
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import os

# 로그 파일 경로
LOG_FILE = "prediction_log.csv"

# 로그 파일 초기화
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=['Name', 'Correct', 'Total']).to_csv(LOG_FILE, index=False)

# 로그 파일을 로드하여 랭킹 표시
def load_log():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=['Name', 'Correct', 'Total'])

# 로그 파일에 기록
def log_prediction(name, correct):
    log_df = load_log()
    if name in log_df['Name'].values:
        idx = log_df.index[log_df['Name'] == name][0]
        log_df.at[idx, 'Total'] += 1
        if correct:
            log_df.at[idx, 'Correct'] += 1
    else:
        new_entry = pd.DataFrame({'Name': [name], 'Correct': [int(correct)], 'Total': [1]})
        log_df = pd.concat([log_df, new_entry], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)

# 랭킹 표시
def show_ranking():
    log_df = load_log()
    if log_df.empty:
        st.write("현재 랭킹 데이터가 없습니다.")
        return
    log_df['Accuracy'] = log_df['Correct'] / log_df['Total']
    log_df.sort_values(by=['Correct', 'Accuracy'], ascending=[False, False], inplace=True)
    st.subheader("랭킹")
    st.write(log_df[['Name', 'Correct', 'Total', 'Accuracy']])

# Streamlit 페이지 구성
st.title("캔들차트 예측 게임")

# 랭킹 표시
show_ranking()

# 사용자 이름 입력
if 'name' not in st.session_state:
    st.session_state['name'] = ''

name = st.text_input("이름을 입력하세요:", value=st.session_state['name'])

if name and not st.session_state.get('name'):
    st.session_state['name'] = name

# 이름이 입력되었을 때만 게임 시작
if st.session_state['name']:
    st.write(f"안녕하세요, {st.session_state['name']}님!")

    # 사용자 입력
    mean = st.number_input("평균 수익률 (mean)", value=0.0, step=0.01, format="%.2f")
    std = st.number_input("표준편차 (std)", value=0.02, step=0.01, format="%.2f")
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
    if 'step' not in st.session_state:
        st.session_state['step'] = 'new_problem'

    # "다음 문제" 버튼을 누르면 새로운 데이터를 생성하기 위해 세션 상태 초기화
    if st.session_state['step'] == 'new_problem':
        # 새로운 데이터를 생성하여 세션 상태에 저장
        np.random.seed()  # 랜덤 시드를 매번 다른 값으로 설정
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
        
        # 새로운 문제를 만든 후 예측 단계로 설정
        st.session_state['step'] = 'predict'

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

    # 예측 단계일 때만 아래 내용 보이기
    if st.session_state['step'] == 'predict':
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
            correct = (prediction == correct_answer)
            if correct:
                st.success("정답입니다!")
            else:
                st.error("오답입니다. 다시 시도해보세요.")
            
            # 결과를 로그 파일에 기록
            log_prediction(st.session_state['name'], correct)

            # 예측 결과 확인 후 새로운 문제 준비 단계로 설정
            st.session_state['step'] = 'review'

    # 예측 결과 확인 후에 "다음 문제" 버튼 표시
    if st.session_state['step'] == 'review':
        if st.button("다음 문제"):
            # 세션 상태를 갱신하여 새로운 문제를 생성하도록 설정
            st.session_state['step'] = 'new_problem'
            # 페이지 리로드 없이도 새로운 데이터를 생성하도록 트리거
            st.experimental_rerun()
