

import streamlit as st # Streamlit을 사용하려면 필수!
import xgboost as xgb
import pandas as pd
import numpy as np
import os # 파일 경로 처리를 위해 추가
from sklearn.model_selection import train_test_split

# 파일 경로 정의 (streamlit_app.py와 같은 폴더에 있다고 가정)
DATA_FILE_PATH = "Student_Study_data.csv"

# 캐싱: Streamlit 앱이 실행될 때 한 번만 데이터를 로드하고 전처리하도록 합니다.


df_raw = pd.read_csv(r"C:\Users\qww93\xgboost\data\Student_Study_data.csv")


# CGPA 등급 문자열 → float 변환
def cgpa_str_to_float(val):
    try:
        if isinstance(val, str) and '-' in val:
            left, right = val.split('-')
            return (float(left.strip()) + float(right.strip())) / 2
        else:
            return float(str(val).strip())
    except Exception:
        return np.nan

df_raw["what is your cgpa"] = df_raw["what is your cgpa"].apply(cgpa_str_to_float)
df_processed = df_raw.dropna(subset=["what is your cgpa"])

df_processed = df_processed.drop(columns=["Name", "Date"], errors='ignore')

# 범주형 변수 One-hot 인코딩
categorical_cols = ["Day", "Marital Status", "Your gender?"]
for col in categorical_cols:
    if col in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=[col], drop_first=True)



# 캐싱: 모델 학습은 한 번만 진행되도록 합니다.
@st.cache_resource
def train_model(df_processed):
    # 특징/타겟 변수 정의
    X = df_processed.drop(columns=["what is your cgpa"])
    y = df_processed["what is your cgpa"]

    # XGBoost 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # predict_best_combination 함수에서 사용할 정보들을 함께 반환합니다.
    # 모든 예측을 위한 샘플 생성 시 필요한 컬럼 목록을 저장
    feature_columns = X.columns
    study_min, study_max = int(X["study hour"].min()), int(X["study hour"].max())
    sleep_min, sleep_max = int(X["your sleep hour?"].min()), int(X["your sleep hour?"].max())

    return model, feature_columns, study_min, study_max, sleep_min, sleep_max

# 데이터 로드 및 모델 학습 실행
df = pd.read_csv(r"C:\Users\qww93\xgboost\data\Student_Study_data.csv")
# 7. 목표 CGPA에 맞춘 추천 공부/수면시간
def predict_best_combination(goal_cgpa, model, feature_columns, study_min, study_max, sleep_min, sleep_max):
    study_range = np.arange(study_min, study_max + 1)
    sleep_range = np.arange(sleep_min, sleep_max + 1)
    min_error = float("inf")
    best_combination = None

    # base_row 대신, 학습에 사용된 모든 컬럼을 포함하는 0으로 채워진 기본 샘플을 생성
    base_sample_data = {col: 0 for col in feature_columns}

    for study in study_range:
        for sleep in sleep_range:
            sample = base_sample_data.copy()
            sample["study hour"] = study
            sample["your sleep hour?"] = sleep
            
            # DataFrame을 만들 때 모델 학습 시 사용한 모든 컬럼이 순서대로 들어가도록 합니다.
            sample_df = pd.DataFrame([sample], columns=feature_columns)
            
            try:
                pred = model.predict(sample_df)[0]
                error = abs(pred - goal_cgpa)
                if error < min_error:
                    min_error = error
                    best_combination = (study, sleep, pred)
            except Exception as e:
                # 예측 중 오류가 나도 앱 전체가 멈추지 않도록 예외 처리
                # print(f"예측 중 오류 발생: {e}") # 디버깅용
                continue
    return best_combination
# Streamlit 앱의 타이틀
st.title("⭐️ 목표 CGPA 맞춤 공부/수면 추천 앱 🌙")

st.write("안녕! 목표 CGPA를 입력하면, 그걸 달성하기 위한 공부 시간과 수면 시간을 추천해 줄게요! 😊")

# 사용자 입력을 받기 (Tkinter의 Entry 위젯 역할)
# `key`는 session_state에서 이 입력 값을 구분하는 고유한 이름이에요.
goal_cgpa_input = st.text_input("🎯 목표 CGPA를 입력하세요 (예: 3.5)", key="cgpa_input")

# 초기 세션 상태 설정
# Streamlit은 스크립트가 실행될 때마다 초기화되므로, 값을 유지하려면 session_state를 사용해요.
if 'recommendation_result' not in st.session_state:
    st.session_state['recommendation_result'] = ""
if 'error_message' not in st.session_state:
    st.session_state['error_message'] = ""
if 'recommended_sleep_hours' not in st.session_state:
    st.session_state['recommended_sleep_hours'] = None

# 추천 버튼 (Tkinter의 Button 위젯 역할)
# 버튼이 눌리면 스크립트가 다시 실행되고, 아래 if 문 안의 코드가 실행됩니다.
if st.button("📚 추천 공부/수면 시간 안내 😴"):
    st.session_state['recommendation_result'] = "" # 이전 결과 초기화
    st.session_state['error_message'] = "" # 이전 에러 메시지 초기화
    st.session_state['recommended_sleep_hours'] = None # 추천 수면 시간 초기화

    try:
        raw_val = goal_cgpa_input.strip()
        if not raw_val:
            raise ValueError("값을 입력해 주세요. 깜빡하셨나요? 😉")
        
        goal_cgpa = float(raw_val)
        if not (0.0 <= goal_cgpa <= 4.5):
            raise ValueError("CGPA는 0.0에서 4.5 사이여야 해요! 범위를 다시 확인해 줄래요? 🤔")

        # predict_best_combination 함수 호출
        best_combination = predict_best_combination(goal_cgpa)

        if best_combination is not None:
            study, sleep, predicted = best_combination
            st.session_state['recommended_sleep_hours'] = sleep
            st.session_state['recommendation_result'] = (
                f"🎯 목표 CGPA: **{goal_cgpa}**\n"
                f"📚 추천 공부 시간: **{study} 시간/주**\n"
                f"😴 추천 수면 시간: **{sleep} 시간/주**\n"
                f"📈 예측된 CGPA: **{predicted:.2f}**\n\n"
                f"이 조합이라면 목표 CGPA 달성에 큰 도움이 될 거예요! 🤩"
            )
        else:
            st.session_state['recommendation_result'] = "⚠️ 으음... 적절한 조합을 찾을 수 없어요. 데이터를 확인하거나 다른 CGPA 값을 시도해 볼까요? 🧐"

    except ValueError as ve:
        st.session_state['error_message'] = f"입력 오류: {ve}"
    except Exception as e:
        st.session_state['error_message'] = f"예상치 못한 오류가 발생했어요: {e} 😱"

# 결과 표시 (Tkinter의 result_label 역할)
if st.session_state['error_message']:
    st.error(st.session_state['error_message']) # 에러 메시지는 빨간색으로!
elif st.session_state['recommendation_result']:
    st.info(st.session_state['recommendation_result']) # 결과는 정보성 메시지로!

# 수면 타이머 버튼 및 메시지 (Tkinter의 sleep_timer_button 역할)
# 추천 수면 시간이 있을 때만 버튼이 활성화됩니다.
if st.session_state['recommended_sleep_hours'] is not None:
    st.write("---") # 구분선
    if st.button("⏰ 추천 수면 타이머 시작 (정보성)"):
        # 웹 환경에서는 실제 카운트다운 타이머를 직접 파이썬으로 구현하기 어렵습니다.
        # 이 버튼은 추천 수면 시간을 다시 상기시키는 용도로 사용합니다.
        st.success(
            f"👍 **추천 수면 시간은 {st.session_state['recommended_sleep_hours']:.2f}시간**입니다!\n"
            "실시간 카운트다운 타이머는 웹 브라우저의 JavaScript로 구현해야 해요. "
            "이 메시지는 단순히 정보를 보여주는 역할을 한답니다! 😊"
        )
else:
    # 아직 추천이 없거나 오류가 있을 때 메시지
    st.info("먼저 '추천 공부/수면 시간 안내' 버튼을 눌러 목표 CGPA에 맞는 추천을 받아주세요! 👆")
