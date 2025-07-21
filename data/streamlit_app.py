import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. 데이터 로딩 및 전처리 ---

# 파일 경로 (현재 폴더 내 data 폴더에 있는 파일)
DATA_FILE_PATH = "data/Student_Study_data.csv"

# 로딩
df_raw = pd.read_csv(DATA_FILE_PATH)

# CGPA 문자열을 평균 float 값으로 변환
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

# 불필요한 열 제거
df_processed = df_processed.drop(columns=["Name", "Date"], errors='ignore')

# 범주형 변수 One-hot 인코딩
categorical_cols = ["Day", "Marital Status", "Your gender?"]
for col in categorical_cols:
    if col in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=[col], drop_first=True)

# --- 2. 모델 학습 함수 (캐시 적용) ---

@st.cache_resource
def train_model(df_processed):
    X = df_processed.drop(columns=["what is your cgpa"])
    y = df_processed["what is your cgpa"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    feature_columns = X.columns
    study_min, study_max = int(X["study hour"].min()), int(X["study hour"].max())
    sleep_min, sleep_max = int(X["your sleep hour?"].min()), int(X["your sleep hour?"].max())

    return model, feature_columns, study_min, study_max, sleep_min, sleep_max

# --- 3. 추천 조합 예측 함수 ---

def predict_best_combination(goal_cgpa, model, feature_columns, study_min, study_max, sleep_min, sleep_max):
    study_range = np.arange(study_min, study_max + 1)
    sleep_range = np.arange(sleep_min, sleep_max + 1)
    min_error = float("inf")
    best_combination = None

    base_sample_data = {col: 0 for col in feature_columns}

    for study in study_range:
        for sleep in sleep_range:
            sample = base_sample_data.copy()
            sample["study hour"] = study
            sample["your sleep hour?"] = sleep

            sample_df = pd.DataFrame([sample], columns=feature_columns)

            try:
                pred = model.predict(sample_df)[0]
                error = abs(pred - goal_cgpa)
                if error < min_error:
                    min_error = error
                    best_combination = (study, sleep, pred)
            except Exception:
                continue

    return best_combination

# --- 4. Streamlit UI 구성 ---

st.title("⭐️ 목표 CGPA 맞춤 공부/수면 추천 앱 🌙")
st.write("목표 CGPA를 입력하면, 추천 공부 시간과 수면 시간을 알려줄게요!")

goal_cgpa_input = st.text_input("🎯 목표 CGPA를 입력하세요 (예: 3.5)", key="cgpa_input")

if 'recommendation_result' not in st.session_state:
    st.session_state['recommendation_result'] = ""
if 'error_message' not in st.session_state:
    st.session_state['error_message'] = ""
if 'recommended_sleep_hours' not in st.session_state:
    st.session_state['recommended_sleep_hours'] = None

if st.button("📚 추천 공부/수면 시간 안내 😴"):
    st.session_state['recommendation_result'] = ""
    st.session_state['error_message'] = ""
    st.session_state['recommended_sleep_hours'] = None

    try:
        raw_val = goal_cgpa_input.strip()
        if not raw_val:
            raise ValueError("값을 입력해 주세요. 😉")

        goal_cgpa = float(raw_val)
        if not (0.0 <= goal_cgpa <= 4.5):
            raise ValueError("CGPA는 0.0에서 4.5 사이여야 해요!")

        # 모델 학습 및 정보 가져오기
        model, feature_columns, study_min, study_max, sleep_min, sleep_max = train_model(df_processed)

        # 추천 조합 예측
        best_combination = predict_best_combination(
            goal_cgpa, model, feature_columns, study_min, study_max, sleep_min, sleep_max
        )

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
            st.session_state['recommendation_result'] = "⚠️ 적절한 조합을 찾을 수 없어요. 다른 값을 시도해 주세요."

    except ValueError as ve:
        st.session_state['error_message'] = f"입력 오류: {ve}"
    except Exception as e:
        st.session_state['error_message'] = f"예상치 못한 오류 발생: {e} 😱"

if st.session_state['error_message']:
    st.error(st.session_state['error_message'])
elif st.session_state['recommendation_result']:
    st.info(st.session_state['recommendation_result'])

if st.session_state['recommended_sleep_hours'] is not None:
    st.write("---")
    if st.button("⏰ 추천 수면 타이머 시작 (정보성)"):
        st.success(
            f"👍 **추천 수면 시간은 {st.session_state['recommended_sleep_hours']:.2f}시간**입니다!\n"
            "실시간 타이머는 구현되지 않지만 이 시간만큼 푹 쉬어 주세요! 😊"
        )
else:
    st.info("먼저 위의 버튼을 눌러 추천을 받아보세요! 👆")
