import tkinter as tk
import xgboost as xgb
import pandas as pd
import numpy as np
import threading
import time
import csv
from sklearn.model_selection import train_test_split

# 1. 데이터셋 불러오기
df = pd.read_csv(r"C:\Users\qww93\xgboost\data\Student_Study_data.csv")


# 2. CGPA 등급 문자열 → float 변환
def cgpa_str_to_float(val):
    try:
        if isinstance(val, str) and '-' in val:
            left, right = val.split('-')
            return (float(left.strip()) + float(right.strip())) / 2
        else:
            return float(str(val).strip())
    except Exception:
        return np.nan

df["what is your cgpa"] = df["what is your cgpa"].apply(cgpa_str_to_float)
df = df.dropna(subset=["what is your cgpa"])

# 3. 불필요 컬럼 제거
df = df.drop(columns=["Name", "Date"], errors='ignore')

# 4. 범주형 변수 One-hot 인코딩
categorical_cols = ["Day", "Marital Status", "Your gender?"]
for col in categorical_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# 5. 특징/타겟 변수 정의
X = df.drop(columns=["what is your cgpa"])
y = df["what is your cgpa"].astype(np.float32)

# 6. XGBoost 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# 7. 목표 CGPA에 맞춘 추천 공부/수면시간
def predict_best_combination(goal_cgpa):
    study_min, study_max = int(X["study hour"].min()), int(X["study hour"].max())
    sleep_min, sleep_max = int(X["your sleep hour?"].min()), int(X["your sleep hour?"].max())
    study_range = np.arange(study_min, study_max + 1)
    sleep_range = np.arange(sleep_min, sleep_max + 1)
    min_error = float("inf")
    best_combination = None

    base_row = X.iloc[0].copy()
    for study in study_range:
        for sleep in sleep_range:
            sample = base_row.copy()
            sample["study hour"] = study
            sample["your sleep hour?"] = sleep
            sample_df = pd.DataFrame([sample], columns=X.columns)
            try:
                pred = model.predict(sample_df)[0]
                error = abs(pred - goal_cgpa)
                if error < min_error:
                    min_error = error
                    best_combination = (study, sleep, pred)
            except Exception:
                continue
    return best_combination

# 8. Tkinter GUI Setup
app = tk.Tk()
app.title("목표 CGPA 맞춤 공부/수면 추천 & 수면 타이머")

cgpa_label = tk.Label(app, text="목표 CGPA를 입력하세요 (예: 3.5)")
cgpa_label.pack()
cgpa_input = tk.Entry(app)
cgpa_input.pack()

result_label = tk.Label(app, text="", wraplength=400)
result_label.pack()

sleep_timer_label = tk.Label(app, text="", wraplength=400, fg="blue")
sleep_timer_label.pack()

# 추천 결과 보관 (수면 타이머 시작 시 사용)
recommend_sleep_time = {'hours': None}

def run_sleep_timer(hours):
    seconds = int(hours * 3600)
    sleep_timer_label.config(text=f"추천 수면시간 {hours:.2f}시간 타이머 시작!")
    # 초 단위로 하나씩 깎임 (시각적 업데이트)
    for sec in range(seconds, -1, -1):
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        sleep_timer_label.config(text=f"남은 수면 시간: {h:02d}:{m:02d}:{s:02d}")
        app.update_idletasks()
        time.sleep(1)
    sleep_timer_label.config(text=f"추천 수면 {hours:.2f}시간 종료! 기상 시간입니다.")

def on_predict_click():
    try:
        raw_val = cgpa_input.get().strip()
        if not raw_val:
            raise ValueError("값을 입력해 주세요.")
        goal_cgpa = float(raw_val)
        if not (0.0 <= goal_cgpa <= 4.5):
            raise ValueError("CGPA는 0.0에서 4.5 사이여야 합니다.")

        best = predict_best_combination(goal_cgpa)
        if best is not None:
            study, sleep, predicted = best
            recommend_sleep_time['hours'] = sleep
            result = (
                f"🎯 목표 CGPA: {goal_cgpa}\n"
                f"📚 추천 공부 시간: {study} 시간/주\n"
                f"😴 추천 수면 시간: {sleep} 시간/주\n"
                f"📈 예측된 CGPA: {predicted:.2f}\n\n"
                f"아래에서 '추천 수면 타이머 시작'을 누르면 {sleep:.0f}시간 동안 수면을 측정합니다."
            )
        else:
            recommend_sleep_time['hours'] = None
            result = "⚠️ 적절한 조합을 찾을 수 없습니다.\n데이터 및 목표 값을 확인하세요."

        result_label.config(text=result)
        sleep_timer_label.config(text="")  # 타이머 메시지 초기화

    except ValueError as ve:
        result_label.config(text=f"입력 오류: {ve}")
        recommend_sleep_time['hours'] = None
    except Exception as e:
        result_label.config(text=f"오류 발생: {e}")
        recommend_sleep_time['hours'] = None

def on_sleep_timer_click():
    hours = recommend_sleep_time['hours']
    if hours is None:
        sleep_timer_label.config(text="먼저 목표 CGPA를 입력해 추천을 받아야 수면 타이머를 시작할 수 있습니다.")
        return
    # 별도 스레드에서 sleep 타이머 실행 (GUI 멈춤 방지)
    threading.Thread(target=run_sleep_timer, args=(hours,), daemon=True).start()

predict_button = tk.Button(app, text="추천 공부/수면 시간 안내", command=on_predict_click)
predict_button.pack()

sleep_timer_button = tk.Button(app, text="추천 수면 타이머 시작", command=on_sleep_timer_click)
sleep_timer_button.pack()

app.mainloop()