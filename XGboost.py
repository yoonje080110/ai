import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 1. 데이터 준비
df = pd.read_csv("Student_Study_data.csv")  # 데이터셋 불러오기

# 2. 전처리
df = df.drop(columns=["Name", "Date"])
df = pd.get_dummies(df, columns=["Day", "weekend", "Marital Status", "Your gender?"], drop_first=True)

X = df.drop(columns=["what is your cgpa"])
y = df["what is your cgpa"]

# 3. 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# 4. 목표 CGPA를 위한 최적의 조합 탐색
study_range = np.arange(10, 60, 1)
sleep_range = np.arange(20, 70, 1)
best_combination = None
min_error = float("inf")

for study in study_range:
    for sleep in sleep_range:
        # 기존 입력 샘플 하나를 복제 후 변경
        sample = X.iloc[0].copy()
        sample["study hour"] = study
        sample["your sleep hour?"] = sleep

        pred = model.predict(pd.DataFrame([sample]))[0]
        error = abs(pred - 3.5)

        if error < min_error:
            min_error = error
            best_combination = (study, sleep, pred)

print("Best study/sleep hours for CGPA 3.5:", best_combination)
