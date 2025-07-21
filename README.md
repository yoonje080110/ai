### Features

- 저녁에 폰을 끄면, 다시 아침에 일어난(수면시간) 시간 측정
- 연령별로 적절한 수면시간 기준. User의 수면시간과 비교
- 알맞게 자면 소정의 기프티콘
- RNN or LSTM  사용시 좋음 -> 이는 시계열 데이터 필요
    (한 사람의 수면시간과 성적데이터를 몇달어치를 싹 분석하는느낌)

-> 여러 사람의 데이터를 수집하여 분석은 XGBoost/MLP 회귀 모델이 좋음
목표 수면시간 + 공부시간 산출 : Grid Search or Scipy optimizer

### Target
고딩

### Thinking
<수면시간>
영향 : 카페인, 학년, 지역 등

Data(이게 제일 찾기 빡세긴함)

- OECD(경제협력개발기구) : 청소년(13세 ~ 18세)에게 8~10h의 수면시간 권장
- 2024년 평균 청소년 남 : 6.1h,  여 : 5.6h 수면(평균값만 나와잇음)  [자료](https://data.seoul.go.kr/dataList/10961/S/2/datasetView.do)
- Student Study Performance [Keggle]( https://www.kaggle.com/datasets/nabilajahan/student-study-performance/data
) (기준 : CPGA)
- CGPA를 등급으로 환산필요. ([정확도](https://www.cgpa2percentage.com/#google_vignette) : 모르겟노 ?)


### Blueprint

- Python을 이용해 XGBoost 모델 정의 및 학습
- User에게 내일 공부할 시간, 수면시간 등 데이터를 받아 모델에 넣으면 예상 성적 산출 가능
- 목표의 성적을 위한 최적의 수면시간 찾기
- app에서 수면시간 받아오기
- Server or 수동으로 모델에 입력하여 최적의 시간 계산.

추천 Tool
이정도 기능이면… 휴대폰 어플보다는 개인 노트북에 프로그램 만들어서 간단히 수면시간 넣어서 확인하는 것이 나을지도 ?
휴대폰 어플로 하면 99%이상 서버 필요. 모델의 계산값을 휴대폰에 받아와야 하기 때문


## 사용 방법
1. Win + R 키를 누르고, cmd 입력해서 들어가기
2.
```
git clone https://github.com/Eligae/xgboost.git
pip install xgboost pandas scikit-learn numpy thinker
cd xgboost
code .
```
순서대로 한 줄씩 cmd에 입력
cgpa 등급 %비율 찾아보면서 몇 퍼센트 넣으면 몇 등급 나오는지 직접 해보면서 찾아보기
