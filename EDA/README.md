# [CV] 마스크 착용 상태 분류- EDA
학습데이터 EDA 과정
## 전체 데이터셋 구성
1. train 데이터셋 (사람 별 7장(마스크 착용:5/마스크 잘못 착용: 1/마스크 미착용:1)으로 구성된 2700명의 총 18,900개의 데이터)
2.  eval 데이터셋(12600개의 사진으로 1800명의 사진)
3.  tot: 4,500명의 31,500개의 데이터셋

## 👉train.csv 데이터 분석 결과
### 1. Age 데이터 분석 결과
- Overview : 주어진 데이터 확인 시 U자형의 데이터 분포를 보여줌
<img src="https://user-images.githubusercontent.com/79644050/197940095-c3ad6651-b5a1-411d-9973-9db2aec294f5.png" width ="60%" height="80%"/>
- 구간으로 나눠서 분석한 경우
<img src="https://user-images.githubusercontent.com/79644050/197940969-758dc20a-6314-4d71-8eca-990767aee2ef.png" width ="60%" height="80%"/>
데이터 구성비가 47.4%, 45.4%, 7.1%로 class imbalance 문제 발견

### 2. Gender 데이터 분석 결과
- Overview: 여성 데이터(1658명)가 남성데이터 보다 더 많이 존재(1042명)
<img src="https://user-images.githubusercontent.com/79644050/197941625-d274ee1f-7f0f-43e4-a884-0824b43c0f94.png" width ="60%" height="80%"/>

### 3. Age & Gender 데이터 분석 결과
 - 30대 미만 -> 총계: 1,272명 남성: 540명, 여성: 732명 (42.4%, 57.5%)
 - 30대 이상 60대 미만 -> 총계: 1227명 남성: 410명, 여성: 817명(33.4%, 66.5%)
 - 60대 이상 -> 총계: 192명 남성: 83명, 여성: 109명(43.2%, 56.7%)
<img src="https://user-images.githubusercontent.com/79644050/197942781-6afc1043-9a61-4cf5-843c-8a21dad57d24.png" width ="80%" height="100%"/>


## 🔥EDA에 기반한 모델 학습 진행방향

📌👉🔥✨
