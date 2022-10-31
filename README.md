# [CV] 마스크 착용 상태 분류
부스트캠프 AI TECH4기 CV-20 대회 준비 과정

## 👉대회 개요
COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

## 🔥레이블 특징  
총 3개의 클래스의 속성에 따라 18개의 클래스로 분류  
- Mask : Wear, Incorrect, Not Wear (세 가지 카테고리)
- Gender: Male, Female (두 가지 카테고리)
- Age : <30, >=30 and <60, >=60(세 가지 카테고리)
<img src="https://user-images.githubusercontent.com/79644050/197935770-883e5583-671d-464b-bdbb-921b62df083a.png" width ="60%" height="80%"/>

## 📌 폴더 세부사항
|  폴더 명  | 폴더 내용 | 링크 |
| :------: | :----: | :--- |
| EDA           | 팀원별 탐색적 데이터 분석 및 요약                               | [링크](https://github.com/boostcampaitech4cv3/level1_imageclassification_cv-level1-cv-20/tree/main/EDA)|
| Preproessing  | 학습 과정에서의 전처리 적용 코드                                | [링크](https://github.com/boostcampaitech4cv3/level1_imageclassification_cv-level1-cv-20/tree/main/Preprocessing)|
| Submission    | 각 일자별 리더보드 제출 관련 기록                               | [링크](https://github.com/boostcampaitech4cv3/level1_imageclassification_cv-level1-cv-20/tree/main/Submission)|
| Baseline      | 리더보드 용 Baseline + 새로운 모델 작성 시 필요한 base code     | [링크](https://github.com/boostcampaitech4cv3/level1_imageclassification_cv-level1-cv-20/tree/main/baseline)|
| Models        | 개인별 시도한 코드 기록                                        | [링크](https://github.com/boostcampaitech4cv3/level1_imageclassification_cv-level1-cv-20/tree/main/models)|

## 📌 모델별 성능
|   기준   | 작성자 | 모델 | 전처리 | 후처리 | 하이퍼 파라미터 | 성능 |
| :------: | :----: | :--: | :----: | :----: | :-------------: | :--- |
| Baseline |   -    |      |        |        |                 |      |
|  Custom  |   -    |      |        |        |                 |      |

## 📌모델 아이디어
- 세가지 클래스의 속성에 따라 최종 클래스가 결정되므로 각각의 클래스를 예측하는 모델 제작해보기
- class imbalance 문제가 존재하기 때문에 GAN 모델 등을 활용하여 imbalance 문제 해결
- 모델 학습 과정에서 resizing이 필요한 경우 Face Detection을 통해 해당 부분을 crop 후 resizing 진행하여 중요 feature의 손실을 최소화
- 얼굴 요소를 탐지하여 얼굴 요소에 따른 마스크 착용 여부 검사하기

## 📌 개인별 대회 활동내역
| 이영섭 | 윤상준 | 서원준 | 전지용 | 신재영 |
| :----: | :----: | :----: | :----: | :-----:|
|        | [개인 회고](https://github.com/boostcampaitech4cv3/level1_imageclassification_cv-level1-cv-20/blob/main/Retrospect/%EC%9C%A4%EC%83%81%EC%A4%80.md)|        |        |        |

## 📌 주의 사항
* 매일 활동한 내역 정리해서 github에 올리기 ✨
* 학습 과정에서 변경한 내용 모두 정리해서 깃허브 변경하기
* 파일명은 다음과 같은 규칙을 따르기 {작업}\_{세부작업}\_{이름}.ipynb

## 📌 참고 자료
* [Moving Window Regression: A Novel Approach to Ordinal Regression](https://github.com/nhshin-mcl/mwr)
* [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
* [How can i process multi loss in pytorch?](https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch)
* [Age And Gender Classification](https://paperswithcode.com/task/age-and-gender-classification)
