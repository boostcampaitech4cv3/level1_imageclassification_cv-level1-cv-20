# 📌 전처리 파일
|  작성자   | 전처리 방식 | 코드 |
| :------: | :--------: | :--- | 
|  윤상준   |  Grab cut을 통한 배경제거 | [코드](https://github.com/boostcampaitech4cv3/level1_imageclassification_cv-level1-cv-20/blob/main/Preprocessing/GrabCut_Background_Preprocessing.ipynb)     |      
|  윤상준   |  train.csv 변형 | [코드](https://github.com/boostcampaitech4cv3/level1_imageclassification_cv-level1-cv-20/blob/main/Preprocessing/train_converter.ipynb)|
|  윤상준   |  Unet 기반 Rembg 라이브러리를 통한 배경제거 | [코드](https://github.com/boostcampaitech4cv3/level1_imageclassification_cv-level1-cv-20/blob/main/Preprocessing/Unet_bg_removal.ipynb)|   


# 📌 주의사항
* grabcut의 경우 전통적인 CV 알고리즘을 사용한 관계로 train 데이터셋에 모두 적용하는데 6시간 이상 소요
* Rembg 사용 시 ONNX Format을 GPU로 가속하지 않을 경우 7시간 정도 소요

# 📌 참고자료
[Remove background using UNET](https://github.com/danielgatis/rembg)
