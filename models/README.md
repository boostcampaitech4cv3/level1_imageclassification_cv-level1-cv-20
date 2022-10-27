# 지금까지 테스트한 모델 성능들 비교

# BaselineV1

|                     모델명                     | 최고 Validation Accuracy | 걸린 시간 |  사람  | 날짜  |
| :--------------------------------------------: | :----------------------: | :-------: | :----: | :---: |
|      ResNet50 + SGD (PyTorch Pretrained)       |          0.747           |    31m    | 신재영 | 10.26 |
| tf_efficientnet_b7_ns + ADAM (Timm Pretrained) |          0.783           |    14m    | 신재영 | 10.26 |
|      vit_l_32 + ADAM (PyTorch Pretrained)      |          0.648           |   2h 6m   | 신재영 | 10.26 |
|     ResNet18 + ADAM  (PyTorch Pretrained)      |          0.773           |    9m     | 신재영 | 10.26 |
|     ResNet50 + ADAM  (PyTorch Pretrained)      |   0.786 (overfitting)    |    28m    | 신재영 | 10.26 |
|       VGG16 + ADAM (PyTorch Pretrained)        |   0.17 (does not work)   |    0m     | 신재영 | 10.27 |
