# 지금까지 테스트한 모델 성능들 비교

# BaselineV1

|                      모델명                      | 최고 Validation Accuracy | 걸린 시간 |  사람  | 날짜  |
| :----------------------------------------------: | :----------------------: | :-------: | :----: | :---: |
|       ResNet50 + SGD (PyTorch Pretrained)        |          0.747           |    31m    | 신재영 | 10.26 |
|  tf_efficientnet_b7_ns + ADAM (Timm Pretrained)  |          0.783           |    14m    | 신재영 | 10.26 |
|       vit_l_32 + ADAM (PyTorch Pretrained)       |          0.648           |   2h 6m   | 신재영 | 10.26 |
|      ResNet18 + ADAM  (PyTorch Pretrained)       |          0.773           |    9m     | 신재영 | 10.26 |
|      ResNet50 + ADAM  (PyTorch Pretrained)       |   0.786 (overfitting)    |    28m    | 신재영 | 10.26 |
|             Wide_ResNet101_2 + ADAM              |  0.706(?) (overfitting)  |    1h     | 이영섭 | 10.26 |
|           EfficientNet-PyTorch + ADAM            |  0.750(?) (overfitting)  |    1h     | 이영섭 | 10.26 |
|        VGG16 + ADAM (PyTorch Pretrained)         |   0.17 (does not work)   |    0m     | 신재영 | 10.27 |
|   MobileNetV3 Large + ADAM(PyTorch Pretrained)   |          0.748           |    3m     | 윤상준 | 10.27 |
|       vit_b_16 + ADAM (PyTorch Pretrained)       |         0.772(?)         |  4h 40m   | 전지용 | 10.27 |
|       vit_b_16 + ADAM + aug(Cr, Flip, GN)        |          0.714           |    2h     | 전지용 | 10.27 |
|  vit_b_16 + ADAM + aug(Cr, Flip, GN, B) + Focal  |          0.714           |    2h     | 전지용 | 10.27 |
|         Wide_ResNet101_2 + ADAM + Focal          |   0.762 (overfitting)    |    1h     | 이영섭 | 10.27 |
| Wide_ResNet101_2 + ADAM + CrossEntropy + aug(Cr) |          0.728           |    1h     | 이영섭 | 10.27 |
| Wide_ResNet101_2 + ADAM + Focal + aug(Cr, B, GN, Flip) + CosineAnnealingLR | 0.7652 | 3h | 이영섭| 10.29 |

> 데이터 수가 작기 때문에 큰 모델을 사용하면 overfitting이 필연적으로 일어난다. Efficientnet이나 ResNet을 사용해보는게 나쁘지 않을듯 하다

# BaselineV2

### 사람별로 나눈 결과(더 test_set에 가까울 것으로 예상됨)

|                       모델명                       |    최고 F1 Score    | 최고 Val Acc | 걸린 시간(F1 Score) |  사람  | 날짜  |            run name            |
| :------------------------------------------------: | :-----------------: | :----------: | :-----------------: | :----: | :---: | :----------------------------: |
|              ResNet50 + ADAM + F1Loss              |        0.645        |    0.788     |         52m         | 신재영 | 10.27 |              run               |
|              ResNet18 + ADAM + F1Loss              |        0.695        |    0.832     |         17m         | 신재영 | 10.27 |    run_resnet18_adam_f1loss    |
|           ResNet18 + ADAM + CrossEntropy           | 0.740 (overfitting) |    0.849     |         3m          | 신재영 | 10.27 | run_resnet18_adam_crossentropy |
|              ResNet50 + ADAM + Focal               |        0.728        |    0.863     |        1h 6m        | 신재영 | 10.27 |    run_resnet50_adam_focal     |
|              ResNet18 + ADAM + Focal               |        0.735        |     0.86     |         11m         | 신재영 | 10.27 |    run_resnet18_adam_focal     |
|               ResNext + ADAM + Focal               |        0.788        |    0.884     |         54m         | 신재영 | 10.28 |       resnext_adam_focal       |
|           Deit3Base16_224 + ADAM + Focal           |        0.635        |    0.753     |        1h 17        | 신재영 | 10.28 |      dei3base_adam_focal       |
|       ResNet18 + ADAM + Focal+ Augmentations       |        0.803        |    0.914     |         45m         | 신재영 | 10.28 |       resnet18_more_aug        |
| ResNet18 + ADAM + Focal+ Augmentations_with_resize |        0.74         |    0.867     |         12m         | 신재영 | 10.28 |   resnet18_more_aug_resized    |
| ResNet18 + Adam + Focal + Augmentation + face data |        0.792        |     0.91     |       1h 11m        | 신재영 | 10.28 |           tqdm_test            |
|  ResNeXt50 + ADAM + Focal + aug(Cr, Flip, GN, B)   |          -          |   0.92       |  1h 30m   | 전지용 | 10.28 |  Resnext50  |
|  ResNeXt50 + ADAM + Focal + aug(Cr, Flip, GN, B) + CosLR + Face data |          -          |    0.90     |  1h 30m   | 전지용 | 10.28 |  Resnext50  |
|  ResNeXt101 + ADAM + Focal + aug(Cr, Flip, GN, B)  |        0.856        |    0.92     |  2h 30m   | 전지용 | 10.29 |  Resnext101  |
|  ResNeXt101 + ADAM + Focal + aug(Cr, Flip, GN, B) + face data |        0.834        |    0.912     |  2h 30m   | 전지용 | 10.29 |  Resnext101  |
|  ResNeXt101 + ADAM + Focal + aug(Cr, Flip, GN, B) + CosLR + face data |        0.838        |    0.92     |  2h 30m   | 전지용 | 10.29 |  Resnext101  |
|  ResNeXt101 + ADAM + Focal + aug(Cr, Flip, GN, B) + CosLR |        0.841        |    0.915     |  5h    | 전지용 | 10.29 |  Resnext101 + batchsize(16)  |
|  ResNeXt50 + Multihead + ADAM + Focal + aug(Cr, Flip, GN, B) + CosLR |        0.8559        |    0.92     |   5h    | 전지용 | 10.30 |  Resnextmultihead + epoch(70)  |
|  ResNeXt50 + Multihead + ADAM + Focal + aug(Cr, Flip, GN, B) + CosLR + face data |        0.83        |    0.92     |   5h    | 전지용 | 10.30 |  Resnextmultihead + epoch(70)  |
|  ResNeXt50 + Multihead + agelayer + ADAM + Focal + aug(Cr, Flip, GN, B) + CosLR |        0.8555        |    0.921     |   5h    | 전지용 | 10.30 |  Resnextmultiagehead + epoch(70)  |
|  ResNeXt50 + Multihead + agelayer + Nestrov SGD + Focal + aug(Cr, Flip, GN, B) + StepLR |        0.8295        |    0.897     |   2h    | 전지용 | 10.31 |  Resnextmultiagehead + epoch(50)  |
|  ResNeXt50 + Multihead + agelayer + RMSprop + Focal + aug(Cr, Flip, GN, B) + StepLR |        0.8101        |    0.896     |   2h    | 전지용 | 10.31 |  Resnextmultiagehead + epoch(50)  |
|  ResNeXt50 + Multihead + agelayer + ADAM + Focal + aug(Cr, Flip, GN, B) + StepLR + earlystop + K-Fold |   0.7784 ~ 0.5519      |    0.874 ~ 0.4262   |   3h    | 전지용 | 11.01 |  kfoldEnsemble  |
|  ResNeXt50 + Multihead + agelayer + ADAM + aug(Cr, Flip, GN, B) + StepLR + sampler + F1 loss + earlystop + K-Fold |   0.7905 ~ 0.5748      |    0.883 ~ 0.676   |   3h    | 전지용 | 11.01 |  kfoldEnsemble  |
|  ResNeXt50 + Multihead + agelayer + ADAM + aug(Cr, Flip, GN, B) + StepLR + sampler + Focal + earlystop + K-Fold |   0.7472 ~ 0.565      |    0.876 ~ 0.7032   |   3h    | 전지용 | 11.01 |  kfoldEnsemble  |
|  ResNeXt50 + Multihead + agelayer + ADAM + aug(Cr, Flip, GN, B, Hue, Scale) + ExponentialLR + Focal+LADE + earlystop + K-Fold |   0.7938 ~ 0.5817      |    0.886 ~ 0.737   |   3h    | 전지용 | 11.01 |  kfoldEnsemble  |
|                                                    |                     |              |                     |        |       |                                |


> 최고 결과 이후에 overfitting이 일어나면 overfitting이라고 써준다

# AgeOnlyTraining

|                       모델명                       | 최고 F1 Score | 최고 Val Acc | 걸린 시간(F1 Score) |  사람  | 날짜  | run name  |
| :------------------------------------------------: | :-----------: | :----------: | :-----------------: | :----: | :---: | :-------: |
| ResNet18+ADAM+Focal+Augmentation+NoResize+FaceData |     0.824     |    0.928     |       1h 47m        | 신재영 | 10.28 | tqdm_test |
|                                                    |               |              |                     |        |       |           |