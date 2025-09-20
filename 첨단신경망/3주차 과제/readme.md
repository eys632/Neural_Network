# AlexNet 경량화 실험 (CIFAR-10)

## 1. 과제 목표
CIFAR-10 데이터셋을 분류하는 AlexNet 모델의 파라미터가 5,800만 개 이상으로 매우 크기 때문에, 모델 경량화를 위해 파라미터 수를 줄이는 실험을 진행한다.

### 적용 조건
1. 5개의 모든 `Convolution` 레이어에 **필터 크기 (3x3)**, **필터 수 10**을 동일하게 적용
2. 첫 번째 `Fully Connected` 레이어에 **노드(뉴런) 수 1000개** 적용
3. 두 번째 `Fully Connected` 레이어에 **노드(뉴런) 수 100개** 적용
4. **10 에폭(Epochs)** 동안 훈련 진행

## 2. 모델 구조
요구사항에 따라 수정한 모델의 전체 구조는 다음과 같다. `BatchNormalization` 레이어를 포함하여 총 파라미터 수는 **466,230**개로 원본 AlexNet에 비해 크게 감소했다.

### 2. 모델 구조 (summary)
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 57, 57, 10)        280       
                                                                 
 batch_normalization (BatchN  (None, 57, 57, 10)       40        
 ormalization)                                                   
                                                                 
 activation (Activation)     (None, 57, 57, 10)        0         
                                                                 
 max_pooling2d (MaxPooling2  (None, 28, 28, 10)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 28, 28, 10)        910       
                                                                 
 batch_normalization_1 (Batc  (None, 28, 28, 10)       40        
 hNormalization)                                                 
                                                                 
 activation_1 (Activation)   (None, 28, 28, 10)        0         
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 13, 13, 10)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 13, 13, 10)        910       
                                                                 
 batch_normalization_2 (Batc  (None, 13, 13, 10)       40        
 hNormalization)                                                 
                                                                 
 activation_2 (Activation)   (None, 13, 13, 10)        0         
                                                                 
 conv2d_3 (Conv2D)           (None, 13, 13, 10)        910       
                                                                 
 batch_normalization_3 (Batc  (None, 13, 13, 10)       40        
 hNormalization)                                                 
                                                                 
 activation_3 (Activation)   (None, 13, 13, 10)        0         
                                                                 
 conv2d_4 (Conv2D)           (None, 13, 13, 10)        910       
                                                                 
 batch_normalization_4 (Batc  (None, 13, 13, 10)       40        
 hNormalization)                                                 
                                                                 
 activation_4 (Activation)   (None, 13, 13, 10)        0         
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 6, 6, 10)          0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 360)               0         
                                                                 
 dense (Dense)               (None, 1000)              361000    
                                                                 
 dropout (Dropout)           (None, 1000)              0         
                                                                 
 dense_1 (Dense)             (None, 100)               100100    
                                                                 
 dropout_1 (Dropout)         (None, 100)               0         
                                                                 
 dense_2 (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 466230 (1.78 MB)
Trainable params: 466130 (1.78 MB)
Non-trainable params: 100 (400.00 Byte)
_________________________________________________________________
```

## 3. 파라미터 수 계산 과정
### Convolutional Layers
| layer | 필터 Width | 필터 Height | 필터 Channel | bias | 필터 수 | 파라미터 수 계산 과정 | 파라미터 수 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Conv 1** | 3 | 3 | 3 | 1 | 10 | `((3*3*3)+1)*10` | 280 |
| **Conv 2** | 3 | 3 | 10 | 1 | 10 | `((3*3*10)+1)*10` | 910 |
| **Conv 3** | 3 | 3 | 10 | 1 | 10 | `((3*3*10)+1)*10` | 910 |
| **Conv 4** | 3 | 3 | 10 | 1 | 10 | `((3*3*10)+1)*10` | 910 |
| **Conv 5** | 3 | 3 | 10 | 1 | 10 | `((3*3*10)+1)*10` | 910 |

### Fully Connected Layers
| layer | 입력 뉴런 | 출력 뉴런 | bias | 파라미터 수 계산 과정 | 파라미터 수 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FC 1** | 360 | 1000 | 1000 | `(360*1000)+1000` | 361,000 |
| **FC 2** | 1000 | 100 | 100 | `1000*100+100` | 100,100 |
| **FC 3** | 100 | 10 | 10 | `100*10+10` | 1,010 |

### 총 파라미터
| 종류 | 파라미터 수 |
| :--- | :--- |
| **학습 param** | 466,130 |
| **비학습 param** | 100 |
| **총 파라미터** | 466,230 |

## 4. 훈련 결과
### 학습 과정 로그
```
Epoch 1/10
1406/1406 [==============================] - 29s 17ms/step - loss: 2.3025 - accuracy: 0.1094 - val_loss: 2.2866 - val_accuracy: 0.1284
Epoch 2/10
1406/1406 [==============================] - 25s 17ms/step - loss: 2.2843 - accuracy: 0.1298 - val_loss: 2.2516 - val_accuracy: 0.1953
... (중략) ...
Epoch 10/10
1406/1406 [==============================] - 26s 17ms/step - loss: 1.8340 - accuracy: 0.3228 - val_loss: 1.6913 - val_accuracy: 0.3934

--- 최종 평가 ---
312/312 [==============================] - 2s 7ms/step - loss: 1.6923 - accuracy: 0.3965
Test Loss: 1.6923
Test Accuracy: 0.3965
```

### 결과 시각화

<img width="1189" height="490" alt="Training History" src="https://github.com/user-attachments/assets/10cc8a6f-bb77-43a9-8068-46811e2abfc8" />


## 5. 결과 분석 및 고찰
- **최종 성능**: 10 에폭 훈련 후 Test Accuracy 약 **39.7%** 달성.
- **분석**: 훈련 및 검증 데이터 모두에서 손실이 꾸준히 감소하고 정확도가 꾸준히 증가하는 추세를 보임. 두 지표 간의 큰 격차가 없어 **과적합(Overfitting)보다는 과소적합(Underfitting) 상태**에 가까움. 이는 모델이 10 에폭이 끝난 시점에도 여전히 학습할 여지가 남았음을 의미한다.
- **원인**: 상대적으로 낮은 최종 정확도의 주된 원인은 **모델의 표현력(Capacity) 부족**으로 판단됨. 모든 합성곱 레이어의 필터 수를 10개로 제한하여 모델의 파라미터 수가 크게 줄었고, 이로 인해 CIFAR-10 데이터셋의 복잡한 특징을 모두 학습하기에는 한계가 있었을 것으로 보임.
- **향후 개선 방안**: 필터 수를 점진적으로 늘리거나(예: 16, 32, 64), 더 많은 에폭 동안 훈련을 진행하여 성능 향상을 기대해 볼 수 있음.

