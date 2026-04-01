# Assignment #1 Report: Adversarial Attacks on Neural Networks

**Course:** Reliable and Trustworthy Artificial Intelligence  
**Student:** KimJaeWon  
**Date:** April 1, 2026

---

## 0. Conclusion

본 실험을 통해 다음을 확인하였다.

1. **FGSM**은 단순하지만 적절한 ε에서 강력한 공격력을 보인다. 빠른 계산이 장점이다.
2. **PGD**는 반복적 탐색으로 일관되게 FGSM보다 높은 성공률을 달성한다. 특히 작은 ε에서 차이가 크다.
3. ε이 클수록 공격 성공률이 증가하지만 섭동이 시각적으로 두드러진다. 
4. MNIST보다 CIFAR-10에서 공격이 어려운 것은 모델 복잡도 및 입력 공간의 차원과 관련이 있다.
5. Target 과 Untarget은 부호의 방향이 반대이다.

## 1. Overview

본 보고서는 MNIST 및 CIFAR-10 데이터셋에 학습된 CNN 모델에 대해 FGSM(Fast Gradient Sign Method)과 PGD(Projected Gradient Descent) 기반의 적대적 공격(Adversarial Attack)을 구현하고 실험한 결과를 정리한다.

---

## 2. Model Architecture

### 2.1 MNIST CNN

MNIST 분류를 위해 두 개의 합성곱 블록과 완전연결층으로 구성된 CNN을 설계하였다.

```
Input (1×28×28)
  → Conv2d(1→32, kernel=3) → ReLU → MaxPool2d(2×2)
  → Conv2d(32→64, kernel=3) → ReLU → MaxPool2d(2×2)
  → Flatten → Linear(1600→128) → ReLU → Dropout(0.5) → Linear(128→10)
```

- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** 10
- **Clean Test Accuracy:** **99.1%**

### 2.2 CIFAR-10 CNN

CIFAR-10 분류를 위해 세 개의 합성곱 블록과 BatchNorm을 포함한 CNN을 설계하였다.

```
Input (3×32×32)
  → Conv2d(3→32, kernel=3, pad=1) → BN → ReLU → MaxPool2d(2×2)
  → Conv2d(32→64, kernel=3, pad=1) → BN → ReLU → MaxPool2d(2×2)
  → Conv2d(64→128, kernel=3, pad=1) → BN → ReLU → MaxPool2d(2×2)
  → Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)
```

- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** 20
- **Clean Test Accuracy:** **83.4%**

---

## 3. Attack Implementation

### 3.1 Targeted FGSM

목표 클래스(target)에 대한 Cross-Entropy Loss를 최소화하는 방향으로 한 번의 gradient step을 수행한다.

$$x_{\text{adv}} = x - \varepsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y_{\text{target}}))$$

```python
def fgsm_targeted(wrapped_model, x, target, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(wrapped_model(x_adv), target)
    loss.backward()
    x_adv = x_adv - eps * x_adv.grad.sign()
    return x_adv.detach().clamp(0, 1)
```

### 3.2 Untargeted FGSM

참 레이블에 대한 Loss를 최대화하는 방향으로 한 번의 gradient step을 수행한다.

$$x_{\text{adv}} = x + \varepsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y_{\text{true}}))$$

```python
def fgsm_untargeted(wrapped_model, x, y_true, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(wrapped_model(x_adv), y_true)
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    return x_adv.detach().clamp(0, 1)
```

### 3.3 Targeted PGD

FGSM step을 반복하면서 각 단계마다 원본 이미지로부터 ε-ball 내에 투영(projection)한다.

$$x_{\text{adv}}^{(i)} = \text{clip}_{[0,1]}\Big(\text{clamp}\big(x_{\text{adv}}^{(i-1)} - \alpha \cdot \text{sign}(\nabla_x \mathcal{L}),\ x-\varepsilon,\ x+\varepsilon\big)\Big)$$

```python
def pgd_targeted(wrapped_model, x, target, eps, alpha=0.01, iters=40):
    x_adv = x.clone().detach()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(wrapped_model(x_adv), target)
        loss.backward()
        x_adv = x_adv - alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv.detach(), x + eps), x - eps).clamp(0, 1)
    return x_adv.detach()
```

### 3.4 Untargeted PGD

Targeted PGD와 동일하되, gradient step 방향을 반전한다.

$$x_{\text{adv}}^{(i)} = \text{clip}_{[0,1]}\Big(\text{clamp}\big(x_{\text{adv}}^{(i-1)} + \alpha \cdot \text{sign}(\nabla_x \mathcal{L}),\ x-\varepsilon,\ x+\varepsilon\big)\Big)$$

---

## 4. Experimental Results

### 4.1 Attack Success Rate

공격 성공률은 각 공격 방법에 대해 테스트셋에서 최소 100개의 샘플을 대상으로 측정하였다.
- **Targeted 공격:** 모델이 지정된 target class로 예측하는 경우
- **Untargeted 공격:** 모델이 참 레이블이 아닌 임의의 클래스로 예측하는 경우

#### MNIST (Clean Accuracy: 99.1%)

| Attack Method     | ε = 0.05 | ε = 0.10 | ε = 0.20 | ε = 0.30 |
|:------------------|:--------:|:--------:|:--------:|:--------:|
| FGSM Targeted     | 38.0%    | 62.0%    | 87.0%    | 96.0%    |
| FGSM Untargeted   | 49.0%    | 78.0%    | 94.0%    | 98.0%    |
| PGD Targeted      | 55.0%    | 81.0%    | 95.0%    | 99.0%    |
| PGD Untargeted    | 68.0%    | 89.0%    | 97.0%    | 100.0%   |

#### CIFAR-10 (Clean Accuracy: 83.4%)

| Attack Method     | ε = 0.05 | ε = 0.10 | ε = 0.20 | ε = 0.30 |
|:------------------|:--------:|:--------:|:--------:|:--------:|
| FGSM Targeted     | 24.0%    | 47.0%    | 68.0%    | 79.0%    |
| FGSM Untargeted   | 38.0%    | 63.0%    | 82.0%    | 91.0%    |
| PGD Targeted      | 41.0%    | 68.0%    | 85.0%    | 93.0%    |
| PGD Untargeted    | 57.0%    | 79.0%    | 93.0%    | 97.0%    |

---

## 5. Discussion

### 5.1 공격 방법 간 효과 비교

**PGD > FGSM** 이고, **Untargeted > Targeted** 임을 일관되게 확인할 수 있었다.

- **MNIST vs CIFAR-10:** CIFAR-10의 공격 성공률이 전반적으로 낮다. CIFAR-10 모델이 더 복잡한 decision boundary를 학습하고, 입력 차원(3×32×32)이 크기 때문에 단순한 gradient sign 방향만으로는 오분류를 유도하기 어렵다. 또한 MNIST는 배경이 단순해 perturbation의 효과가 더 직접적으로 나타난다.

### 5.2 ε에 따른 공격 성공률과 시각적 품질의 트레이드오프

ε(perturbation magnitude)은 공격의 핵심 하이퍼파라미터로, 성공률과 지각 불가능성(imperceptibility) 사이의 트레이드오프를 결정한다.

- **ε = 0.05:** perturbation이 육안으로 거의 보이지 않지만 성공률이 낮다. 특히 CIFAR-10 Targeted FGSM에서는 24%에 그쳐 실용적이지 않다.
- **ε = 0.3:** MNIST에서는 PGD Untargeted가 100%에 달하지만, 섭동이 픽셀값의 30%에 해당해 일부 이미지에서 육안으로 노이즈가 식별된다.
- **ε = 0.1~0.2 구간**이 공격 성공률과 시각적 자연스러움의 균형이 가장 적절한 범위로 나타났다.

### 5.3 Wrapped Model 설계에 대한 고찰

---

