# Reliable and Trustworthy AI — Assignment #1
**Adversarial Attacks on Neural Networks**

---

## Overview

MNIST 및 CIFAR-10 데이터셋에서 CNN 분류 모델을 학습하고,  
**FGSM**과 **PGD** 기반의 Targeted/Untargeted 적대적 공격을 구현한 프로젝트입니다.

---

## Project Structure

```
.
├── test.py              # 학습 + 전체 공격 실행 스크립트
├── mnist.ipynb          # 실험 과정 Jupyter Notebook
├── report.md            # 분석 보고서
├── report.pdf           # 분석 보고서 (PDF)
├── requirements.txt     # Python 의존성 목록
├── results/             # 공격 결과 시각화 PNG 파일
└── data/                # 자동 다운로드되는 데이터셋 (MNIST, CIFAR-10)
```

---

## Requirements

- Python 3.12
- PyTorch 2.5.0 (CUDA 12.1)
- torchvision 0.20.0
- numpy
- matplotlib

패키지 설치:

```bash
pip install torch torchvision matplotlib numpy
```

또는 conda 환경 전체 복원:

```bash
conda create --name <env> --file requirements.txt
```

---

## How to Run

```bash
python test.py
```

실행 시 다음 작업이 순서대로 진행됩니다:

1. MNIST CNN 학습 (10 epochs) → `mnist_cnn.pth` 저장
2. CIFAR-10 CNN 학습 (20 epochs) → `cifar_cnn.pth` 저장
3. 학습 결과 시각화 저장 (`mnist_results.png`, `cifar_results.png`)
4. 아래 4가지 공격을 MNIST / CIFAR-10 각각에 대해 실행:
   - **Problem 1:** Targeted FGSM
   - **Problem 2:** Untargeted FGSM
   - **Problem 3-1:** Targeted PGD
   - **Problem 3-2:** Untargeted PGD
5. 각 공격별 성공률 출력 및 결과 이미지 `results/` 폴더에 저장

> 체크포인트(`.pth`)가 이미 존재하면 학습을 건너뜁니다.

---

## Attack Methods

| Method | Formula | Description |
|:---|:---|:---|
| **FGSM Targeted** | $x_{adv} = x - \varepsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y_{target}))$ | 단일 스텝, 특정 클래스로 유도 |
| **FGSM Untargeted** | $x_{adv} = x + \varepsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y_{true}))$ | 단일 스텝, 오분류 유도 |
| **PGD Targeted** | FGSM 스텝 반복 + L∞ 투영 (−방향) | 반복적 최적화, 특정 클래스 유도 |
| **PGD Untargeted** | FGSM 스텝 반복 + L∞ 투영 (+방향) | 반복적 최적화, 오분류 유도 |

**Hyperparameters:**
- MNIST: `eps=0.3`, `alpha=0.01`, `iters=40`
- CIFAR-10: `eps=0.05`, `alpha=0.01`, `iters=40`

---

## Results

공격 성공률 요약 (각 공격당 최소 100 샘플 평가):

### MNIST (Clean Accuracy: 99.1%)

| Attack | ε=0.05 | ε=0.10 | ε=0.20 | ε=0.30 |
|:---|:---:|:---:|:---:|:---:|
| FGSM Targeted   | 38% | 62% | 87% | 96% |
| FGSM Untargeted | 49% | 78% | 94% | 98% |
| PGD Targeted    | 55% | 81% | 95% | 99% |
| PGD Untargeted  | 68% | 89% | 97% | 100% |

### CIFAR-10 (Clean Accuracy: 83.4%)

| Attack | ε=0.05 | ε=0.10 | ε=0.20 | ε=0.30 |
|:---|:---:|:---:|:---:|:---:|
| FGSM Targeted   | 24% | 47% | 68% | 79% |
| FGSM Untargeted | 38% | 63% | 82% | 91% |
| PGD Targeted    | 41% | 68% | 85% | 93% |
| PGD Untargeted  | 57% | 79% | 93% | 97% |

결과 이미지 예시 (`results/` 폴더):

```
results/
├── mnist_fgsm_targeted_1.png
├── mnist_fgsm_untargeted_1.png
├── mnist_pgd_targeted_1.png
├── mnist_pgd_untargeted_1.png
├── cifar_fgsm_targeted_1.png
...
```

각 이미지는 **원본 | 섭동(×10) | 적대적 예시** 3-panel 형태로 저장됩니다.

---

## Report

자세한 분석 내용은 [`report.md`](./report.md) 또는 [`report.pdf`](./report.pdf)를 참고하세요.
