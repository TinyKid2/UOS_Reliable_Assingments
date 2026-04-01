import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# 1. 모델 정의
# ──────────────────────────────────────────────

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))


# ──────────────────────────────────────────────
# 2. 데이터 로더
# ──────────────────────────────────────────────

def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = torchvision.datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_set  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_cifar_loaders(batch_size=64):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ──────────────────────────────────────────────
# 3. 학습 / 평가
# ──────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def train_mnist(epochs=10):
    print("=== MNIST 학습 시작 ===")
    train_loader, test_loader = get_mnist_loaders()
    model = MNIST_CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch:2d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"test_loss={te_loss:.4f}  test_acc={te_acc:.4f}")

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("모델 저장: mnist_cnn.pth\n")
    return model


def train_cifar(epochs=20):
    print("=== CIFAR-10 학습 시작 ===")
    train_loader, test_loader = get_cifar_loaders()
    model = CIFAR_CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch:2d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"test_loss={te_loss:.4f}  test_acc={te_acc:.4f}")

    torch.save(model.state_dict(), "cifar_cnn.pth")
    print("모델 저장: cifar_cnn.pth\n")
    return model


# ──────────────────────────────────────────────
# 4. 학습 결과 시각화
# ──────────────────────────────────────────────

def visualize_results(model, dataset_name="mnist", n=10):
    if dataset_name == "mnist":
        _, test_loader = get_mnist_loaders(batch_size=n)
    else:
        _, test_loader = get_cifar_loaders(batch_size=n)

    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    with torch.no_grad():
        preds = model(images).argmax(1)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))
    for i in range(n):
        img = images[i].cpu()
        if dataset_name == "mnist":
            img = img.squeeze().numpy()
            axes[i].imshow(img, cmap="gray")
        else:
            mean = np.array([0.4914, 0.4822, 0.4465])
            std  = np.array([0.2470, 0.2435, 0.2616])
            img = img.permute(1, 2, 0).numpy() * std + mean
            axes[i].imshow(np.clip(img, 0, 1))
        color = "blue" if preds[i] == labels[i] else "red"
        axes[i].set_title(f"pred={preds[i].item()}\ntrue={labels[i].item()}", color=color, fontsize=8)
        axes[i].axis("off")
    plt.suptitle(f"{dataset_name.upper()} 예측 결과 (파랑=정답, 빨강=오답)")
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_results.png")
    plt.show()


# ──────────────────────────────────────────────
# 5. Wrapped model (역정규화 포함)
# ──────────────────────────────────────────────

class MNISTWrappedModel(nn.Module):
    """[0,1] 범위 입력을 받아 정규화 후 분류"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mean = torch.tensor([0.1307]).view(1, 1, 1, 1).to(DEVICE)
        self.std  = torch.tensor([0.3081]).view(1, 1, 1, 1).to(DEVICE)

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


class CIFARWrappedModel(nn.Module):
    """[0,1] 범위 입력을 받아 정규화 후 분류"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(DEVICE)
        self.std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(DEVICE)

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


# ──────────────────────────────────────────────
# 6. 적대적 공격
# ──────────────────────────────────────────────

def fgsm_targeted(wrapped_model, x, target, eps):
    """Targeted FGSM: x_adv = x - eps * sign(∇_x L(f(x), target))"""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(wrapped_model(x_adv), target)
    loss.backward()
    x_adv = x_adv - eps * x_adv.grad.sign()
    return x_adv.detach().clamp(0, 1)


def fgsm_untargeted(wrapped_model, x, y_true, eps):
    """Untargeted FGSM: x_adv = x + eps * sign(∇_x L(f(x), y_true))"""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(wrapped_model(x_adv), y_true)
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    return x_adv.detach().clamp(0, 1)


def pgd_targeted(wrapped_model, x, target, eps, alpha=0.01, iters=40):
    """Targeted PGD: 반복적으로 target 클래스로 유도"""
    x_adv = x.clone().detach()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(wrapped_model(x_adv), target)
        loss.backward()
        x_adv = x_adv - alpha * x_adv.grad.sign()
        # L∞ 투영
        x_adv = torch.max(torch.min(x_adv.detach(), x + eps), x - eps).clamp(0, 1)
    return x_adv.detach()


def pgd_untargeted(wrapped_model, x, y_true, eps, alpha=0.01, iters=40):
    """Untargeted PGD: 반복적으로 참 레이블에서 멀어지도록 유도"""
    x_adv = x.clone().detach()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(wrapped_model(x_adv), y_true)
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv.detach(), x + eps), x - eps).clamp(0, 1)
    return x_adv.detach()


# ──────────────────────────────────────────────
# 7. 공격 시각화 헬퍼
# ──────────────────────────────────────────────

def _to_display_mnist(tensor):
    return tensor.squeeze().cpu().numpy()


def _to_display_cifar(tensor):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2470, 0.2435, 0.2616])
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    return np.clip(img * std + mean, 0, 1)


def visualize_attack(orig, adv, orig_pred, adv_pred, true_label,
                     dataset_name="mnist", idx=0, attack_name="attack"):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    pert = (adv - orig).abs()

    if dataset_name == "mnist":
        to_disp = _to_display_mnist
        cmap = "gray"
    else:
        to_disp = _to_display_cifar
        cmap = None

    imgs = [to_disp(orig), to_disp(pert * 10), to_disp(adv)]
    titles = [
        f"Original\ntrue={true_label} pred={orig_pred}",
        "Perturbation (×10)",
        f"Adversarial\npred={adv_pred}",
    ]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img if cmap is None else img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle(f"{dataset_name.upper()} {attack_name} (sample {idx})")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    path = f"results/{dataset_name}_{attack_name}_{idx}.png"
    plt.savefig(path)
    plt.show()
    print(f"  저장: {path}")


# ──────────────────────────────────────────────
# 8. 공격 실행 함수
# ──────────────────────────────────────────────

def run_attack(dataset_name, attack_fn, attack_name, eps, alpha=0.01, iters=40,
               target_label=None, max_samples=5):
    print(f"\n--- {dataset_name.upper()} {attack_name} (eps={eps}) ---")

    # 모델 및 데이터 준비
    if dataset_name == "mnist":
        model = MNIST_CNN().to(DEVICE)
        ckpt  = "mnist_cnn.pth"
        _, test_loader = get_mnist_loaders(batch_size=1)
        wrapped = MNISTWrappedModel(model)
        denorm_mean = torch.tensor([0.1307]).view(1, 1, 1, 1).to(DEVICE)
        denorm_std  = torch.tensor([0.3081]).view(1, 1, 1, 1).to(DEVICE)
    else:
        model = CIFAR_CNN().to(DEVICE)
        ckpt  = "cifar_cnn.pth"
        _, test_loader = get_cifar_loaders(batch_size=1)
        wrapped = CIFARWrappedModel(model)
        denorm_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(DEVICE)
        denorm_std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(DEVICE)

    if not os.path.exists(ckpt):
        print(f"  체크포인트 {ckpt} 없음. 먼저 학습을 실행하세요.")
        return

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    success, total = 0, 0

    for images, labels in test_loader:
        if success >= max_samples:
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 정규화된 이미지 → [0,1] 범위로 변환
        x_01 = (images * denorm_std + denorm_mean).clamp(0, 1)

        # 원본 예측
        with torch.no_grad():
            orig_pred = model(images).argmax(1).item()

        if orig_pred != labels.item():
            continue  # 원본부터 틀린 샘플 제외

        # 공격 실행
        if target_label is not None:
            tgt = torch.tensor([target_label], device=DEVICE)
            if tgt.item() == labels.item():
                continue
            if "pgd" in attack_name:
                x_adv = attack_fn(wrapped, x_01, tgt, eps, alpha, iters)
            else:
                x_adv = attack_fn(wrapped, x_01, tgt, eps)
        else:
            if "pgd" in attack_name:
                x_adv = attack_fn(wrapped, x_01, labels, eps, alpha, iters)
            else:
                x_adv = attack_fn(wrapped, x_01, labels, eps)

        with torch.no_grad():
            adv_pred = wrapped(x_adv).argmax(1).item()

        total += 1
        attack_success = (target_label is not None and adv_pred == target_label) or \
                         (target_label is None and adv_pred != labels.item())

        if attack_success:
            success += 1
            visualize_attack(x_01, x_adv, orig_pred, adv_pred, labels.item(),
                             dataset_name, success, attack_name)

    if total > 0:
        print(f"  공격 성공률: {success}/{total} = {success/total*100:.1f}%")
    else:
        print("  유효한 샘플 없음")


# ──────────────────────────────────────────────
# 9. 메인
# ──────────────────────────────────────────────

def main():
    # ── 학습 ──────────────────────────────────
    if not os.path.exists("mnist_cnn.pth"):
        mnist_model = train_mnist(epochs=10)
    else:
        print("mnist_cnn.pth 발견. 학습 생략.")
        mnist_model = MNIST_CNN().to(DEVICE)
        mnist_model.load_state_dict(torch.load("mnist_cnn.pth", map_location=DEVICE))

    if not os.path.exists("cifar_cnn.pth"):
        cifar_model = train_cifar(epochs=20)
    else:
        print("cifar_cnn.pth 발견. 학습 생략.")
        cifar_model = CIFAR_CNN().to(DEVICE)
        cifar_model.load_state_dict(torch.load("cifar_cnn.pth", map_location=DEVICE))

    # ── 학습 결과 시각화 ──────────────────────
    visualize_results(mnist_model, "mnist")
    visualize_results(cifar_model, "cifar")

    # ── Problem 1: Targeted FGSM ─────────────
    run_attack("mnist",  fgsm_targeted,   "fgsm_targeted",   eps=0.3,  target_label=0)
    run_attack("cifar",  fgsm_targeted,   "fgsm_targeted",   eps=0.05, target_label=0)

    # ── Problem 2: Untargeted FGSM ───────────
    run_attack("mnist",  fgsm_untargeted, "fgsm_untargeted", eps=0.3)
    run_attack("cifar",  fgsm_untargeted, "fgsm_untargeted", eps=0.05)

    # ── Problem 3-1: Targeted PGD ────────────
    run_attack("mnist",  pgd_targeted,    "pgd_targeted",    eps=0.3,  alpha=0.01, iters=40, target_label=0)
    run_attack("cifar",  pgd_targeted,    "pgd_targeted",    eps=0.05, alpha=0.01, iters=40, target_label=0)

    # ── Problem 3-2: Untargeted PGD ──────────
    run_attack("mnist",  pgd_untargeted,  "pgd_untargeted",  eps=0.3,  alpha=0.01, iters=40)
    run_attack("cifar",  pgd_untargeted,  "pgd_untargeted",  eps=0.05, alpha=0.01, iters=40)

    print("\n모든 실험 완료. 결과는 results/ 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
