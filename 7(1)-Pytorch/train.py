import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Any
from resnet import ResNet, BasicBlock
from config import *
# from typing import Optional  # 추가
# from tqdm import tqdm #추가

NUM_CLASSES = 10  

#추가
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 랜덤 크롭 (데이터 증강)
    transforms.RandomHorizontalFlip(),  # 좌우 반전 (데이터 증강)
    # transforms.RandomRotation(15),  # 15도 회전
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변화
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 이동 변환
    transforms.ToTensor(),  # PIL → Tensor 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 정규화
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# CIFAR-10 데이터셋 로드
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resnet 18 선언하기
## TODO
# model = None
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES).to(device)

criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
# optimizer: optim.Adam = optim.Adam(model.parameters(), lr=LEARNING_RATE) # 변경
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)  # 🔥 5e-4 → 1e-4
# 학습 
def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
    model.train()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    # loop = tqdm(loader, desc="Training", leave=True)  # 🔥 tqdm 추가
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # loop.set_postfix(loss=loss.item(), acc=100. * correct / total) #tqdm progress bar

    accuracy: float = 100. * correct / total
    print(f"Train Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")

# 평가 
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float: #반환값 수정
    model.eval()
    total_loss: float = 0
    correct: int = 0
    total: int = 0


    # loop = tqdm(loader, desc="Evaluating", leave=True)  # 🔥 tqdm 추가
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # loop.set_postfix(loss=loss.item(), acc=100. * correct / total) #tqdm progress bar

    accuracy: float = 100. * correct / total
    print(f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy 

# 학습 및 평가 루프
# 최고 성능 저장 & early stopping 적용
best_acc = 0.0  # 최고 성능 저장용 변수
patience = 5  # 몇 epoch 동안 개선되지 않으면 중단할지 설정
counter = 0  # 개선되지 않은 epoch 카운터

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train(model, train_loader, criterion, optimizer, device)
    
    # 테스트 평가
    test_acc = evaluate(model, test_loader, criterion, device)

    # 최고 모델 저장 (파일명 유지)
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "resnet18_checkpoint.pth")
        print(f"🔥 New Best Model Saved! Accuracy: {best_acc:.2f}%")
        counter = 0  # 개선되었으므로 초기화
    else:
        counter += 1  # 개선되지 않으면 증가

    # Early Stopping 조건 확인
    if counter >= patience:
        print(f"⏹ Early stopping triggered! Best Accuracy: {best_acc:.2f}%")
        break  # 학습 중단

print(f"✅ Best Model Accuracy: {best_acc:.2f}%")
print(f"Model saved to resnet18_checkpoint.pth")
