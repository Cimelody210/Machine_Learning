import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

batchsize = 64
learning_rate = 0.001

epoch = 20

# Chuyển đổi hình ảnh sang tensor và biến đổi chúng thành vector 1D (28*28)
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

# Tải bộ dữ liệu MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Tạo DataLoader cho train và test
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Lớp đầu tiên, 28x28 hình ảnh đầu vào
        self.fc1 = nn.Linear(28 * 28, 256)

        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

        # Lớp đầu ra, 10 lớp cho các chữ số 0-9
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Hàm kích hoạt ReLU cho lớp 1
        x = torch.relu(self.fc2(x))  # Hàm kích hoạt ReLU cho lớp 2
        x = torch.relu(self.fc3(x))  # Hàm kích hoạt ReLU cho lớp 3
        x = self.fc4(x)              # Lớp đầu ra
        return x

# Khởi tạo mô hình, hàm mất mát và tối ưu hóa
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Huấn luyện mô hình
for e in range(epoch):
    model.train()
    correct_train, total_train = 0, 0
    ytruetrain, ypredtrain = [], []
    for image, labels in tqdm(train_loader, desc=f'Epoch {e+1}/{epoch}', unit="batch"):
        output = model(image)
        loss = criterion(output, labels)

        optimizer.zero_grad()  # Đặt gradient về 0
        loss.backward()        # Tính gradient
        optimizer.step()       # Cập nhật trọng số

        _, predicted = torch.max(output, 1)  # Tìm nhãn dự đoán
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        ytruetrain.extend(labels.tolist())
        ypredtrain.extend(predicted.tolist())

    train_acc = 100 * correct_train / total_train
    train_f1 = f1_score(ytruetrain, ypredtrain, average='macro')

    # Kiểm tra mô hình
    model.eval()
    correct_test, total_test = 0, 0
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Lấy nhãn dự đoán

            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            y_true_test.extend(labels.tolist())
            y_pred_test.extend(predicted.tolist())

    val_acc = 100 * correct_test / total_test
    val_f1 = f1_score(y_true_test, y_pred_test, average='macro')

    # In kết quả
    print(f"\nEpoch [{e+1}/{epoch}]")
    print(f"Train Accuracy: {train_acc:.2f}% | Train F1 Macro: {train_f1:.2f}")
    print(f"Val Accuracy: {val_acc:.2f}% | Val F1 Macro: {val_f1:.2f}")

fig = plt.figure(figsize=(10, 6))
fig.suptitle("Sample Prediction on MNIST test Set")

for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(test_dataset.data[i].numpy(), cmap='gray')
    ax.set_title(f"True: {y_true_test[i]} | Pred: {y_pred_test[i]}")
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
