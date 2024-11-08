# Dựa trên câu 3 nhưng sử dụng CNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load MNIST dataset with transformations suitable for CNN
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create a smaller subset of the MNIST dataset for quick training
train_indices = np.random.choice(len(mnist_train), size=int(0.1 * len(mnist_train)), replace=False)
test_indices = np.random.choice(len(mnist_test), size=int(0.1 * len(mnist_test)), replace=False)
train_subset = Subset(mnist_train, train_indices)
test_subset = Subset(mnist_test, test_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=1000, shuffle=False)

# Initialize CNN model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the CNN model
n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on the test subset
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# Calculate F1 scores
fl_macro = f1_score(y_true, y_pred, average='macro')
fl_micro = f1_score(y_true, y_pred, average='micro')
print("F1 Score (Macro):", fl_macro)
print("F1 Score (Micro):", fl_micro)

# Visualize some test images with their predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("CNN on MNIST Test Subset")

for i, ax in enumerate(axes.flat):
    ax.imshow(test_subset[i][0].numpy().squeeze(), cmap='gray')
    ax.set_title(f'True: {y_true[i]}, Pred: {y_pred[i]}')
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
