import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# Configuration
DATA_DIR = "datasets/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
MODEL_SAVE_PATH = "models/skin_cancer_model.pth"
NUM_CLASSES = 9
BATCH_SIZE = 32
EPOCHS = 1

# Transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Data
print("Loading Skin Cancer dataset...")
dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = dataset.classes
print(f"Classes: {class_names}")

# Model Setup (ResNet18)
print("Setting up ResNet18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training Loop
print("Starting training...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Batch {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Loss: {running_loss / len(dataloader):.4f}")

# Save Model
print("Saving model...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
# Save class names
with open("models/skin_cancer_classes.txt", "w") as f:
    f.write("\n".join(class_names))

print("Done!")
