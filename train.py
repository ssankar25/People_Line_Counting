import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from CNN import CNNnet
from CustomDataset import CustomDataset

# Transformation: Resize and convert to Tensor
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# Initialize dataset
dataset = CustomDataset(
    annotations_file='TVHeads/segments/labels.csv',
    img_dir='TVHeads/segments',
    transform=transform
)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset)) # 80% of data is used for training
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) # Pytorch function that randomly splits the dataset into the sizes specified
                                                                             # by train_size and test_size, ensuring no overlap

# Initialize DataLoader
data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = CNNnet()
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

num_epochs = 10  # Total epochs for training

for epoch in range(num_epochs):
    for inputs, labels in data_loader: # Each iteration retrieves batches of images and labels, assigned to inputs and labels
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Backward propogation
        optimizer.step()  # Update parameters

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:  # Assuming you already have test_loader
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')