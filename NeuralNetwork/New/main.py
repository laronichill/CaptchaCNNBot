import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Grayscale, Resize
from PIL import Image

# Directory containing CAPTCHA images
data_dir = 'E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/base_images'

# Get list of all the images that end with .png and their respective ground truth (GT)
image_paths = sorted(list(glob.glob(f"{data_dir}/*.png")))
gt_list = [i.split('\\')[-1][:-4] for i in image_paths]  # for paths using '\'

# Number of characters per CAPTCHA
CHAR_PER_LABEL = 3

# Update list_char with only the valid letters in your dataset
list_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# One-hot encoding functions
def char_to_1_hot(char: str):
    """Convert a character to one-hot encoding."""
    out = np.zeros(len(list_char))
    idx = list_char.index(char)
    out[idx] = 1
    return out

def one_hot(characters: str):
    """Convert a string of characters to one-hot encoding."""
    return np.hstack([char_to_1_hot(c) for c in characters]).astype('uint8')

def one_hot_to_char(x: np.array):
    """Convert one-hot encoding back to a character."""
    y = np.array(x).squeeze()
    idx = np.argmax(y)
    return list_char[idx]

def one_hot_to_label(x):
    """Convert one-hot encoding back to a string label."""
    y = np.array(x).squeeze()
    label_list = []
    for i in range(0, CHAR_PER_LABEL):
        start = i * len(list_char)
        end = start + len(list_char)
        label_list.append(one_hot_to_char(y[start:end]))
    return "".join(label_list)

# Global Attention Module
class GlobalAttention(nn.Module):
    def __init__(self, num_channels):
        super(GlobalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

# Model with Attention
class ModelWithAttention(nn.Module):
    def __init__(self, num_characters):
        super(ModelWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same')  # Reduced channels
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same')
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        # Global Attention Layer
        self.attention = GlobalAttention(256)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16384, 256)  # Adjusted input size
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)

        self.output = nn.Linear(256, num_characters * CHAR_PER_LABEL)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        x = self.attention(x)

        x = self.flatten(x)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout2(x)

        x = self.output(x)

        return x

# Custom loss function
def my_loss_pytorch(y_pred, y_true, CHAR_PER_LABEL=3, NUM_CHAR=24):
    tot = 0.0
    for i in range(CHAR_PER_LABEL):
        start = i * NUM_CHAR
        end = start + NUM_CHAR
        tot += F.cross_entropy(y_pred[:, start:end], y_true[:, start:end].argmax(dim=1), reduction='sum')
    return tot

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, gt_one_hot, transform=None):
        self.image_paths = image_paths
        self.gt_one_hot = gt_one_hot
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize((128, 64))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = self.gt_one_hot[idx]

        return image, label

# Image preprocessing - convert to grayscale and tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalization
])

# Create training and validation datasets and dataloaders
gt_one_hot = [one_hot(gt) for gt in gt_list]

datasetTrain = CustomDataset(image_paths[:2*len(image_paths)//3], gt_one_hot[:2*len(gt_one_hot)//3], transform=transform)
dataloaderTrain = DataLoader(datasetTrain, batch_size=128, shuffle=True)

datasetVal = CustomDataset(image_paths[2*len(image_paths)//3:], gt_one_hot[2*len(gt_one_hot)//3:], transform=transform)
dataloaderVal = DataLoader(datasetVal, batch_size=128, shuffle=False)

# Initialize model, optimizer, scheduler, and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelWithAttention(num_characters=len(list_char)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop
total_time = 0
max_epochs = 50
val_interval = 1  # Validate every epoch
best_val_loss = float('inf')
trainingEpoch_loss = []
trainStepsLoss = []
validationEpoch_loss = []
print_interval = 5

for epoch in range(max_epochs):
    model.train()  # Set model to training mode
    print(f"{'-'*10} Epoch {epoch + 1}/{max_epochs} {'-'*10}")
    epoch_loss = 0
    step = 0

    # Training loop
    for batch_data in dataloaderTrain:
        step += 1
        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass and optimization
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = my_loss_pytorch(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if step % print_interval == 0:
            print(f"Batch {step}/{len(dataloaderTrain)}, Training Loss: {loss.item():.4f}")

        trainStepsLoss.append(loss.item())

    epoch_loss /= step
    trainingEpoch_loss.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{max_epochs} completed. Average Training Loss: {epoch_loss:.4f}")

    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch_data in dataloaderVal:
                inputs, labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = my_loss_pytorch(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(dataloaderVal)
        validationEpoch_loss.append(val_loss)
        print(f"Validation completed. Validation Loss: {val_loss:.4f}")

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            print(f"Validation Loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving the model.")
            torch.save(model.state_dict(), 'model_best.pth')
            best_val_loss = val_loss

        # Step the scheduler
        scheduler.step(val_loss)
