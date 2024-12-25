import os
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datetime import datetime
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import random_split

# Label processing
def load_label_mapping(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    label_mapping = {line.split()[0]: int(line.split()[1]) for line in lines}
    return label_mapping

label_mapping = load_label_mapping("label_to_idx.txt")
label_num = len(label_mapping)
print(label_mapping)

# Image Processing
class TrademarkDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_mapping=None, transform=None, is_train=True, processed=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_mapping = label_mapping
        self.is_train = is_train
        self.processed = processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.processed:
            image_name = self.data.iloc[idx, 0].replace('.jpg', '.pt')
            image_path = os.path.join(self.img_dir, image_name)
            image = torch.load(image_path, weights_only=True)
        else:
            img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        if self.is_train:
            labels = [label.strip() for label in self.data.iloc[idx, 1].split(',')]
            label_vec = np.zeros(len(self.label_mapping))
            for label in labels:
                label_vec[self.label_mapping[label]] = 1
            return image, torch.tensor(label_vec, dtype=torch.float32)
        return image

# class PreprocessedDataset(torch.utils.data.Dataset):
#     def __init__(self, csv_file, processed_image_dir, label_mapping=None, is_train=True):
#         self.data = pd.read_csv(csv_file)
#         self.processed_image_dir = processed_image_dir
#         self.label_mapping = label_mapping
#         self.is_train = is_train

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # 加載處理後的 .pt 文件
#         image_name = self.data.iloc[idx, 0].replace('.jpg', '.pt')
#         image_path = os.path.join(self.processed_image_dir, image_name)
#         image = torch.load(image_path, weights_only=True)

#         if self.is_train:
#             labels = [label.strip() for label in self.data.iloc[idx, 1].split(',')]
#             label_vec = np.zeros(len(self.label_mapping))
#             for label in labels:
#                 label_vec[self.label_mapping[label.strip()]] = 1
#             return image, torch.tensor(label_vec, dtype=torch.float32)

#         return image

# map calculate
def calculate_map(y_true, y_pred):
    aps = []
    for i in range(y_true.shape[1]):
        aps.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(aps)

# Image transform
transform = transforms.Compose([
    transforms.RandomResizedCrop((244,244)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_image_dir = 'train_data'
train_image_process_dir = 'processed_train_data3'
os.makedirs(train_image_process_dir, exist_ok=True)

# 預處理並保存所有圖片
for image_name in os.listdir(train_image_dir):
    if image_name.endswith('.jpg'):  # 根據文件類型過濾
        image_path = os.path.join(train_image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        processed_image = transform(image)
        
        save_path = os.path.join(train_image_process_dir, image_name.replace('.jpg', '.pt'))
        torch.save(processed_image, save_path)
print(f"Saved processed image success!")

# Load data
dataset = TrademarkDataset('train_data.csv', train_image_process_dir, label_mapping, is_train=True, processed=True)
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)  # 90% train data
val_size = dataset_size - train_size  # 10% validation data

# torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = TrademarkDataset("test_data_public.csv", "test_data/", label_mapping, transform, is_train=False, processed=False)
test_loader = DataLoader(test_dataset, shuffle=False)

# Model
model = models.efficientnet_b3(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, label_num),
    nn.Sigmoid()
    # nn.Dropout(0.3),
    # nn.Linear(model.classifier[1].in_features, 128),
    # nn.ReLU(),
    # nn.Linear(128, label_num),
    # nn.Sigmoid()
)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


# Training loop
epochs = 30
minLoss = 0
minValLoss = 0
maxMap = 1
best_model_path = 'best_model.pth'
early_stop_patience = 5
early_stop_counter = 0
import_model = False

if import_model: # decide whether enter pretrained model
    model.load_state_dict(torch.load('best_model.pth'))

for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}],  Learning Rate: {optimizer.param_groups[0]['lr']} Start!")
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    # print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    avg_train_loss = running_loss / len(train_loader)
    current_time = datetime.now()
    timestamp = current_time.strftime("%H:%M:%S")
    print(f"Epoch [{epoch+1}/{epochs}], Average training Loss: {avg_train_loss:.4f}, {timestamp}")

    # 驗證階段
    model.eval()
    val_loss = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(torch.sigmoid(outputs).cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    map_score = calculate_map(all_labels, all_predictions)  # 或使用 torchmetrics

    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, mAP: {map_score}")

    scheduler.step()
    
    if maxMap == 1:
        maxMap = map_score
        torch.save(model.state_dict(), best_model_path)
    elif maxMap > map_score:
        early_stop_counter += 1
        print(f"EarlyStopping counter: {early_stop_counter} out of {early_stop_patience}")
        if early_stop_counter >= early_stop_patience:
            print(f"EarlyStopping counter: {early_stop_counter}, Stop!")
            break
    else:
        maxMap = map_score
        torch.save(model.state_dict(), best_model_path)
        early_stop_counter = 0

# mAP Calculation and probability output
def calculate_probabilities(model, test_loader):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    filenames = []
    probabilities = []
    batch_start = 0
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to('cuda')
            outputs = model(images).cpu().numpy()
            
            batch_size = len(images)
            batch_filenames = test_loader.dataset.data.iloc[batch_start:batch_start + batch_size, 0].tolist()
            filenames.extend(batch_filenames)  # 添加到總的 filename 列表中
            probabilities.extend(outputs)
            batch_start += batch_size  # 更新批次的起始索引
    
    return filenames, np.array(probabilities)

# Export results to CSV
def export_probabilities(filenames, probabilities, output_file="predictions.csv"):
    probabilities = np.round(probabilities, 7)
    columns = ['filename'] + [f'class_{i}_prob' for i in range(probabilities.shape[1])]
    data = np.column_stack((filenames, probabilities))
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Probabilities saved to {output_file}")

print("Generating probabilities for test set...")
print(len(test_loader))
filenames, probabilities = calculate_probabilities(model, test_loader)
export_probabilities(filenames, probabilities, f"predictions.csv")
