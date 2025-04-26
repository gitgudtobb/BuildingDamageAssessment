import json
import os
import pickle
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from shapely.wkt import loads as wkt_loads
from tqdm import tqdm
from augmentations import load_image
from PIL import Image as PILImage
from ultralytics import YOLO
from transformers import ViTModel, get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
from scipy.optimize import linear_sum_assignment

# Dataset Class with Localization
class BuildingDamageDataset(Dataset):
    def __init__(self, labels_path, images_path, transform=None):
        self.features = []
        self.images_path = images_path
        self.transform = transform
        self.label_map = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3} # Treat un-classified as no-damage

        no_damage_features = []
        minor_damage_features = []
        major_damage_features = []
        destroyed_features = []

        print("Loading and categorizing features...")
        for root, _, files in os.walk(labels_path):
            for file in files:
                if file.endswith('.json'):
                    label_path = os.path.join(root, file)
                    with open(label_path) as f:
                        data = json.load(f)
                        image_id = os.path.splitext(file)[0]

                        pre_img_path = self._find_image_path(images_path, image_id, "pre_disaster")
                        post_img_path = self._find_image_path(images_path, image_id, "post_disaster")

                        if pre_img_path and post_img_path:
                            if 'features' in data and 'xy' in data['features']:
                                for feature in data['features']['xy']:
                                    properties = feature.get('properties', {})
                                    subtype = properties.get('subtype',  'no-damage')
                                    feature_data = {
                                        'feature': feature,
                                        'pre_img_path': pre_img_path,
                                        'post_img_path': post_img_path
                                    }
                                    if subtype == 'no-damage':
                                        no_damage_features.append(feature_data)
                                    elif subtype == 'minor-damage':
                                        minor_damage_features.append(feature_data)
                                    elif subtype == 'major-damage':
                                        major_damage_features.append(feature_data)
                                    elif subtype == 'destroyed':
                                        destroyed_features.append(feature_data)
                                    elif subtype == 'un-classified':
                                        continue

        # Undersample no-damage features
        num_no_damage_to_keep = int(len(no_damage_features) * 0.09)
        self.features.extend(random.sample(no_damage_features, num_no_damage_to_keep))
        self.features.extend(minor_damage_features)
        self.features.extend(minor_damage_features)
        self.features.extend(major_damage_features)
        self.features.extend(major_damage_features)
        self.features.extend(destroyed_features)

        random.shuffle(self.features) # Shuffle the combined features
        print(f"Total features loaded after undersampling no-damage: {len(self.features)}")

    def _find_image_path(self, images_path, image_id, disaster_type):
        for ext in ['.tif', '.png', '.jpg']:
            pattern = f"{image_id}{ext}"
            path = os.path.join(images_path, pattern)
            if os.path.exists(path):
                return path

        print(f"Warning: Could not find {disaster_type} image for {image_id}")
        return None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]
        feature = item['feature']

        # Load images
        pre_img = load_image(item['pre_img_path'])
        post_img = load_image(item['post_img_path'])
        if pre_img.dtype != np.uint8:
            pre_img = pre_img.astype(np.uint8)
        if post_img.dtype != np.uint8:
            post_img = post_img.astype(np.uint8)

        # Extract polygon coordinates
        polygon = wkt_loads(feature['wkt'])
        coords = np.array(list(polygon.exterior.coords)).astype(int)

        mask = np.zeros((pre_img.shape[0], pre_img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [coords], 1)
        y, x = np.where(mask)

        if len(y) == 0 or len(x) == 0:
            pre_crop = np.zeros((224, 224, 3), dtype=np.uint8)
            post_crop = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            xmin, xmax = np.min(x), np.max(x)
            ymin, ymax = np.min(y), np.max(y)

            pre_crop = self.process_crop(pre_img, ymin, ymax, xmin, xmax)
            post_crop = self.process_crop(post_img, ymin, ymax, xmin, xmax)

        if 'properties' in feature:
            properties = feature['properties']
            subtype = properties.get('subtype', 'no-damage')
            label = self.label_map.get(subtype, 0)
        else:
            label = 0

        if self.transform:
            pre_crop = self.transform(pre_crop)
            post_crop = self.transform(post_crop)

        return {'pre': pre_crop, 'post': post_crop}, torch.tensor(label)

    def process_crop(self, img, ymin, ymax, xmin, xmax):
        crop = img[ymin:ymax+1, xmin:xmax+1]
        h, w = crop.shape[:2]

        if h == 0 or w == 0:
            crop = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            scale = 224 / max(h, w)
            crop = cv2.resize(crop, (int(w*scale), int(h*scale)))
            pad_h = 224 - crop.shape[0]
            pad_w = 224 - crop.shape[1]
            crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w,
                                        cv2.BORDER_CONSTANT, value=0)
        return crop

class TwinViT(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(TwinViT, self).__init__()

        self.stream_pre = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.stream_post = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # Freeze ViT layers
        for param in self.stream_pre.parameters(): param.requires_grad = False
        for param in self.stream_post.parameters(): param.requires_grad = False

        # Get the hidden size of the ViT model
        hidden_size = self.stream_pre.config.hidden_size

        # Classification head
        hidden_size = self.stream_pre.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(128, num_classes)
        )

    def forward(self, pre_img, post_img):
        outputs_pre = self.stream_pre(pre_img)
        outputs_post = self.stream_post(post_img)
        pre_features = outputs_pre.pooler_output
        post_features = outputs_post.pooler_output
        combined = torch.cat((pre_features, post_features), dim=1)
        return self.classifier(combined)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()

        probs_flat = probs.view(probs.size(0), -1)
        targ_flat  = one_hot.view(one_hot.size(0), -1)

        intersection = (probs_flat * targ_flat).sum(1)
        union        = probs_flat.sum(1) + targ_flat.sum(1)
        dice_score   = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_score.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        p_t = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma) * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()

def main_classification(base_path="geotiffs/tier1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    base_path="geotiffs/tier1"
    base_path_yolo = "datasets/classification"   # for yolo
    images_path  = os.path.join(base_path_yolo, "images")
    labels_path = os.path.join(base_path, "labels")

    # Ensure paths exist
    for path in [images_path, labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4), 
        transforms.RandomRotation(degrees=45, fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.334, 0.347, 0.262], std=[0.174, 0.143, 0.134]),
    ])

    # Dataset & Loader
    dataset = BuildingDamageDataset(
        labels_path=labels_path,
        images_path=images_path,
        transform=transform
    )

    if len(dataset) == 0:
        raise ValueError("No valid data found. Check your directory structure and file formats.")

    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = TwinViT(num_classes=4).to(device)

    # Loss and Optimizer
    dice_loss  = DiceLoss()
    focal_loss = FocalLoss(gamma=2.0, alpha=None)
    ce_loss    = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.8, 0.8, 0.7])).to(device)

    num_epochs = 100

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_f1 = 0.0

    class_names = ["0", "1", "2", "3"]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch
            pre = inputs['pre'].float().to(device)
            post = inputs['post'].float().to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(pre, post)
            l_ce    = ce_loss(outputs, labels)
            l_focal = focal_loss(outputs, labels)
            l_dice  = dice_loss(outputs, labels)
            loss = l_ce + 0.5 * l_focal + 0.5 * l_dice

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = batch
                pre = inputs['pre'].float().to(device)
                post = inputs['post'].float().to(device)
                labels = labels.to(device)

                outputs = model(pre, post)
                l_ce    = ce_loss(outputs, labels)
                l_focal = focal_loss(outputs, labels)
                l_dice  = dice_loss(outputs, labels)
                loss = l_ce + 0.5 * l_focal + 0.5 * l_dice

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Per‑class F1
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        for i, f1 in enumerate(f1_per_class):
            print(f"    F1({class_names[i]}): {f1:.4f}")

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"    Macro F1: {macro_f1:.4f}")

        print("\n" + classification_report(all_labels, all_preds, target_names=class_names, digits=4))
        scheduler.step(val_loss)

        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), "best_twin_vit.pth")
            print(f"  Saved best model with macro f1: {best_f1:.2f}%")

    print("Training Complete")
    print(f"Best validation accuracy: {best_f1:.2f}%")


if __name__ == "__main__":
    main_classification("datasets/classification")
