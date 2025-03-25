import json
import os
import pickle
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from shapely.wkt import loads as wkt_loads
from tqdm import tqdm
from augmentations import load_image
from PIL import Image as PILImage
from ultralytics import YOLO
from transformers import ViTModel

# 1. Dataset Class with Localization
class BuildingDamageDataset(Dataset):
    def __init__(self, labels_path, images_path, transform=None):
        self.features = []
        self.images_path = images_path
        self.transform = transform
        self.label_map = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3, 'un-classified': 0} # Treat un-classified as no-damage

        no_damage_features = []
        minor_damage_features = []
        major_damage_features = []
        destroyed_features = []
        un_classified_features = []

        print("Loading and categorizing features...")
        for root, _, files in os.walk(labels_path):
            for file in files:
                if file.endswith('.json') and random.randint(0,100) < 10:
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
                                    subtype = properties.get('subtype',
                                                                       properties.get('damage_subtype',
                                                                                      properties.get('damage',
                                                                                                     'no-damage')))
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
                                        un_classified_features.append(feature_data)

        # Undersample no-damage features
        num_no_damage_to_keep = int(len(no_damage_features) * 0.1)
        self.features.extend(random.sample(no_damage_features, num_no_damage_to_keep))
        self.features.extend(minor_damage_features)
        self.features.extend(major_damage_features)
        self.features.extend(destroyed_features)
        self.features.extend(un_classified_features)

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

# 2. Two-Stream ViT Model
class TwinViT(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(TwinViT, self).__init__()

        self.stream_pre = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.stream_post = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # Freeze ViT layers
        for param in self.stream_pre.parameters():
            param.requires_grad = False
        for param in self.stream_post.parameters():
            param.requires_grad = False

        # Get the hidden size of the ViT model
        hidden_size = self.stream_pre.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pre_img, post_img):
        outputs_pre = self.stream_pre(pre_img)
        outputs_post = self.stream_post(post_img)

        # Use the pooled output for classification
        pre_features = outputs_pre.pooler_output
        post_features = outputs_post.pooler_output

        # Concatenate features
        combined = torch.cat((pre_features, post_features), dim=1)
        return self.classifier(combined)

def perform_instance_segmentation(image_path, weights_path="instance_segmentation/l_seg/weights/best.pt"):
    model = YOLO(weights_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # Ensure model is on the correct device

    # Perform inference
    results = model(image_path)

    # Iterate through the detected objects and yield cropped images, bounding boxes, and masks
    for result in results:
        if result.masks is not None:
            masks = result.masks.xy  # Get the mask coordinates
            boxes = result.boxes.xyxy.int()  # Get the bounding box coordinates
            orig_img = PILImage.open(image_path).convert("RGB")
            orig_img_np = np.array(orig_img)

            for i, (mask_coords, bbox) in enumerate(zip(masks, boxes)):
                # Convert mask coordinates to integers
                mask = np.array(mask_coords, dtype=np.int32)

                # Create a boolean mask for the current building
                building_mask = np.zeros(orig_img_np.shape[:2], dtype=np.uint8)
                cv2.fillPoly(building_mask, [mask], color=255)
                bool_mask = building_mask > 0

                # Extract the building instance using the mask
                segmented_building_np = np.zeros_like(orig_img_np)
                segmented_building_np[bool_mask] = orig_img_np[bool_mask]

                # Crop the image to the bounding box to remove extra padding
                x_min, y_min, x_max, y_max = bbox
                cropped_building_np = segmented_building_np[y_min:y_max, x_min:x_max]
                cropped_building_pil = PILImage.fromarray(cropped_building_np)

                # Resize the predicted mask to the size of the cropped building
                predicted_mask_cropped = building_mask[y_min:y_max, x_min:x_max]

                yield cropped_building_pil, (x_min, y_min, x_max, y_max), predicted_mask_cropped

def predict(base_path="datasets/classification",
            yolo_weights_path="instance_segmentation/l_seg/weights/best.pt",
            model_path="best_twin_vit.pth",
            device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classification model
    classification_model = TwinViT(num_classes=4).to(device)
    try:
        classification_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded classification model weights from: {model_path}")
    except FileNotFoundError:
        print(f"Warning: Classification model weights not found at: {model_path}. Using randomly initialized weights.")
    classification_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    images_path = os.path.join(base_path, "images")
    image_filenames = [f for f in os.listdir(images_path) if f.endswith(('.tif', '.png', '.jpg'))]
    image_filenames.sort()

    print("Starting prediction using instance segmentation with TwinViT...")

    results = {} # Dictionary to store results per image

    with torch.no_grad():
        for filename in tqdm(image_filenames, desc="Processing images"):
            if "_pre_disaster" in filename and random.randint(0,100) < 5:
                base_id = filename.replace("_pre_disaster", "").split('.')[0]
                post_filename = f"{base_id}_post_disaster{os.path.splitext(filename)[1]}"
                pre_image_path = os.path.join(images_path, filename)
                post_image_path = os.path.join(images_path, post_filename)

                if os.path.exists(post_image_path):
                    pre_building_data = list(perform_instance_segmentation(pre_image_path, yolo_weights_path))
                    post_building_data = list(perform_instance_segmentation(post_image_path, yolo_weights_path))

                    print(f"Processing pair: {filename} and {post_filename}")
                    print(f"  Detected {len(pre_building_data)} buildings in pre-disaster image.")
                    print(f"  Detected {len(post_building_data)} buildings in post-disaster image.")

                    predictions = []
                    bounding_boxes = []
                    predicted_masks = []

                    num_buildings = min(len(pre_building_data), len(post_building_data))
                    for i in range(num_buildings):
                        try:
                            pre_crop, _, _ = pre_building_data[i]
                            post_crop, bbox, predicted_mask = post_building_data[i]

                            # Preprocess for classification
                            pre_tensor = transform(pre_crop).unsqueeze(0).to(device)
                            post_tensor = transform(post_crop).unsqueeze(0).to(device)

                            # Predict damage
                            outputs = classification_model(pre_tensor, post_tensor)
                            _, predicted = torch.max(outputs, 1)
                            damage_level = {0: 'no-damage', 1: 'minor-damage', 2: 'major-damage', 3: 'destroyed'}[predicted.item()]
                            print(f"  Building {i+1}: Predicted damage - {damage_level}, Bounding Box: {bbox}")
                            predictions.append(damage_level)
                            bounding_boxes.append(bbox)
                            predicted_masks.append(predicted_mask)
                        except IndexError:
                            print(f"Warning: Index out of bounds for building {i+1}. Skipping.")
                            continue

                    results[post_filename] = list(zip(bounding_boxes, predictions, predicted_masks)) # Store bounding boxes, predictions, and predicted masks
                else:
                    print(f"Warning: Post-disaster image not found for {filename}")
    return results

def visualize_masks(base_path="geotiffs/tier1", results=None):
    if results is None:
        print("No results to visualize.")
        return

    images_path = os.path.join(base_path, "images")
    labels_path = os.path.join(base_path, "labels")
    output_dir = "visualize_preds"
    os.makedirs(output_dir, exist_ok=True)

    # Define color map for damage levels
    damage_color_map = {
        0: [0, 255, 0],     # Green for no damage
        1: [255, 255, 0],   # Yellow for minor damage
        2: [255, 165, 0],   # Orange for major damage
        3: [255, 0, 0]      # Red for destroyed
    }
    damage_label_map = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}


    for post_filename, building_results in results.items():
        image_id = post_filename.replace("_post_disaster" + os.path.splitext(post_filename)[1], "")
        post_image_path = os.path.join(images_path, post_filename)

        if os.path.exists(post_image_path):
            original_image = cv2.imread(post_image_path)
            height, width, _ = original_image.shape

            # Create ground truth mask
            gt_mask_colored = np.zeros((height, width, 3), dtype=np.uint8)
            label_filename = f"{image_id}_post_disaster.txt"
            label_file_path = os.path.join(labels_path, label_filename)

            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 1:
                            level = int(parts[0]) # Assuming level is the class ID
                            coords_normalized = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                            coords_denormalized = (coords_normalized * np.array([width, height])).astype(np.int32)
                            color_rgb = damage_color_map.get(level, [0, 0, 0]) # Default to black if level not found
                            color = color_rgb[::-1] # Convert RGB to BGR
                            cv2.fillPoly(gt_mask_colored, [coords_denormalized], color)
                    
            output_path_gt = os.path.join(output_dir, f"{image_id}_gt_mask.png")
            cv2.imwrite(output_path_gt, gt_mask_colored)
            print(f"Saved ground truth mask to: {output_path_gt}")

            # Create prediction mask (colored, black background) with all buildings
            prediction_mask_colored = np.zeros((height, width, 3), dtype=np.uint8)
            for i, (bbox_pred, prediction, predicted_mask_cropped) in enumerate(building_results):
                x_min_pred, y_min_pred, x_max_pred, y_max_pred = bbox_pred
                resized_mask = cv2.resize(predicted_mask_cropped, (int(x_max_pred - x_min_pred), int(y_max_pred - y_min_pred)), interpolation=cv2.INTER_NEAREST)

                predicted_level = damage_label_map.get(prediction.lower(), 0)
                color_rgb = damage_color_map.get(predicted_level, [0, 0, 0])
                color = color_rgb[::-1] # Convert RGB to BGR

                # Place the colored mask onto the black background
                prediction_mask_colored[y_min_pred:y_max_pred, x_min_pred:x_max_pred][resized_mask > 0] = color

            output_path_pred = os.path.join(output_dir, f"{image_id}_pred.png")
            cv2.imwrite(output_path_pred, prediction_mask_colored)
            print(f"Saved predicted masks to: {output_path_pred}")

        else:
            print(f"Warning: Could not find image file for {image_id}")


def main_classification(base_path="geotiffs/tier1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    images_path = os.path.join(base_path, "images")
    labels_path = os.path.join(base_path, "labels")

    # Ensure paths exist
    for path in [images_path, labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Model
    model = TwinViT(num_classes=4).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Training Loop
    num_epochs = 20
    best_acc = 0.0

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
            loss = criterion(outputs, labels)

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

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = batch
                pre = inputs['pre'].float().to(device)
                post = inputs['post'].float().to(device)
                labels = labels.to(device)

                outputs = model(pre, post)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_twin_vit.pth")
            print(f"  Saved best model with val acc: {best_acc:.2f}%")

    print("Training Complete")
    print(f"Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Building Damage Assessment")
    parser.add_argument('--base_path', type=str, default="datasets/classification",
                        help="Base path for data")
    parser.add_argument('--mode', choices=['train','predict', 'visualize_masks'], default='predict',
                        help="Mode: train, predict, or visualize_masks")
    parser.add_argument('--model_path', type=str, default="best_twin_vit.pth",
                        help="Path to classification model weights")
    parser.add_argument('--yolo_weights_path', type=str, default="instance_segmentation/l_seg/weights/best.pt",
                        help="Path to YOLO segmentation model weights")
    parser.add_argument('--results_path', type=str, default="predictions_masks.pkl",
                        help="Path to save/load prediction results with masks")

    args = parser.parse_args()

    if args.mode == 'train':
        main_classification(args.base_path)
    elif args.mode == 'predict':
        results = predict(args.base_path, args.yolo_weights_path, args.model_path)
        with open(args.results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Predictions with masks saved to: {args.results_path}")
    elif args.mode == 'visualize_masks':
        try:
            with open(args.results_path, 'rb') as f:
                results = pickle.load(f)
            visualize_masks(args.base_path, results)
        except FileNotFoundError:
            print(f"Error: Results file not found at {args.results_path}. Run prediction first.")