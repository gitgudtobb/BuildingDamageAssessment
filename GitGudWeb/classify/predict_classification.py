import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
from ultralytics import YOLO
from transformers import ViTModel
from tqdm import tqdm


class TwinViT(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(TwinViT, self).__init__()

        # Use large ViT model
        self.stream_pre = ViTModel.from_pretrained('google/vit-large-patch16-224')
        self.stream_post = ViTModel.from_pretrained('google/vit-large-patch16-224')

        # Freeze ViT layers
        for param in self.stream_pre.parameters():
            param.requires_grad = False
        for param in self.stream_post.parameters():
            param.requires_grad = False

        # Classification head
        hidden_size = self.stream_pre.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes))

    def forward(self, pre_img, post_img):
        outputs_pre = self.stream_pre(pre_img)
        outputs_post = self.stream_post(post_img)
        pre_features = outputs_pre.pooler_output
        post_features = outputs_post.pooler_output
        combined = torch.cat((pre_features, post_features), dim=1)
        return self.classifier(combined)


def perform_instance_segmentation(image_path, weights_path="best.pt"):
    model = YOLO(weights_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    results = model(image_path)

    for result in results:
        if result.masks is not None:
            masks = result.masks.xy
            boxes = result.boxes.xyxy.int()
            orig_img = PILImage.open(image_path).convert("RGB")
            orig_img_np = np.array(orig_img)

            for i, (mask_coords, bbox) in enumerate(zip(masks, boxes)):
                mask = np.array(mask_coords, dtype=np.int32)
                building_mask = np.zeros(orig_img_np.shape[:2], dtype=np.uint8)
                cv2.fillPoly(building_mask, [mask], color=255)
                bool_mask = building_mask > 0

                segmented_building_np = np.zeros_like(orig_img_np)
                segmented_building_np[bool_mask] = orig_img_np[bool_mask]

                x_min, y_min, x_max, y_max = bbox
                cropped_building_np = segmented_building_np[y_min:y_max, x_min:x_max]
                cropped_building_pil = PILImage.fromarray(cropped_building_np)
                predicted_mask_cropped = building_mask[y_min:y_max, x_min:x_max]

                yield cropped_building_pil, (x_min, y_min, x_max, y_max), predicted_mask_cropped


def visualize_masks(base_path, results, labels=True):
    if results is None:
        print("No results to visualize.")
        return

    images_path = os.path.join(base_path, "images")
    labels_path = os.path.join(base_path, "labels")
    output_dir = "visualize_preds"
    os.makedirs(output_dir, exist_ok=True)

    damage_color_map = {
        0: [0, 255, 0],  # Green for no damage
        1: [255, 255, 0],  # Yellow for minor damage
        2: [255, 165, 0],  # Orange for major damage
        3: [255, 0, 0]  # Red for destroyed
    }
    damage_label_map = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}

    for post_filename, building_results in results.items():
        image_id = post_filename.replace("_post_disaster" + os.path.splitext(post_filename)[1], "")
        post_image_path = os.path.join(images_path, post_filename)

        if os.path.exists(post_image_path):
            original_image = cv2.imread(post_image_path)
            height, width, _ = original_image.shape

            # Create ground truth mask if labels exist
            if labels:
                gt_mask_colored = np.zeros((height, width, 3), dtype=np.uint8)
                label_filename = f"{image_id}_post_disaster.txt"
                label_file_path = os.path.join(labels_path, label_filename)

                if os.path.exists(label_file_path):
                    with open(label_file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) > 1:
                                level = int(parts[0])
                                coords_normalized = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                                coords_denormalized = (coords_normalized * np.array([width, height])).astype(np.int32)
                                color_rgb = damage_color_map.get(level, [0, 0, 0])
                                color = color_rgb[::-1]
                                cv2.fillPoly(gt_mask_colored, [coords_denormalized], color)

                    output_path_gt = os.path.join(output_dir, f"{image_id}_gt_mask.png")
                    cv2.imwrite(output_path_gt, gt_mask_colored)

            # Create prediction overlay
            predictions_overlay = original_image.copy()
            prediction_counts = {0: 0, 1: 0, 2: 0, 3: 0}

            for i, (bbox_pred, prediction, predicted_mask_cropped) in enumerate(building_results):
                x_min_pred = max(0, int(bbox_pred[0]))
                y_min_pred = max(0, int(bbox_pred[1]))
                x_max_pred = min(width, int(bbox_pred[2]))
                y_max_pred = min(height, int(bbox_pred[3]))

                resized_mask = cv2.resize(predicted_mask_cropped,
                                          (int(x_max_pred - x_min_pred),
                                           int(y_max_pred - y_min_pred)),
                                          interpolation=cv2.INTER_NEAREST)

                predicted_level = damage_label_map.get(prediction.lower(), 0)
                prediction_counts[predicted_level] += 1

                color_rgb = damage_color_map.get(predicted_level, [0, 0, 0])
                color = color_rgb[::-1]

                roi = predictions_overlay[y_min_pred:y_max_pred, x_min_pred:x_max_pred]
                mask_boolean = resized_mask > 0
                alpha = 0.7  # Transparency factor
                roi[mask_boolean] = cv2.addWeighted(roi[mask_boolean], 1 - alpha,
                                                    np.full_like(roi[mask_boolean], color), alpha, 0)

            # Save prediction counts
            counts_output_filepath = os.path.join(output_dir, f"{image_id}_pred_counts.txt")
            with open(counts_output_filepath, 'w') as f_counts:
                for level in sorted(prediction_counts.keys()):
                    f_counts.write(f"Level {level}: {prediction_counts[level]}\n")

            # Save prediction overlay
            output_path_pred = os.path.join(output_dir, f"{image_id}_pred.png")
            cv2.imwrite(output_path_pred, predictions_overlay)


def predict(base_path="classification",
            yolo_weights_path="best.pt",
            model_path="best_twin_vit.pth",
            results_path="prediction_masks.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classification model
    classification_model = TwinViT(num_classes=4).to(device)
    try:
        classification_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded classification model weights from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Classification model weights not found at: {model_path}")
        return None

    classification_model.eval()

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Process images
    images_path = os.path.join(base_path, "images")
    image_filenames = [f for f in os.listdir(images_path) if f.endswith(('.tif', '.png', '.jpg'))]
    image_filenames.sort()

    print("Starting prediction using instance segmentation with TwinViT...")
    results = {}

    with torch.no_grad():
        for filename in tqdm(image_filenames, desc="Processing images"):
            if "_pre_disaster" in filename:
                base_id = filename.replace("_pre_disaster", "").split('.')[0]
                post_filename = f"{base_id}_post_disaster{os.path.splitext(filename)[1]}"
                pre_image_path = os.path.join(images_path, filename)
                post_image_path = os.path.join(images_path, post_filename)

                if os.path.exists(post_image_path):
                    pre_buildings = list(perform_instance_segmentation(pre_image_path, yolo_weights_path))
                    post_buildings = list(perform_instance_segmentation(post_image_path, yolo_weights_path))

                    num_buildings = min(len(pre_buildings), len(post_buildings))
                    building_results = []

                    for i in range(num_buildings):
                        try:
                            pre_crop, _, _ = pre_buildings[i]
                            post_crop, bbox, predicted_mask = post_buildings[i]

                            pre_tensor = transform(pre_crop).unsqueeze(0).to(device)
                            post_tensor = transform(post_crop).unsqueeze(0).to(device)

                            outputs = classification_model(pre_tensor, post_tensor)
                            _, predicted = torch.max(outputs, 1)
                            damage_level = {0: 'no-damage', 1: 'minor-damage',
                                            2: 'major-damage', 3: 'destroyed'}[predicted.item()]

                            building_results.append((bbox, damage_level, predicted_mask))
                        except Exception as e:
                            print(f"Error processing building {i}: {str(e)}")
                            continue

                    results[post_filename] = building_results

    # Save results
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Predictions saved to: {results_path}")

    # Visualize results
    visualize_masks(base_path, results, labels=True)

    return results


if __name__ == "__main__":
    predict()