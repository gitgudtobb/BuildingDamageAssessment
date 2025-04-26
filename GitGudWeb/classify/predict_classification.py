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
from scipy.optimize import linear_sum_assignment

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


def perform_instance_segmentation(image_path, weights_path="best.pt"):
    model = YOLO(weights_path)
    #model.to('cpu')  # Force CPU execution
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

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

            for bbox_pred, prediction, pre_mask in building_results:
                if prediction != 'destroyed':
                    continue

                x0, y0, x1, y1 = map(int, bbox_pred)
                w, h = x1 - x0, y1 - y0
                if w <= 0 or h <= 0:
                    continue

                if pre_mask is None or pre_mask.size == 0:
                    continue
                resized = cv2.resize(pre_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_boolean = resized > 0
                if not mask_boolean.any():
                    continue

                prediction_counts[3] += 1

                roi = predictions_overlay[y0:y1, x0:x1]
                alpha = 0.7
                red = np.array([0, 0, 255], dtype=np.uint8)
                overlay = np.full_like(roi, red)
                roi[mask_boolean] = cv2.addWeighted(
                    roi[mask_boolean], 1 - alpha,
                    overlay[mask_boolean], alpha, 0
                )

            for bbox_pred, prediction, post_mask in building_results:
                if prediction == 'destroyed':
                    continue

                x0, y0, x1, y1 = map(int, bbox_pred)
                w, h = x1 - x0, y1 - y0
                if w <= 0 or h <= 0 or post_mask is None or post_mask.size == 0:
                    continue

                resized = cv2.resize(post_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_boolean = resized > 0
                if not mask_boolean.any():
                    continue

                lvl = damage_label_map[prediction.lower()]
                prediction_counts[lvl] += 1

                color_bgr = damage_color_map[lvl][::-1]
                roi = predictions_overlay[y0:y1, x0:x1]
                alpha = 0.7
                overlay = np.full_like(roi, color_bgr)
                roi[mask_boolean] = cv2.addWeighted(
                    roi[mask_boolean], 1 - alpha,
                    overlay[mask_boolean], alpha, 0
                )

            # Save prediction counts
            counts_output_filepath = os.path.join(output_dir, f"{image_id}_pred_counts.txt")
            with open(counts_output_filepath, 'w') as f_counts:
                for level in sorted(prediction_counts.keys()):
                    f_counts.write(f"Level {level}: {prediction_counts[level]}\n")

            # Save prediction overlay
            output_path_pred = os.path.join(output_dir, f"{image_id}_pred.png")
            cv2.imwrite(output_path_pred, predictions_overlay)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    if mask1 is None or mask2 is None:
        return 0.0
    if mask1.shape != mask2.shape:
        min_h = min(mask1.shape[0], mask2.shape[0])
        min_w = min(mask1.shape[1], mask2.shape[1])
        mask1 = mask1[:min_h, :min_w]
        mask2 = mask2[:min_h, :min_w]
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0


def match_instances(pre_instances, post_instances, iou_threshold=0.3):
    N, M = len(pre_instances), len(post_instances)
    cost = np.zeros((N, M), dtype=np.float32)

    # Build cost matrix as negative IoU (to minimize)
    for i, (_, _, pre_mask) in enumerate(pre_instances):
        pre_bool = pre_mask > 0
        for j, (_, _, post_mask) in enumerate(post_instances):
            post_bool = post_mask > 0
            cost[i, j] = -mask_iou(pre_bool, post_bool)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_ind, col_ind):
        if -cost[i, j] >= iou_threshold:
            matches.append((i, j))
    return matches

def predict(base_path="classification",
            yolo_weights_path="best.pt",
            model_path="best_twin_vit.pth",
            results_path="prediction_masks.pkl",
            id_list=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #device = torch.device('cpu')
    
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
    
    if id_list is not None:
        image_filenames = [f for f in image_filenames if any(f.startswith(id_) for id_ in id_list)]

    print(image_filenames)
    
    print("Starting prediction using instance segmentation with TwinViT...")
    results = {}

    with torch.no_grad():
        for pre_filename in tqdm(image_filenames, desc="Processing images"):
            base_id = pre_filename.replace("_pre_disaster", "").split('.')[0]
            post_filename = f"{base_id}_post_disaster{os.path.splitext(pre_filename)[1]}"
            pre_path = os.path.join(images_path, pre_filename)
            post_path = os.path.join(images_path, post_filename)

            if not os.path.exists(post_path):
                print(f"Warning: Post-disaster image not found for {pre_filename}")
                continue

            pre_instances = list(perform_instance_segmentation(pre_path, yolo_weights_path))
            post_instances = list(perform_instance_segmentation(post_path, yolo_weights_path))

            print(f"Processing pair: {pre_filename} & {post_filename}")
            print(f"  Detected {len(pre_instances)} pre-disaster instances, {len(post_instances)} post-disaster instances.")

            # Match instances by IoU
            matches = match_instances(pre_instances, post_instances)
            matched_pre = {i for i, _ in matches}
            matched_post = {j for _, j in matches}

            predictions = []
            boxes = []
            masks = []

            # Handle matched pairs
            for i_pre, i_post in matches:
                pre_crop, _, _ = pre_instances[i_pre]
                post_crop, bbox, post_mask = post_instances[i_post]

                pre_t = transform(pre_crop).unsqueeze(0).to(device)
                post_t = transform(post_crop).unsqueeze(0).to(device)
                outputs = classification_model(pre_t, post_t)
                _, pred = torch.max(outputs, 1)
                level = {0:'no-damage',1:'minor-damage',2:'major-damage',3:'destroyed'}[pred.item()]

                predictions.append(level)
                boxes.append(bbox)
                masks.append(post_mask)

            # Handle unmatched pre instances as destroyed
            for i_pre in set(range(len(pre_instances))) - matched_pre:
                _, bbox_pre, pre_mask = pre_instances[i_pre]

                iou_with_post = [
                    mask_iou(pre_mask > 0, post_mask > 0) if post_mask is not None else 0
                    for _, _, post_mask in post_instances
                ]

                if max(iou_with_post, default=0) < 0.1:
                    predictions.append('destroyed')
                    boxes.append(bbox_pre)
                    masks.append(pre_mask)
                else:
                    continue
            for j_post in set(range(len(post_instances))) - matched_post:
                _, bbox, post_mask = post_instances[j_post]
                predictions.append('no-damage')
                boxes.append(bbox)
                masks.append(post_mask)
            # Store results
            results[post_filename] = list(zip(boxes, predictions, masks))

    # Save results
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    # Visualize results
    visualize_masks(base_path, results, labels=True)

    return results


if __name__ == "__main__":
    predict()

