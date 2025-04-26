import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
from ultralytics import YOLO
from transformers import ViTModel, logging
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import glob
import re
import warnings
import concurrent.futures

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms.functional')
warnings.filterwarnings("ignore", category=FutureWarning)


class TwinViT(nn.Module):
    def __init__(self, num_classes=4):
        super(TwinViT, self).__init__()
        logging.set_verbosity_error()
        self.stream_pre = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.stream_post = ViTModel.from_pretrained('google/vit-base-patch16-224')

        for param in self.stream_pre.parameters():
            param.requires_grad = False
        for param in self.stream_post.parameters():
            param.requires_grad = False

        hidden_size = self.stream_pre.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 1024),
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


def perform_instance_segmentation(image_path, yolo_model):
    orig_img = PILImage.open(image_path).convert("RGB")
    orig_img_np = np.array(orig_img)
    img_h, img_w = orig_img_np.shape[:2]

    results = yolo_model(image_path, verbose=False)

    for result in results:
        if result.masks is not None and result.boxes is not None:
            masks_coords_list = result.masks.xy
            boxes_xyxy = result.boxes.xyxy.int().cpu().numpy()

            for i, mask_coords_normalized in enumerate(masks_coords_list):
                if i >= len(boxes_xyxy): continue

                bbox_int = boxes_xyxy[i]
                x_min, y_min, x_max, y_max = bbox_int

                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(img_w, x_max), min(img_h, y_max)
                if x_min >= x_max or y_min >= y_max:
                    continue

                cropped_original_np = orig_img_np[y_min:y_max, x_min:x_max]
                cropped_original_pil = PILImage.fromarray(cropped_original_np)

                mask_coords_denormalized = np.array(mask_coords_normalized, dtype=np.int32)
                full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(full_mask, [mask_coords_denormalized], color=255)

                yield cropped_original_pil, tuple(bbox_int), full_mask


def visualize_masks(base_path, results, labels=True):
    if results is None:
        return

    images_path = os.path.join(base_path, "images")
    labels_path = os.path.join(base_path, "labels")
    output_dir = "visualize_preds"
    os.makedirs(output_dir, exist_ok=True)

    damage_color_map = {
        0: [0, 255, 0], 1: [255, 255, 0], 2: [255, 165, 0], 3: [255, 0, 0]
    }
    damage_label_map = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}

    for post_filename, building_results in results.items():
        image_id_match = re.match(r"(.+)_post_disaster\.(png|jpg|tif|jpeg)$", post_filename, re.IGNORECASE)
        if not image_id_match: continue
        image_id = image_id_match.group(1)

        post_image_path = os.path.join(images_path, post_filename)
        original_image = cv2.imread(post_image_path)
        height, width, _ = original_image.shape

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

        predictions_overlay = original_image.copy()
        prediction_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        alpha = 0.7

        for bbox_pred, prediction, instance_mask in building_results:
            level = damage_label_map.get(prediction.lower(), -1)
            if level == -1: continue

            prediction_counts[level] += 1

            if instance_mask.shape[0] != height or instance_mask.shape[1] != width:
                 print(f"Warning: Mask dimensions ({instance_mask.shape}) mismatch image ({height}, {width}) for {post_filename}. Skipping instance.")
                 continue

            color_rgb = damage_color_map.get(level)
            color_bgr = np.array(color_rgb[::-1], dtype=np.uint8)
            mask_boolean = instance_mask > 0
            colored_mask_layer = np.zeros_like(predictions_overlay)
            colored_mask_layer[mask_boolean] = color_bgr
            predictions_overlay[mask_boolean] = cv2.addWeighted(
                original_image[mask_boolean], 1 - alpha,
                colored_mask_layer[mask_boolean], alpha, 0
            )

        counts_output_filepath = os.path.join(output_dir, f"{image_id}_pred_counts.txt")
        with open(counts_output_filepath, 'w') as f_counts:
            for level in sorted(prediction_counts.keys()):
                f_counts.write(f"Level {level}: {prediction_counts[level]}\n")

        output_path_pred = os.path.join(output_dir, f"{image_id}_pred.png")
        cv2.imwrite(output_path_pred, predictions_overlay)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1_bool = mask1 > 0
    mask2_bool = mask2 > 0

    if mask1_bool.shape != mask2_bool.shape:
         print(f"Warning: mask_iou received masks with different shapes: {mask1.shape} vs {mask2.shape}. Result might be inaccurate.")
         return 0.0

    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    return intersection / union if union > 0 else 0.0


def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
    return iou

def match_instances(pre_instances, post_instances, iou_threshold=0.3, bbox_iou_filter_threshold=0.01):

    N = len(pre_instances)
    M = len(post_instances)
    if N == 0 or M == 0:
        return []

    cost_matrix = np.full((N, M), 0.0)

    for i in range(N):
        _, pre_bbox, pre_mask = pre_instances[i]
        if pre_mask is None: continue

        for j in range(M):
            _, post_bbox, post_mask = post_instances[j]
            if post_mask is None: continue

            iou_bb = bbox_iou(pre_bbox, post_bbox)

            if iou_bb > bbox_iou_filter_threshold:
                x_min_i, y_min_i, x_max_i, y_max_i = map(int, pre_bbox)
                x_min_j, y_min_j, x_max_j, y_max_j = map(int, post_bbox)

                x_int_min = max(x_min_i, x_min_j)
                y_int_min = max(y_min_i, y_min_j)
                x_int_max = min(x_max_i, x_max_j)
                y_int_max = min(y_max_i, y_max_j)

                iou_m = 0.0

                if x_int_min < x_int_max and y_int_min < y_int_max:
                    pre_mask_roi = pre_mask[y_int_min:y_int_max, x_int_min:x_int_max]
                    post_mask_roi = post_mask[y_int_min:y_int_max, x_int_min:x_int_max]

                    iou_m = mask_iou(pre_mask_roi, post_mask_roi)

                cost_matrix[i, j] = -iou_m

    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError as e:
        print(f"Error during linear sum assignment: {e}. Cost matrix shape: {cost_matrix.shape}")
        return []

    matches = []
    for r, c in zip(row_ind, col_ind):
        mask_iou_score = -cost_matrix[r, c]
        if mask_iou_score >= iou_threshold:
            matches.append((r, c))

    return matches

def process_single_pair(pre_path, yolo_weights_path, classification_model, device, transform, damage_levels):
    try:
        yolo_model = YOLO(yolo_weights_path)

        pre_filename = os.path.basename(pre_path)
        match = re.match(r"(.+)_pre_disaster(\..+)$", pre_filename, re.IGNORECASE)
        if not match:
            return None
        base_id, extension = match.groups()
        post_filename = f"{base_id}_post_disaster{extension}"
        post_path = os.path.join(os.path.dirname(pre_path), post_filename)

        pre_instances = list(perform_instance_segmentation(pre_path, yolo_model))
        post_instances = list(perform_instance_segmentation(post_path, yolo_model))

        matches = match_instances(pre_instances, post_instances, iou_threshold=0.3)
        matched_pre_indices = {i for i, j in matches}
        matched_post_indices = {j for i, j in matches}

        current_pair_results = []

        with torch.no_grad():
            for i_pre, i_post in matches:
                pre_crop_pil, _, pre_mask_full = pre_instances[i_pre]
                post_crop_pil, post_bbox, _ = post_instances[i_post]

                pre_tensor = transform(pre_crop_pil).unsqueeze(0).to(device)
                post_tensor = transform(post_crop_pil).unsqueeze(0).to(device)
                outputs = classification_model(pre_tensor, post_tensor)
                _, predicted_idx = torch.max(outputs, 1)
                prediction_label = damage_levels[predicted_idx.item()]

                current_pair_results.append((post_bbox, prediction_label, pre_mask_full))

        unmatched_pre_indices = set(range(len(pre_instances))) - matched_pre_indices
        for i_pre in unmatched_pre_indices:
            _, pre_bbox, pre_mask_full = pre_instances[i_pre]
            max_iou_with_post = 0.0
            if post_instances:
                 max_iou_with_post = max(
                     (mask_iou(pre_mask_full, post_inst[2]) for post_inst in post_instances),
                      default=0.0
                 )
            if max_iou_with_post < 0.1:
                current_pair_results.append((pre_bbox, 'destroyed', pre_mask_full))

        return post_filename, current_pair_results

    except Exception as e:
        print(f"Error processing pair {pre_path}: {e}")
        return None


def predict(base_path="datasets/classification",
            yolo_weights_path="instance_segmentation/l_seg/weights/best.pt",
            model_path="best_twin_vit.pth",
            results_path="prediction_masks.pkl"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classification_model = TwinViT(num_classes=4).to(device)
    classification_model.load_state_dict(torch.load(model_path, map_location=device))
    classification_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    damage_levels = {0: 'no-damage', 1: 'minor-damage', 2: 'major-damage', 3: 'destroyed'}

    images_path = os.path.join(base_path, "images")
    pre_image_files = glob.glob(os.path.join(images_path, '*_pre_disaster.*'))
    pre_image_files = [f for f in pre_image_files if f.lower().endswith(('.png', '.jpg', '.tif', '.jpeg'))]
    pre_image_files.sort()

    results = {}
    max_workers = min(os.cpu_count() or 1, 4)

    print(f"\nProcessing image pairs in parallel with {max_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_single_pair, pre_path, yolo_weights_path, classification_model, device, transform, damage_levels): pre_path for pre_path in pre_image_files}

        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(pre_image_files), desc="Processing Pairs"):
            pre_path = future_to_path[future]
            try:
                pair_result = future.result()
                if pair_result is not None:
                    post_filename, current_pair_results = pair_result
                    results[post_filename] = current_pair_results
            except Exception as exc:
                print(f'{pre_path} generated an exception in main thread collector: {exc}')

    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    visualize_masks(base_path, results, labels=True)

    return results


if __name__ == "__main__":
    predict(
        base_path="classification",
        yolo_weights_path="best.pt",
        model_path="best_twin_vit.pth",
        results_path="prediction_masks.pkl"
    )