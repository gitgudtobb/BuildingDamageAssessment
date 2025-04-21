import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import cv2

def perform_instance_segmentation(image_path, weights_path="instance_segmentation/l_seg/weights/best.pt"):

    model = YOLO(weights_path)

    results = model(image_path)

    segmented_buildings = []

    # Iterate through the detected objects
    for result in results:
        if result.masks is not None:
            masks = result.masks.xy  # Get the mask coordinates
            boxes = result.boxes.xyxy.int()  # Get the bounding box coordinates
            orig_img = Image.open(image_path).convert("RGB")
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
                cropped_building_pil = Image.fromarray(cropped_building_np)

                segmented_buildings.append({
                    "cropped_image": cropped_building_pil,
                    "mask": building_mask
                })

    return segmented_buildings

if __name__ == "__main__":

    image_file = "datasets/yolo_dataset/images/train/guatemala-volcano_00000000_pre_disaster.png"

    if os.path.exists(image_file):
        segmented_buildings_data = perform_instance_segmentation(image_file)

        if segmented_buildings_data:
            print(f"Found {len(segmented_buildings_data)} building instances.")
            output_dir = "segmented_buildings"
            os.makedirs(output_dir, exist_ok=True)

            for i, building_data in enumerate(segmented_buildings_data):
                cropped_image = building_data["cropped_image"]
                mask = building_data["mask"]

                cropped_image.save(os.path.join(output_dir, f"building_{i+1}_cropped.png"))

                mask_image = Image.fromarray(mask).convert("L")
                mask_image.save(os.path.join(output_dir, f"building_{i+1}_mask.png"))

            print(f"Cropped building instances and masks saved to the '{output_dir}' directory.")
        else:
            print("No buildings detected in the image.")
    else:
        print(f"Error: Image file '{image_file}' not found.")