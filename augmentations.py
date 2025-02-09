import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import numpy as np
import rasterio
import json
import cv2
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from albumentations.pytorch import ToTensorV2

# Load TIFF image
def load_image(image_path):
    with rasterio.open(image_path) as src:
        img = src.read()
        img = reshape_as_image(img)  # Convert (C, H, W) to (H, W, C)
        return img

# Load JSON labels and create masks
def load_mask(json_path, img_shape):
    with open(json_path, "r") as f:
        label_data = json.load(f)
    
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)

    for feature in label_data["features"]["xy"]:
        if "wkt" in feature:
            # Extract mask coordinates
            polygon_str = feature["wkt"].replace("POLYGON ((", "").replace("))", "")
            coords = [list(map(float, pt.split())) for pt in polygon_str.split(", ")]

            pts = np.array(coords, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color=(255, 255, 255))

    return mask


def apply_transforms(image, mask, augment=True):
    mask = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    transform = A.Compose([
        A.Resize(512, 512),  # Force resize first

        A.RandomGridShuffle(grid=(2,2), p=0.3),
        A.ElasticTransform(p=0.2),

        A.Rotate(limit=30, p=0.5),

        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(1, 32),
            hole_width_range=(1, 32),
            fill="random",
            p=0.2
        ),

        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.15),
        A.RandomGamma((80, 120), p=0.15),

        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.1),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.GaussNoise(std_range=(0.2, 0.44), mean_range=(0.0, 0.0), per_channel=True, noise_scale_factor=1, p=0.1),

        A.HorizontalFlip(p=0.4) if augment else A.NoOp(),
        A.VerticalFlip(p=0.4) if augment else A.NoOp(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

    transformed = transform(
        image=image,
        mask=mask,
        mask_interpolation=cv2.INTER_NEAREST  # Crucial for masks
    )

    # Post-process
    image_tensor = transformed['image']
    mask_tensor = transformed['mask'].float().unsqueeze(0)  # Add channel dim
    mask_tensor = (mask_tensor > 0.5).float()  # Ensure binary

    return image_tensor, mask_tensor
if __name__ == "__main__":

    image_path = "geotiffs/tier1/images/hurricane-florence_00000451_pre_disaster.tif"
    json_path = "geotiffs/tier1/labels/hurricane-florence_00000451_pre_disaster.json"

    image = load_image(image_path)
    mask = load_mask(json_path, img_shape=image.shape[:2])

    transformed_image, transformed_mask = apply_transforms(image, mask)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(transformed_image.permute(1, 2, 0).numpy())
    axs[0].set_title("Augmented Image")
    
    axs[1].imshow(transformed_mask.squeeze(), cmap="gray")
    axs[1].set_title("Augmented Mask")
    plt.show()
