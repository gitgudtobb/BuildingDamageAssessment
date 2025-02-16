import os
import numpy as np
import rasterio

def calculate_normalization_values(dataset_root_dir):

    # Calculates mean and standard deviation normalization values for RGB TIFF images
    image_dirs = [
        os.path.join(dataset_root_dir, 'tier1', 'images'),
        os.path.join(dataset_root_dir, 'tier3', 'images')
    ]

    total_pixel_count = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_squared = np.zeros(3, dtype=np.float64)

    image_count = 0

    for image_dir in image_dirs:
        if not os.path.isdir(image_dir):
            print(f"Warning: Directory not found: {image_dir}")
            continue

        print(image_dir)
        for filename in os.listdir(image_dir):
            if filename.endswith(".tif"):
                image_path = os.path.join(image_dir, filename)
                try:
                    with rasterio.open(image_path) as src:
                        image = src.read([1, 2, 3])  # R, G, B channels
                        image = image / 255.0

                        channel_sum += np.sum(image, axis=(1, 2))
                        # Sum squared pixel values for each channel
                        channel_sum_squared += np.sum(image**2, axis=(1, 2))

                        total_pixel_count += image.shape[1] * image.shape[2]
                        image_count += 1

                except rasterio.errors:
                    print(f"Warning: Could not read image: {image_path}. Skipping.")

    if image_count == 0:
        print("No images found in the specified directories.")
        return None

    mean = channel_sum / total_pixel_count
    std = np.sqrt((channel_sum_squared / total_pixel_count) - (mean ** 2))

    return mean, std


if __name__ == '__main__':
    dataset_root = 'geotiffs'
    normalization_values = calculate_normalization_values(dataset_root)

    if normalization_values:
        mean, std = normalization_values
        print("Mean values (RGB):", mean)
        print("Std values (RGB):", std)
    else:
        print("Normalization calculation failed.")