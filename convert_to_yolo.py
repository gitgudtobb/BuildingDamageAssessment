import os
import json
import argparse
from pathlib import Path
import random
import rasterio
import shutil
from sklearn.model_selection import train_test_split
import glob
import imageio
from PIL import Image
import numpy as np

def get_image_dimensions(image_path):
    try:
        with rasterio.open(image_path) as src:
            return src.width, src.height
    except Exception as e:
        print(f"Error reading {image_path}: {str(e)}")
        return None, None

def convert_to_yolo_seg_format(images_dir, labels_dir, output_dir):
    # Create output directories
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)

    # Process all JSON files
    json_files = list(Path(labels_dir).glob('*.json'))

    for json_path in json_files:
        # Get corresponding image path
        image_name = json_path.name.replace('_disaster.json', '_disaster.png')
        image_path = Path(images_dir) / image_name

        if not image_path.exists():
            continue

        # Get image dimensions using rasterio
        img_width, img_height = get_image_dimensions(image_path)
        if not img_width or not img_height:
            continue

        # Load JSON annotations
        with open(json_path) as f:
            data = json.load(f)

        # Prepare YOLO segmentation annotations
        yolo_seg_annotations = []
        features = data['features'].get('xy', [])

        for feature in features:
            if 'wkt' not in feature:
                print(f"Skipping feature without WKT in {json_path.name}")
                continue

            try:
                wkt = feature['wkt']
                if 'POLYGON' not in wkt:
                    continue

                # Extract coordinates from WKT string
                coords_str = wkt.split('((')[1].split('))')[0]
                pairs = [list(map(float, p.split()))
                         for p in coords_str.replace('(', '').replace(')', '').split(', ')]

                # Normalize polygon coordinates and format for YOLO segmentation
                normalized_coords = []
                for x_coord, y_coord in pairs:
                    normalized_x = x_coord / img_width
                    normalized_y = y_coord / img_height
                    normalized_coords.extend([f"{normalized_x:.6f}", f"{normalized_y:.6f}"]) # Format and add x, then y

                yolo_seg_line = f"0 {' '.join(normalized_coords)}" # Class ID 0 (building)
                yolo_seg_annotations.append(yolo_seg_line)

            except (KeyError, IndexError, ValueError) as e:
                print(f"Skipping invalid feature in {json_path.name}: {str(e)}")
                continue

        img_output = output_dir / 'images' / image_name
        if not img_output.exists():
            try:
                shutil.copy(image_path, img_output)
            except Exception as e:
                print(f"Error copying {image_path}: {str(e)}")

        # Save YOLO segmentation annotations
        if yolo_seg_annotations:
            output_path = output_dir / 'labels' / json_path.name.replace('.json', '.txt')
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_seg_annotations))

def organize_yolo_dataset(data_dir):
    data_path = Path(data_dir)
    
    # Create directories
    (data_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (data_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (data_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (data_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    all_images = list((data_path / 'images').glob('*.png'))
    all_labels = list((data_path / 'labels').glob('*.txt'))
    
    
    # Split while maintaining image-label correspondence
    train_images, val_images = train_test_split(
        all_images, test_size=0.2, random_state=42
    )
    train_labels, val_labels = train_test_split(
        all_labels, test_size=0.2, random_state=42
    )
    # Move files function
    def move_files(files, target_dir):
        for f in files:
            if f.exists():
                shutil.move(str(f), str(target_dir / f.name))

    # Move training files
    move_files(train_images, data_path / 'images' / 'train')
    move_files(train_labels, data_path / 'labels' / 'train')

    # Move validation files
    move_files(val_images, data_path / 'images' / 'val')
    move_files(val_labels, data_path / 'labels' / 'val')

    print(f"Moved {len(train_images)} training and {len(val_images)} validation samples")

def x():
     # Define input directories
    input_dirs = [
        "geotiffs/tier3/images"
    ]

    os.makedirs("datasets/yolo_dataset/images", exist_ok=True)
    for input_dir in input_dirs:
        for tif_path in glob.glob(os.path.join(input_dir, "*.tif")):
            if "post" in tif_path and  random.randint(1, 100) > 4:
                continue
            try:
                # Use imageio to read the TIFF file
                arr = imageio.imread(tif_path)
                # If the array is not of type uint8, normalize it
                if arr.dtype != np.uint8:
                    arr = arr.astype('float')
                    arr = 255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
                    arr = arr.astype(np.uint8)
                # Create a PIL Image from the array
                img = Image.fromarray(arr)
                base_name = os.path.splitext(os.path.basename(tif_path))[0]
                png_path = os.path.join("datasets/yolo_dataset/images", base_name + ".png")
                img.save(png_path, "PNG")
                print(f"Converted {tif_path} to {png_path}")
            except Exception as e:
                print(f"Error converting {tif_path}: {e}")

def main():

    x()

    parser = argparse.ArgumentParser(description='Convert xView2 annotations to YOLO format')
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Path to directory containing TIFF images')
    parser.add_argument('--labels-dir', type=str, required=True,
                        help='Path to directory containing JSON annotations')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for YOLO-formatted dataset')
    
    args = parser.parse_args()
    
    convert_to_yolo_seg_format(
        Path(args.images_dir),
        Path(args.labels_dir),
        Path(args.output_dir)
    )

    organize_yolo_dataset('datasets/yolo_dataset')



if __name__ == '__main__':
    main()

