import torch
from torchvision import transforms
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

from localization_model import LocalizationModel
from train_classification import TwinResNet  # TwinResNet modelinizi burada import edin

# Renk eşleme (damage categories)
categories = {0: 'no-damage', 1: 'minor-damage', 2: 'major-damage', 3: 'destroyed'}
color_mapping = {
    0: (0, 255, 0),    # no-damage: green
    1: (255, 255, 0),  # minor-damage: yellow
    2: (255, 165, 0),  # major-damage: orange
    3: (255, 0, 0)     # destroyed: red
}

def load_image(image_path):
    """Load an image from the given path."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def detect_and_color_damage(pre_image_path, post_image_path, localization_model, damage_model, device):
    """Detect buildings, predict damage, and color the post-disaster image."""
    # Load images
    pre_img = load_image(pre_image_path)
    post_img = load_image(post_image_path)

    # Preprocess images for models
    pre_img_tensor = preprocess_image(pre_img).to(device)
    post_img_tensor = preprocess_image(post_img).to(device)

    # Get building mask using the localization model
    with torch.no_grad():
        building_mask = (localization_model(post_img_tensor) > 0.8).float().cpu().squeeze().numpy()

    # Predict damage category using the TwinResNet model
    with torch.no_grad():
        damage_outputs = damage_model(pre_img_tensor, post_img_tensor)
        _, damage_preds = torch.max(damage_outputs, 1)
        damage_category = damage_preds.item()

    # Apply color to the post-disaster image based on the predicted category
    overlay = np.zeros_like(post_img)
    overlay[:] = color_mapping[damage_category]
    alpha = 0.5  # Transparency factor

    # Apply the color only to the building areas
    post_img_colored = post_img.copy()
    post_img_colored[building_mask == 1] = cv2.addWeighted(
        post_img_colored[building_mask == 1], 1 - alpha,
        overlay[building_mask == 1], alpha, 0
    )

    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(pre_img)
    plt.title("Pre-disaster")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(post_img_colored)
    plt.title(f"Post-disaster: {categories[damage_category]}")
    plt.axis('off')

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Detect and color damage in buildings')
    parser.add_argument('--pre-image', type=str, required=True, default= "socal-fire_00000360_pre_disaster.png",
                        help='Path to the pre-disaster image') 
    parser.add_argument('--post-image', type=str, required=True, default= "socal-fire_00000360_post_disaster.png",
                        help='Path to the post-disaster image')
    parser.add_argument('--localization-checkpoint', type=str, required=True, default="best_resnet34_localization.pth",
                        help='Path to the trained localization model checkpoint')
    parser.add_argument('--damage-checkpoint', type=str, required=True, default="best_twin_resnet.pth",
                        help='Path to the trained damage classification model checkpoint')
    args = parser.parse_args()

    # pre-image geotiffs/tier1/images/socal-fire_00000387_pre_disaster.png --post-image geotiffs/tier1/images/socal-fire_00000387_post_disaster.png --localization-checkpoint best_resnet34_localization.pth --damage-checkpoint best_twin_resnet.pth
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load models
    localization_model = LocalizationModel(encoder_name='resnet34').to(device)
    damage_model = TwinResNet(num_classes=4).to(device)

    # Load trained model weights
    localization_checkpoint = torch.load(args.localization_checkpoint, map_location=device)
    localization_model.load_state_dict(localization_checkpoint['model_state_dict'])
    print(f"Loaded localization checkpoint from epoch {localization_checkpoint['epoch']}, Loss: {localization_checkpoint['loss']:.4f}")

    damage_checkpoint = torch.load(args.damage_checkpoint, map_location=device)
    if 'model_state_dict' in damage_checkpoint:
        damage_model.load_state_dict(damage_checkpoint['model_state_dict'])
    else:
        damage_model.load_state_dict(damage_checkpoint)  # Directly load the state_dict
    print(f"Loaded damage checkpoint")
    #print(f"Loaded damage checkpoint from epoch {damage_checkpoint['epoch']}, Loss: {damage_checkpoint['loss']:.4f}")

    # Run detection and coloring
    detect_and_color_damage(
        args.pre_image, args.post_image,
        localization_model, damage_model, device
    )


if __name__ == '__main__':
    main()