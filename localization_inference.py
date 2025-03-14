import argparse
import os

import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torchvision.utils import save_image # type: ignore

from localization_model import LocalizationModel
from train_localization import DamageLocalizationDataset


def infer(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad(): # Disable gradient calculations during inference
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            filenames = batch['filename']

            outputs = model(images)

            preds = (outputs > 0.8).float().cpu() # Apply threshold and move to CPU

            for i in range(preds.shape[0]):
                mask_pred = preds[i]
                filename = filenames[i]

                # Save the predicted mask
                output_mask_path = str(os.path.join(output_dir, filename.replace(".tif", "_mask.png")))
                save_image(mask_pred, output_mask_path) # Save mask as PNG image

                print(f'Saved mask: {output_mask_path}')


def main():
    parser = argparse.ArgumentParser(description='Damage Localization Inference')
    parser.add_argument('--data-dir', type=str, default='geotiffs/tier1',
                        help='Path to dataset directory')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        choices=['resnet34', 'convnext_base'],
                        help='Encoder architecture')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Input batch size for inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Path to save predicted masks')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create model
    model = LocalizationModel(encoder_name=args.encoder).to(device)

    # Load trained model weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

    # Data loaders
    infer_dataset = DamageLocalizationDataset(
        os.path.join(args.data_dir, 'images'),
        os.path.join(args.data_dir, 'labels'),
        is_train=False  # Disable augmentations for inference
    )

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle for inference
        num_workers=2,
        pin_memory=True
    )

    # Run inference
    infer(model, infer_loader, device, args.output_dir)

    print(f'Inference complete. Predicted masks saved to: {args.output_dir}')


if __name__ == '__main__':
    main()