import os
import random
import numpy as np
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from localization_model import LocalizationModel, combined_loss
from augmentations import load_image, load_mask, apply_transforms
from torch_ema import ExponentialMovingAverage

class DamageLocalizationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, is_train=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.is_train = is_train  # Boolean flag for training mode

        self.file_list = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith("_pre_disaster.tif") or (f.endswith("_post_disaster.tif") and random.randint(1, 100) <= 4)])
        
        split_idx = int(0.8 * len(self.file_list))
        self.file_list = self.file_list[:split_idx] if is_train else self.file_list[split_idx:]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_path = os.path.join(self.images_dir, file_name)
        label_file = (file_name.replace("_pre_disaster.tif", "_pre_disaster.json")
                      .replace("_post_disaster.tif", "_post_disaster.json"))
        label_path = os.path.join(self.labels_dir, label_file)

        image = load_image(image_path)
        mask = load_mask(label_path, img_shape=image.shape[:2])

        if np.mean(mask) == 0 and random.randint(1,20) == 1:
            return self[np.random.randint(len(self))]

        image_tensor, mask_tensor = apply_transforms(
            image,
            mask,
            augment=self.is_train
        )

        return {
            'image': image_tensor,
            'mask': mask_tensor.float(),
            'filename': file_name
        }


def train_epoch(model, dataloader, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    dice_score = 0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            preds = (outputs > 0.5).float()
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum()
            dice_score += (2 * intersection) / (union + 1e-6)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{num_batches} '
                  f'({100. * batch_idx / num_batches:.0f}%)]\tLoss: {loss.item():.4f}')

            # TensorBoard logging
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/dice', dice_score / (batch_idx + 1), global_step)

    avg_loss = total_loss / num_batches
    avg_dice = dice_score.item() / num_batches
    return avg_loss, avg_dice


def validate(model, dataloader, device, epoch, writer):
    model.eval()
    total_loss = 0
    dice_score = 0
    num_batches = len(dataloader)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            with ema.average_parameters():
                outputs = model(images)
                loss = combined_loss(outputs, masks)

                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dice_score += (2 * intersection) / (union + 1e-6)

    avg_loss = total_loss / num_batches
    avg_dice = dice_score.item() / num_batches

    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/dice', avg_dice, epoch)
    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description='Damage Localization Training')
    parser.add_argument('--data-dir', type=str, default='geotiffs/tier1',
                        help='Path to dataset directory')
    parser.add_argument('--encoder', type=str, default='convnext_base',
                        choices=['resnet34', 'convnext_base'],
                        help='Encoder architecture')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create model
    model = LocalizationModel(encoder_name=args.encoder).to(device)

    # Data loaders
    train_dataset = DamageLocalizationDataset(
        os.path.join(args.data_dir, 'images'),
        os.path.join(args.data_dir, 'labels'),
        is_train=True  # Set training mode
    )
    val_dataset = DamageLocalizationDataset(
        os.path.join(args.data_dir, 'images'),
        os.path.join(args.data_dir, 'labels'),
        is_train=False  # Disable augmentations
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    last_lr = args.lr

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join('runs', args.encoder))

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Training
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, device, epoch, writer
        )

        # Validation
        val_loss, val_dice = validate(model, val_loader, device, epoch, writer)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f'\nLearning rate reduced from {last_lr:.2e} to {current_lr:.2e}')
            last_lr = current_lr

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, f'best_{args.encoder}_localization.pth')

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch} completed in {epoch_time:.1f}s')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}\n')

    writer.close()


if __name__ == '__main__':
    main()