import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import argparse
import time
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter

class YOLODamageLocalization:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device.type)
        self.model = self._init_model()
        self._prepare_data_yaml()

    def _init_model(self):
        model_name = f'yolo11{self.args.encoder}-seg.pt'
        print(f"Initializing YOLO model: {model_name}")
        model = YOLO(model_name)
        model.to(self.device)
        return model

    def _prepare_data_yaml(self):
        data_config = {
            'path': 'C:/Users/yusuf/Documents/GitHub/BuildingDamageAssessment/datasets/yolo_dataset',
            'train': 'images/train',
            'val': 'images/val',
            'names': ['building'],
            'nc': 1
        }

        config_path = Path(self.args.data_dir) / 'buildings.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(data_config, f)

        self.args.data = str(config_path)

    def train(self):
        train_args = {
            'data': self.args.data,
            'epochs': self.args.epochs,
            'batch': self.args.batch_size,
            'lr0': self.args.lr,
            'device': self.device.type,
            'imgsz': 640,
            'workers': 8,
            'optimizer': 'AdamW',
            'seed': 42,
            'pretrained': True,
            'verbose': True,
            'project': 'yolo_building_localization',
            'name': self.args.encoder + '_yolo',
             
            'degrees': 45.0,      
            'translate': 0.2,     
            'scale': 0.5,
            'shear': 15.0,        
            'perspective': 0.0015,
            'flipud': 0.5,        
            'fliplr': 0.5,        
            
            # Color Transformations
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            
            # Advanced Augmentations
            'mosaic': 1.0,
            'mixup': 0.2,
            'copy_paste': 0.2,
            'erasing': 0.1,
            
            # Regularization
            'label_smoothing': 0.1,
            'dropout': 0.1,
        }

        # Train and capture results
        results = self.model.train(**train_args)

        # Process and display results
        print("\nTraining Results Summary:")
        print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
        print(f"Best Precision: {results.results_dict['metrics/precision(B)']:.3f}")
        print(f"Best Recall: {results.results_dict['metrics/recall(B)']:.3f}")

        # Save metrics to file
        metrics_path = Path(self.model.ckpt_dir) / 'training_metrics.csv'
        results.results_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")


        writer = SummaryWriter(log_dir=Path(self.model.ckpt_dir) / 'tensorboard_logs')

        metrics_dict = results.results_dict
        if metrics_dict:
            writer.add_scalar('mAP50', metrics_dict.get('metrics/mAP50(B)', 0), self.args.epochs)
            writer.add_scalar('Precision', metrics_dict.get('metrics/precision(B)', 0), self.args.epochs)
            writer.add_scalar('Recall', metrics_dict.get('metrics/recall(B)', 0), self.args.epochs)
            writer.add_scalar('box_loss', metrics_dict.get('train/box_loss', 0), self.args.epochs)
            writer.add_scalar('cls_loss', metrics_dict.get('train/cls_loss', 0), self.args.epochs)
            writer.add_scalar('dfl_loss', metrics_dict.get('train/dfl_loss', 0), self.args.epochs)

        writer.close()
        print(f"TensorBoard logs saved to: {Path(self.model.ckpt_dir) / 'tensorboard_logs'}")


        return results

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Building Localization Training')
    parser.add_argument('--data_dir', type=str, default='datasets/yolo_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--encoder', type=str, default='m',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO architecture version (n, s, m, l, x)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()

    # Setup training
    trainer = YOLODamageLocalization(args)

    # Start training
    print(f'Starting YOLOv11 training with {args.encoder} encoder...')
    start_time = time.time()
    results = trainer.train()

    print("\nFinal Training Metrics:")
    print(f"Best Model: {Path(results.best).name}")
    print(f"Training Duration: {results.training_duration:.1f} seconds")
    print(f"Final mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")

    results.plot_labels()
    results.plot_results()
    duration = time.time() - start_time

    print(f'\nTraining completed in {duration:.1f} seconds')
    print(f'Best model saved to: {trainer.model.ckpt_dir}')
    print(f'TensorBoard logs can be viewed with: tensorboard --logdir {trainer.model.ckpt_dir}/tensorboard_logs')


if __name__ == '__main__':
    main()