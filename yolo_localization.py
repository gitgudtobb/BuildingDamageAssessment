import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

os.environ['YOLO_VERBOSE'] = 'false'

import argparse
import time
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.data import build
import random

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, data=None, task="train", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)
        self.train_mode = "train" in self.prefix
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts
        self.class_weights = np.array(class_weights)
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        self.counts = np.zeros(self.data['nc'], dtype=int)
        for label in self.labels:
            cls = label['cls'].astype(int)
            if cls.size > 0:
                self.counts[cls] += 1
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_probabilities(self):
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            if cls.size == 0:
                weights.append(1)
                continue
            weight = np.mean(self.class_weights[cls])
            weights.append(weight)
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        return probabilities

    def __getitem__(self, index):
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            # Sample index based on pre-calculated probabilities
            sampled_index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(sampled_index))

def build_weighted_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    dataset = YOLOWeightedDataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        data=data,
        task=cfg.task
    )

# Monkey patch the build_yolo_dataset function
build.build_yolo_dataset = build_weighted_yolo_dataset

class YOLODamageLocalization:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device.type}")
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
            'path': '/home/gitgud/BuildingDamageAssessment/datasets/yolo_dataset',
            'train': 'images/train',
            'val': 'images/val',
            'names': ['building'],
            'nc': 1,
        }

        config_path = Path(self.args.data_dir) / 'assess.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(data_config, f)

        self.args.data = str(config_path)
        print(f"Data YAML written to: {config_path}")

    def train(self):
        train_args = {
            'data': self.args.data,
            'epochs': self.args.epochs,
            'batch': self.args.batch_size,
            'lr0': self.args.lr,
            'lrf': 0.01,
            'device': self.device.type,
            'imgsz': 1024,
            'workers': 8,
            'optimizer': 'AdamW',
            'seed': 42,
            'pretrained': True,
            'verbose': False,
            'project': 'instance_segmentation',
            'name': self.args.encoder + '_seg',
            'task': 'segment',
            'weight_decay': 0.001,

            'degrees': 15.0,
            'scale': 0.5,
            'shear': 3.0,
            'flipud': 0.5,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'mosaic': 1.0,
            'mixup': 0.2,
            'copy_paste': 0.2,
            'erasing': 0.1,

            'dropout': 0.1,
        }

        # Calculate class weights based on the training data
        try:
            dataset = YOLOWeightedDataset(
                img_path=Path(self.args.data_dir) / 'images' / 'train',
                imgsz=train_args['imgsz'],
                batch_size=train_args['batch'],
                augment=True,
                hyp=train_args,
                rect=False,
                cache=False,
                single_cls=False,
                stride=32,
                data=self.args.data,
                task='segment'
            )
            class_counts = dataset.counts
            total_instances = np.sum(class_counts)
            class_weights_loss = total_instances / (len(class_counts) * (class_counts + 1e-6))
            train_args['class_weights'] = class_weights_loss.tolist()
            print(f"Calculated class weights for loss: {train_args['class_weights']}")
        except Exception as e:
            print(f"Error calculating class weights: {e}")
            print("Continuing training without explicit class weights in loss.")
            if 'class_weights' in train_args:
                del train_args['class_weights']

        print("Starting training with the following parameters:")
        for k, v in train_args.items():
            print(f"{k}: {v}")

        # Train the model
        results = self.model.train(**train_args)

        print("\nTraining Results Summary:")
        if 'metrics/mAP50(B)' in results.results_dict:
            print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
        if 'metrics/precision(B)' in results.results_dict:
            print(f"Best Precision: {results.results_dict['metrics/precision(B)']:.3f}")
        if 'metrics/recall(B)' in results.results_dict:
            print(f"Best Recall: {results.results_dict['metrics/recall(B)']:.3f}")

        return results

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Instance Segmentation Training')
    parser.add_argument('--data_dir', type=str, default='datasets/yolo_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--encoder', type=str, default='l',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO architecture version (n, s, m, l, x)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()

    # Setup training
    trainer = YOLODamageLocalization(args)

    # Start training
    print(f'Starting instance segmentation training with {args.encoder} encoder...')
    start_time = time.time()
    results = trainer.train()

    print("\nFinal Training Metrics:")
    if hasattr(results, 'best'):
        print(f"Best Model: {Path(results.best).name}")
    print(f"Training Duration: {results.training_duration:.1f} seconds")
    if 'metrics/mAP50(B)' in results.results_dict:
        print(f"Final mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")

    duration = time.time() - start_time

    print(f'\nTraining completed in {duration:.1f} seconds')
    print(f'Best model saved to: {trainer.model.ckpt_dir}')
    print(f'TensorBoard logs can be viewed with: tensorboard --logdir {trainer.model.ckpt_dir}/tensorboard_logs')

if __name__ == '__main__':
    main()