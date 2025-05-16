""" =================================================

# @Time: 2024/9/2 11:28

# @Author: Gringer

# @File: function.py

# @Software: PyCharm

================================================== """
import logging
import os
import random
from collections import Counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
from torchvision import datasets, transforms
import seaborn as sns

import torch.onnx


def export_model_to_onnx(model, device, log_dir):
    # 定义一个假输入张量，形状应与模型期望的输入一致
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    onnx_export_path = os.path.join(log_dir, "model.onnx")

    # 导出模型为 ONNX 格式
    torch.onnx.export(
        model,  # 要导出的模型
        dummy_input,  # 模型的输入
        onnx_export_path,  # ONNX 文件的路径
        export_params=True,  # 导出时包含训练好的权重
        opset_version=11,  # ONNX 的操作集版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 输入的名称
        output_names=['output'],  # 输出的名称
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 动态轴设置
    )
    logging.info(f"Model exported to {onnx_export_path}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(dataloaders, num_classes, device):
    """Calculate class weights based on the frequency of each class."""
    all_labels = []
    for inputs, labels in dataloaders['train']:
        all_labels.extend(labels.tolist())

    # Count the number of samples per class
    label_count = Counter(all_labels)
    class_counts = torch.tensor([label_count[i] for i in range(num_classes)], dtype=torch.float)

    # Calculate class weights: inverse of class frequency
    class_weights = 1.0 / (class_counts + 1e-6)

    # Normalize weights so that they sum to 1
    class_weights = class_weights / class_weights.sum()

    # Return the class weights as a tensor
    return class_weights.to(device)


# def prepare_data(data_dir, batch_size=32):
#     # Data transformations
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
#
#     # Loading data using ImageFolder
#     full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
#
#     # 手动打乱数据集
#     indices = torch.randperm(len(full_dataset)).tolist()  # 生成随机索引
#     shuffled_dataset = torch.utils.data.Subset(full_dataset, indices)  # 按随机索引创建新的数据集
#
#     # Splitting the shuffled dataset
#     train_size = int(0.8 * len(shuffled_dataset))
#     val_size = int(0.1 * len(shuffled_dataset))
#     test_size = len(shuffled_dataset) - train_size - val_size
#
#     train_dataset, val_dataset, test_dataset = random_split(shuffled_dataset, [train_size, val_size, test_size])
#
#     # Applying different transforms to each dataset
#     train_dataset.dataset.transform = data_transforms['train']
#     val_dataset.dataset.transform = data_transforms['val']
#     test_dataset.dataset.transform = data_transforms['test']
#
#     # Creating data loaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#
#     dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
#     dataset_sizes = {'train': train_size, 'val': val_size, 'test': test_size}
#     class_names = full_dataset.classes
#     logging.info(f'Dataset sizes: {dataset_sizes}, Class names: {class_names}')
#
#     return dataloaders, dataset_sizes, class_names
def prepare_data(data_dir, batch_size=32):
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),  # 增加旋转增强
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Loading datasets separately
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
    class_names = train_dataset.classes
    logging.info(f'Dataset sizes: {dataset_sizes}, Class names: {class_names}')

    return dataloaders, dataset_sizes, class_names


def save_model_summary(model, input_size, save_path):
    summary_str = str(summary(model, input_size))
    print(model)
    with open(save_path, 'w') as f:
        f.write(summary_str)


def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 8))

    # 计算占比
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    # 构建带有数量和占比的注释
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            annot[i, j] = f'{c}\n{p:.1f}%'

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    if save_path:
        plt.savefig(save_path)
    plt.close()


class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        os.makedirs(os.path.dirname(self.save_path),exist_ok=True)
        print(os.path.exists(os.path.dirname(self.save_path)))
        torch.save(model.state_dict(), self.save_path)

        logging.info(f'Saving model with validation loss: {val_loss:.4f} as best model')
