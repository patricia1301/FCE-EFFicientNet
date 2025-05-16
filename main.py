""" =================================================

# @Time: 2024/9/15 21:16

# @Author: Gringer

# @File: 最终结果.py

# @Software: PyCharm

================================================== """
import copy
import csv
import logging
import os
from torchvision.models.efficientnet import SqueezeExcitation  # 官方 SE 类
from torchvision.models import EfficientNet_B0_Weights
from attention import ELA, MultiScaleELAEnhanced
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
from tqdm import tqdm

from function import set_seed, save_model_summary, calculate_class_weights, plot_confusion_matrix, \
    EarlyStopping, plot_training_history, export_model_to_onnx
from loss import FocalCosineLoss, FocalLoss
from model import replace_mbconv_with_ca, replace_mbconv_with_cbam, replace_mbconv_with_eca
# from 网络结构 import model_ft


def configure_logging(log_dir, log_filename='training.log'):
    log_path = os.path.join(log_dir, log_filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )


def train_on_multiple_datasets(root_data_dir, log_root_dir,pretrained=True,weight=0.5):
    datasets = os.listdir(root_data_dir)
    print(datasets)
    for dataset in datasets:
        data_dir = os.path.join(root_data_dir, dataset)
        log_dir = os.path.join(log_root_dir, f'logs_{dataset}')
        log_filename = f'training_{dataset}.log'

        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        try:
            main(data_dir, log_dir, log_filename, pretrained=pretrained,weight=weight)
        except Exception as e:
            logging.error(f"An error occurred during training on dataset {dataset}: {e}")


def prepare_data(data_dir, batch_size=32, test_size=0.2, val_size=0.1):
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

    # Load full dataset without predefined splits
    full_dataset = datasets.ImageFolder(data_dir)

    # Get indices and class names
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    class_names = full_dataset.classes

    # Split dataset into train, test, and validation sets
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42,
                                                   stratify=full_dataset.targets)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size / (1 - test_size), random_state=42,
                                                  stratify=[full_dataset.targets[i] for i in train_indices])

    # Create dataset subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Apply transforms
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['test']

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    logging.info(f'Dataset sizes: {dataset_sizes}, Class names: {class_names}')
    return dataloaders, dataset_sizes, class_names


def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler, num_epochs=25,
                log_dir='./logs'):
    # 检查并创建 log_dir 目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    # 初始化 EarlyStopping 实例
    early_stopping = EarlyStopping(patience=100, delta=0.01, save_path=os.path.join(log_dir, 'best_model.pth'))

    # CSV文件路径
    history_file = os.path.join(log_dir, 'training_history.csv')

    # 初始化CSV文件
    with open(history_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            with tqdm(total=len(dataloaders[phase]), desc=f'{phase} Epoch {epoch}/{num_epochs - 1}',
                      unit='batch') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    pbar.update(1)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.cpu().numpy())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.cpu().numpy())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # 调用调度器时传入验证损失
        scheduler.step(val_losses[-1])
        # 记录学习率
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        logging.info(f'Current learning rate: {current_lr:.6f}')

        early_stopping(val_losses[-1], model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        # 保存每个epoch的数据到CSV文件
        with open(history_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_losses[-1], val_losses[-1], train_accs[-1], val_accs[-1]])

    logging.info(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)

    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                          save_path=os.path.join(log_dir, 'training_history.png'))

    return model


def evaluate_model(model, dataloaders, device, class_names, log_dir=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    logging.info('Test Accuracy: %.4f' % acc)
    logging.info('Test Precision: %.4f' % precision)
    logging.info('Test Recall: %.4f' % recall)
    logging.info('Test F1 Score: %.4f' % f1)
    logging.info('\n' + classification_report(all_labels, all_preds, target_names=class_names))

    if log_dir:
        # 定义文件路径
        result_txt_path = os.path.join(log_dir, 'test_results.txt')

        # 将结果写入文本文件
        with open(result_txt_path, 'w') as f:
            f.write(f'Test Accuracy: {acc:.4f}\n')
            f.write(f'Test Precision: {precision:.4f}\n')
            f.write(f'Test Recall: {recall:.4f}\n')
            f.write(f'Test F1 Score: {f1:.4f}\n')
            f.write('\nClassification Report:\n')
            f.write(classification_report(all_labels, all_preds, target_names=class_names))

        # 保存混淆矩阵图像
        plot_confusion_matrix(cm, class_names, save_path=os.path.join(log_dir, 'confusion_matrix.png'))


def efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1):
    return models.efficientnet_b0(weights=weights)


def efficientnet_b0_with_CA():
    model = efficientnet_b0()
    model = replace_mbconv_with_ca(model)
    return model


def efficientnet_b0_with_CBAM():
    model = efficientnet_b0()
    model = replace_mbconv_with_cbam(model)
    return model


def efficientnet_b0_with_ECA():
    model = efficientnet_b0()
    model = replace_mbconv_with_eca(model)
    return model

def replace_se_with_mse(module: nn.Module):
    """
    遍历任意 nn.Module，遇到 SqueezeExcitation 就替换为 MultiDimELA。
    """

    for name, child in list(module.named_children()):
        if isinstance(child, SqueezeExcitation):
            # —— 关键修正行 ——
            channels = child.fc2.out_channels  # Conv2d -> out_channels

            # 替换
            setattr(module, name, MultiScaleELAEnhanced(channels))
        else:
            replace_se_with_mse(child)
def modify_efficientnet_b0_with_MSE(num_classes=1000, pretrained=False):
    # ① 加载官方模型
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    # ② 替换所有 SE
    replace_se_with_mse(model)

    # ③ 如需迁移学习，改分类器
    if num_classes != 1000:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    print(any(isinstance(m, SqueezeExcitation) for m in model.modules()))  # -> False
    print(sum(isinstance(m, MultiScaleELAEnhanced) for m in model.modules()))  # 替换后数量

    return model


# 冻结卷积层的函数
def freeze_layers(model, num_unfreeze=0):
    """
    冻结 EfficientNet 中的卷积层，参数 `num_unfreeze` 控制解冻的层数：
    - 如果 `num_unfreeze=0`，所有卷积层都被冻结（即只训练全连接层）。
    - 如果 `num_unfreeze=1`，解冻最后一层卷积层。
    - 如果 `num_unfreeze=n`，解冻最后 n 层卷积层。
    """
    # 获取卷积层参数
    conv_layers = list(model.features.parameters())

    # 冻结所有层
    for param in conv_layers:
        param.requires_grad = False

    # 解冻最后 num_unfreeze 层
    if num_unfreeze > 0:
        for param in conv_layers[-num_unfreeze:]:
            param.requires_grad = True


def main(data_dir, log_dir, log_filename, pretrained,weight):
    configure_logging(log_dir, log_filename)
    logging.info("Starting training process...")

    batch_size = 64
    dataloaders, dataset_sizes, class_names = prepare_data(data_dir, batch_size=batch_size)

    num_classes = len(class_names)
    logging.info(f'Number of classes: {num_classes}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    lr = 0.001
    weight_decay = 5e-4
    logging.info(f'Batch_size: {batch_size}')
    logging.info(f'lr: {lr}')
    logging.info(f'weight_decay: {weight_decay}')

    # 创建模型实例
    if pretrained:
        model_ft = modify_efficientnet_b0_with_MSE(pretrained=True)
    else:
        model_ft = modify_efficientnet_b0_with_MSE(pretrained=False)

    # 修改模型的分类器以适应新任务的类别数量
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # 将模型移到设备上
    model_ft = model_ft.to(device)

    # 保存模型结构摘要
    model_summary_path = os.path.join(log_dir, f'model_summary.txt')
    save_model_summary(model_ft, input_size=(3, 224, 224), save_path=model_summary_path)

    # 计算类别权重，并使用 FocalCosineLoss 作为损失函数
    class_weights = calculate_class_weights(dataloaders, num_classes, device)
    print(class_weights)
    criterion = FocalCosineLoss(alpha=class_weights, gamma=2, xent=0.9, reduction="mean", cosine_weight=weight,
                                focal_weight=1.0-weight)

    num_epochs = 100
    optimizer = optim.Adam(model_ft.parameters(), lr=lr, weight_decay=weight_decay)

    # 初始化学习率调度器
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        threshold=0.0001,
        cooldown=0,
        min_lr=1e-6
    )

    # 训练模型
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, device, criterion, optimizer, exp_lr_scheduler,
                           num_epochs=num_epochs, log_dir=log_dir)

    # 评估模型
    evaluate_model(model_ft, dataloaders, device, class_names, log_dir=log_dir)

    logging.info("Training process finished.")


if __name__ == "__main__":
    set_seed(42)
    root_data_dir = r'data/dataset'

    for exp in range(4,8):
        log_root_dir = os.path.join('Outputs_Final_Results',str(exp))


        train_on_multiple_datasets(root_data_dir, os.path.join(log_root_dir,'with_pretrained_Lamba', ),pretrained=True, weight = 0.5)

        train_on_multiple_datasets(root_data_dir, os.path.join(log_root_dir, 'without_pretrained'),pretrained=False)
