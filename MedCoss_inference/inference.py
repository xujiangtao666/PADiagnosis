import os
import numpy as np
import torch
import nibabel as nib
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import argparse
from tqdm import tqdm

# 导入模型和数据加载器
from model.Unimodel import Unified_Model
from dataloader import RICORD_Dataset, CustomDataset


def parse_args():
    parser = argparse.ArgumentParser(description="推理MedCoSS在RICORD数据集上的表现")
    parser.add_argument("--checkpoint_path", type=str,
                        default="snapshots/downstream/dim_3/3D_RICORD_MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/seed_0/lr_0.00001/checkpoint.pth",
                        help="模型权重路径")
    parser.add_argument("--data_path", type=str,
                        default="/path/to/your/data",
                        help="数据集路径")
    parser.add_argument("--test_list", type=str,
                        default="RICORD_test.txt",
                        help="测试集列表文件")
    parser.add_argument("--input_size", type=str,
                        default="64,192,192",
                        help="输入大小")
    parser.add_argument("--batch_size", type=int,
                        default=1,
                        help="批次大小")
    parser.add_argument("--num_classes", type=int,
                        default=2,
                        help="类别数量")
    parser.add_argument("--use_custom_dataset", type=bool,
                        default=False,
                        help="是否使用自定义数据集(True)或RICORD数据集(False)")
    parser.add_argument("--label_csv", type=str,
                        default="/path/to/test_labels.csv",
                        help="如果使用自定义数据集，提供标签CSV文件路径")
    return parser.parse_args()


def inference():
    args = parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 解析输入大小
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    # 初始化模型
    print("加载模型...")
    model = Unified_Model(now_3D_input_size=input_size,
                          num_classes=args.num_classes,
                          pre_trained=False)

    # 加载权重
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print(f"模型已加载，来自: {args.checkpoint_path}")

    # 准备数据集
    if args.use_custom_dataset:
        print("使用自定义数据集进行推理...")
        test_dataset = CustomDataset(
            data_dir=args.data_path,
            label_csv=args.label_csv,
            transform=None,
            num_classes=args.num_classes
        )
    else:
        print("使用RICORD数据集进行推理...")
        test_dataset = RICORD_Dataset(
            args.data_path,
            list_path=args.test_list,
            crop_size_3D=input_size,
            split="test"
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"数据集加载完成，共{len(test_dataset)}个样本")

    # 进行推理
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="推理中"):
            inputs = inputs.to(device)

            # 对于RICORD数据集
            if not args.use_custom_dataset:
                # 为单类别分类准备数据
                labels = labels.long().to(device)
                data = {"data": inputs, "labels": labels, "modality": "3D image"}

                # 获取预测结果
                if hasattr(model, 'cal_acc'):
                    model.cal_acc = True
                    _, logits = model(data)
                else:
                    logits = model(data)

                # 单类别分类的处理
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                y_true = labels.cpu().numpy()

            # 对于自定义数据集（多标签）
            else:
                # 为多标签分类准备数据
                labels = labels.float().to(device)
                data = {"data": inputs, "labels": labels, "modality": "3D image"}

                # 获取预测结果
                logits = model(data)

                # 多标签分类的处理
                probs = torch.sigmoid(logits).cpu().numpy()
                y_true = labels.cpu().numpy()

            predictions.append(probs)
            true_labels.append(y_true)

    # 合并所有批次的预测和真实标签
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # 评估结果
    if args.num_classes == 2 and not args.use_custom_dataset:
        # 二分类评估（单标签）
        y_pred = (predictions[:, 1] >= 0.5).astype(int)
        accuracy = accuracy_score(true_labels, y_pred)
        auc = roc_auc_score(true_labels, predictions[:, 1])
        f1 = f1_score(true_labels, y_pred)

        print(f"测试集评估结果:")
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1分数: {f1:.4f}")

    else:
        # 多标签分类评估
        y_pred = (predictions >= 0.5).astype(int)

        # 计算评估指标
        exact_match = accuracy_score(true_labels, y_pred)
        auc_micro = roc_auc_score(true_labels, predictions, average='micro')
        auc_macro = roc_auc_score(true_labels, predictions, average='macro')
        f1_micro = f1_score(true_labels, y_pred, average='micro')
        f1_macro = f1_score(true_labels, y_pred, average='macro')

        print(f"测试集评估结果:")
        print(f"完全匹配准确率: {exact_match:.4f}")
        print(f"AUC (micro): {auc_micro:.4f}")
        print(f"AUC (macro): {auc_macro:.4f}")
        print(f"F1 (micro): {f1_micro:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")

    # 保存预测结果到文件
    result_dir = os.path.dirname(args.checkpoint_path)
    np.save(os.path.join(result_dir, "predictions.npy"), predictions)
    np.save(os.path.join(result_dir, "true_labels.npy"), true_labels)
    print(f"预测结果已保存到: {result_dir}")


def inference_single_case(nifti_path, checkpoint_path, input_size=(64, 192, 192), num_classes=2):
    """
    对单个NIfTI文件进行推理

    参数:
    - nifti_path: NIfTI文件路径
    - checkpoint_path: 模型权重路径
    - input_size: 输入大小，默认(64, 192, 192)
    - num_classes: 类别数量，默认2

    返回:
    - 预测概率
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = Unified_Model(now_3D_input_size=input_size,
                          num_classes=num_classes,
                          pre_trained=False)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 加载并预处理NIfTI文件
    nii = nib.load(nifti_path)
    image = nii.get_fdata()
    image = image.transpose((2, 0, 1))

    # 进行截断操作（与训练时相同）
    min_HU = -1024
    max_HU = 325
    subtract = 158.58
    divide = 324.70

    image[np.where(image <= min_HU)] = min_HU
    image[np.where(image >= max_HU)] = max_HU
    image = image - subtract
    image = image / divide

    # 增加batch和channel维度
    image = image[np.newaxis, np.newaxis, :]
    image_tensor = torch.tensor(image, dtype=torch.float32).to(device)

    # 进行推理
    with torch.no_grad():
        data = {"data": image_tensor, "modality": "3D image"}
        output = model(data)

        # 根据类别数量确定处理方式
        if num_classes == 2:
            # 二分类
            probs = torch.softmax(output, dim=1).cpu().numpy()
            pred_class = np.argmax(probs, axis=1)[0]
            confidence = probs[0, pred_class]
            print(f"预测类别: {pred_class}, 置信度: {confidence:.4f}")
        else:
            # 多标签分类
            probs = torch.sigmoid(output).cpu().numpy()
            pred_classes = (probs >= 0.5).astype(int)[0]
            print(f"预测类别: {pred_classes}")
            for i, prob in enumerate(probs[0]):
                print(f"类别 {i}: 概率 {prob:.4f}")

    return probs


if __name__ == "__main__":
    inference()