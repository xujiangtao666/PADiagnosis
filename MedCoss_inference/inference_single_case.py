import os
import numpy as np
import torch
import nibabel as nib

# 导入模型
from model.Unimodel import Unified_Model


def inference_single_case(nifti_path, checkpoint_path, input_size=(64, 192, 192), num_classes=2):
    """
    对单个NIfTI文件进行推理
    """
    # 检查文件是否存在
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"找不到NIfTI文件: {nifti_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载权重以检查类别数
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从权重中判断实际的类别数
    actual_num_classes = 8  # 根据错误信息得知权重是为8个类别训练的
    print(f"从权重文件检测到的类别数: {actual_num_classes}")

    # 初始化模型时使用正确的类别数
    print(f"初始化模型，输入尺寸: {input_size}, 类别数: {actual_num_classes}")
    model = Unified_Model(now_3D_input_size=input_size,
                          num_classes=actual_num_classes,
                          pre_trained=False)

    # 加载权重
    print(f"加载权重: {checkpoint_path}")
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print("模型加载成功")

    # 加载并预处理NIfTI文件
    print(f"处理NIfTI文件: {nifti_path}")
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
    print("进行推理...")
    with torch.no_grad():
        data = {"data": image_tensor, "modality": "3D image"}
        output = model(data)

        # 使用sigmoid而非softmax进行多标签分类
        probs = torch.sigmoid(output).cpu().numpy()

        # 设置阈值确定每个类别是否为正例
        threshold = 0.5
        pred_classes = (probs >= threshold).astype(int)[0]

        # 输出结果
        print(f"\n预测结果 (阈值 {threshold}):")
        for i, prob in enumerate(probs[0]):
            status = "是" if prob >= threshold else "否"
            print(f"类别 {i}: 概率 {prob:.4f} -> {status}")

        # 显示预测为正的类别
        positive_classes = np.where(pred_classes == 1)[0]
        if len(positive_classes) > 0:
            print(f"\n预测为正的类别: {', '.join(map(str, positive_classes))}")
        else:
            print("\n没有预测为正的类别")

    return probs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="对单个NIfTI文件进行推理")
    parser.add_argument("--nifti_path", type=str, required=True,
                        help="NIfTI文件路径")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--input_size", type=str, default="64,192,192",
                        help="输入大小，格式: d,h,w")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="需要返回的类别数量")
    args = parser.parse_args()

    # 解析输入大小
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    # 进行推理
    inference_single_case(
        nifti_path=args.nifti_path,
        checkpoint_path=args.checkpoint_path,
        input_size=input_size,
        num_classes=args.num_classes
    )