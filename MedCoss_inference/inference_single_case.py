import os
import argparse
import numpy as np
import torch
import nibabel as nib

from model.Unimodel import Unified_Model


def infer_num_classes_from_ckpt(state_dict, fallback=2):
    """
    从 checkpoint 的线性层权重推断类别数（out_features）。
    优先选择包含 head/fc/classifier 且 out!=in 的层，否则取 out!=in 的最小值，再退回最小 out。
    """
    candidates = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 2 and k.endswith("weight"):  # Linear: [out, in]
            out_f, in_f = int(v.shape[0]), int(v.shape[1])
            tag = "pri" if any(t in k.lower() for t in ["head", "classifier", "fc"]) else "sec"
            candidates.append((tag, out_f, in_f, k))

    if not candidates:
        return int(fallback)

    pri = [c for c in candidates if c[0] == "pri" and c[1] != c[2]]
    if pri:
        return int(pri[0][1])

    diff = [c for c in candidates if c[1] != c[2]]
    if diff:
        diff.sort(key=lambda x: x[1])
        return int(diff[0][1])

    candidates.sort(key=lambda x: x[1])
    return int(candidates[0][1])


def build_model(checkpoint_path, input_size, num_classes=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    actual_num_classes = num_classes if num_classes is not None else infer_num_classes_from_ckpt(state, fallback=2)
    print(f"[info] 推断到 num_classes = {actual_num_classes}")

    model = Unified_Model(now_3D_input_size=input_size, num_classes=actual_num_classes, pre_trained=False)
    msg = model.load_state_dict(state, strict=True)
    print(f"[info] load_state_dict msg: {msg}")
    model.to(device)
    model.eval()
    if hasattr(model, "cal_acc"):
        model.cal_acc = False
    return model, device, actual_num_classes


def load_nifti_as_chw(nifti_path):
    nii = nib.load(nifti_path)
    image = nii.get_fdata()
    image = image.transpose((2, 0, 1))  # [D,H,W]
    return image


def truncate_ct_like_dataset(image):
    # 与 RICORD_Dataset.truncate 一致
    min_HU = -1024
    max_HU = 325
    subtract = 158.58
    divide = 324.70
    image = np.clip(image, min_HU, max_HU)
    image = (image - subtract) / divide
    return image


def preprocess_single_nifti(nifti_path, dataset):
    img = load_nifti_as_chw(nifti_path)
    if dataset == "ricord":
        img = truncate_ct_like_dataset(img)
    # [B,C,D,H,W]
    img = img[np.newaxis, np.newaxis, :]
    return img.astype(np.float32)


def inference_single_case(nifti_path, checkpoint_path, input_size, dataset="custom", num_classes=None):
    assert os.path.isfile(nifti_path), f"找不到文件 `{nifti_path}`"
    assert os.path.isfile(checkpoint_path), f"找不到文件 `{checkpoint_path}`"

    model, device, actual_num_classes = build_model(checkpoint_path, input_size, num_classes=num_classes)

    x = preprocess_single_nifti(nifti_path, dataset)
    x = torch.from_numpy(x).to(device)

    with torch.no_grad():
        data = {"data": x, "modality": "3D image"}
        logits = model(data)
        if dataset == "custom":
            probs = torch.sigmoid(logits)                  # 多标签
        else:
            probs = torch.softmax(logits, dim=1)           # 单/多类
        probs = probs.cpu().numpy()

    print(f"[done] num_classes={actual_num_classes}, probs shape={probs.shape}")
    print("probs:", probs[0])
    return probs


def parse_args():
    p = argparse.ArgumentParser(description="一致的推理脚本（单例）")
    p.add_argument("--mode", type=str, default="single", choices=["single"])
    p.add_argument("--dataset", type=str, default="custom", choices=["custom", "ricord"])
    p.add_argument("--nifti_path", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--input_size", type=str, default="64,192,192")
    p.add_argument("--num_classes", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    d, h, w = map(int, args.input_size.split(","))
    input_size = (d, h, w)

    inference_single_case(
        nifti_path=args.nifti_path,
        checkpoint_path=args.checkpoint_path,
        input_size=input_size,
        dataset=args.dataset,
        num_classes=args.num_classes
    )