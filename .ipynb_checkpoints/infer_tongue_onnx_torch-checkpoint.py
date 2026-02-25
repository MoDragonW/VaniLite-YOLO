# -*- coding: utf-8 -*-
import argparse, os
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h0, w0 = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / w0, new_shape[1] / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    dw /= 2; dh /= 2
    if (w0, h0) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (left, top)  # ratio, pad=(padw,padh)

def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(boxes, scores, iou_thres=0.5, topk=300):
    idxs = scores.argsort(descending=True)
    idxs = idxs[:min(topk, idxs.numel())]
    keep = []
    while idxs.numel():
        i = idxs[0].item()
        keep.append(i)
        if idxs.numel() == 1:
            break
        cur = boxes[i].unsqueeze(0)
        rest = boxes[idxs[1:]]
        inter_x1 = torch.maximum(cur[:, 0], rest[:, 0])
        inter_y1 = torch.maximum(cur[:, 1], rest[:, 1])
        inter_x2 = torch.minimum(cur[:, 2], rest[:, 2])
        inter_y2 = torch.minimum(cur[:, 3], rest[:, 3])
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter = inter_w * inter_h
        area_cur = (cur[:, 2] - cur[:, 0]) * (cur[:, 3] - cur[:, 1])
        area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        iou = inter / (area_cur + area_rest - inter + 1e-6)
        idxs = idxs[1:][iou <= iou_thres]
    return torch.tensor(keep, dtype=torch.long)

def process_masks(protos, masks_in, boxes, input_hw, bin_thres=0.5):
    nm, mh, mw = protos.shape
    ih, iw = input_hw
    masks = (masks_in @ protos.reshape(nm, -1)).sigmoid().reshape(-1, mh, mw)  # (N, mh, mw)
    # 将 boxes 映射到 proto 分辨率并裁剪
    scale_x = mw / float(iw); scale_y = mh / float(ih)
    b = boxes.clone()
    b[:, [0, 2]] *= scale_x; b[:, [1, 3]] *= scale_y
    n = b.size(0)
    ys = torch.arange(mh, device=masks.device).view(1, mh, 1).expand(n, mh, mw)
    xs = torch.arange(mw, device=masks.device).view(1, 1, mw).expand(n, mh, mw)
    x1 = b[:, 0].view(n, 1, 1); y1 = b[:, 1].view(n, 1, 1)
    x2 = b[:, 2].view(n, 1, 1); y2 = b[:, 3].view(n, 1, 1)
    crop = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    masks = masks * crop
    masks = F.interpolate(masks.unsqueeze(1).float(), size=(ih, iw), mode="bilinear", align_corners=False).squeeze(1)
    return (masks > bin_thres).to(torch.uint8)  # (N, ih, iw)

def infer_onnx_seg(
    onnx_path: str,
    img_path: str,
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    max_det: int = 50,
    save_prefix: str = "runs/seg_vis/tongue",
    mask_thr: float = 0.45,
    pre_topk: int = 200,
    center_prior: bool = False,   # 舌头通常居中，可开
    no_crop: bool = False,        # 不按框裁剪，直接用最大掩码
    debug: bool = True
):
    im0 = cv2.imread(img_path); assert im0 is not None, f"读图失败: {img_path}"
    h0, w0 = im0.shape[:2]

    im_in, ratio, pad = letterbox(im0, imgsz)
    ih, iw = im_in.shape[:2]
    rgb = cv2.cvtColor(im_in, cv2.COLOR_BGR2RGB)
    inp = (torch.from_numpy(rgb).float() / 255.0).permute(2, 0, 1).unsqueeze(0)  # (1,3,ih,iw)

    if debug:
        print(f"DEBUG|input sizes: raw(H,W)=({h0},{w0}), letterbox(H,W)=({ih},{iw}), ratio={ratio}, pad={pad}")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_in = {sess.get_inputs()[0].name: inp.numpy()}
    outs = sess.run(None, ort_in)
    if debug:
        print("DEBUG|outs shapes:", [tuple(o.shape) for o in outs])

    # 识别 pred/proto
    a, b = outs[0].shape, outs[1].shape
    if len(a) == 4 and a[1] in (16, 32, 64):
        proto = torch.from_numpy(outs[0]); pred = torch.from_numpy(outs[1])
    else:
        pred = torch.from_numpy(outs[0]); proto = torch.from_numpy(outs[1])

    # pred -> (N,C)
    if pred.ndim != 3: raise RuntimeError(f"不认识的 pred 维度: {pred.shape}")
    pred = pred.squeeze(0).transpose(0, 1) if pred.shape[1] < pred.shape[2] else pred.squeeze(0)

    nm = proto.shape[1]; Hm, Wm = proto.shape[2], proto.shape[3]
    C = pred.shape[1]; nc = C - 4 - nm
    if nc < 1: raise RuntimeError(f"通道数解析失败：C={C}, nm={nm} -> nc={nc}")
    if debug:
        print(f"DEBUG|channels: C={C}, nm(proto)={nm}, nc={nc}; proto(Hm,Wm)=({Hm},{Wm})")

    boxes_raw = pred[:, :4]
    cls_logits = pred[:, 4:4+nc]
    coef = pred[:, 4+nc:]

    # 分数：分类 * 掩码强度
    if nc == 1:
        score_cls = torch.sigmoid(cls_logits).squeeze(1)
    else:
        score_cls = torch.sigmoid(cls_logits).max(1).values

    maskness = coef.abs().mean(1)             # 掩码力度
    score = (score_cls - 0.5).clamp(min=0) + 1e-3  # 把0.5作为“起点”，避免全一样
    score = score * (maskness / (maskness.max() + 1e-6))

    # 可选：中心先验（舌头居中更常见）
    if center_prior:
        cxcy = boxes_raw[:, :2]
        if boxes_raw.max() <= 1.2:  # 归一化
            cxcy = cxcy * float(imgsz)
        dx = (cxcy[:, 0] - iw / 2).abs() / (iw / 2)
        dy = (cxcy[:, 1] - ih / 2).abs() / (ih / 2)
        center_w = (1 - (0.5 * (dx + dy))).clamp(0, 1)
        score = score * (0.5 + 0.5 * center_w)

    # 预筛 top-K
    topk = min(pre_topk, score.numel())
    pre_idx = torch.topk(score, k=topk).indices
    boxes_raw = boxes_raw[pre_idx]; coef = coef[pre_idx]; score = score[pre_idx]; score_cls = score_cls[pre_idx]

    # 坐标归一化判断 & xywh/xyxy 判断
    if boxes_raw.max() <= 1.2:
        boxes_raw = boxes_raw * float(imgsz)
    boxes_xyxy_try = xywh2xyxy(boxes_raw.clone())
    w = (boxes_xyxy_try[:, 2] - boxes_xyxy_try[:, 0]).clamp(min=0)
    h = (boxes_xyxy_try[:, 3] - boxes_xyxy_try[:, 1]).clamp(min=0)
    treat_as_xywh = not(((w == 0).float().mean() > 0.5) or ((h == 0).float().mean() > 0.5))
    boxes_xyxy = boxes_xyxy_try if treat_as_xywh else boxes_raw
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, iw - 1)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, ih - 1)

    # NMS 用新分数
    keep_idx = nms(boxes_xyxy, score, iou_thres=iou_thres, topk=max_det)
    boxes_xyxy = boxes_xyxy[keep_idx]; coef = coef[keep_idx]; score = score[keep_idx]; score_cls = score_cls[keep_idx]

    if debug:
        k = min(5, boxes_xyxy.size(0))
        print(f"DEBUG|pre-NMS topk={topk}, after-NMS={boxes_xyxy.size(0)}")
        print(f"DEBUG|boxes[:{k}] (xyxy in {iw}x{ih}):\n{boxes_xyxy[:k]}")
        print(f"DEBUG|score_cls[:{k}]: {score_cls[:k]}")
        print(f"DEBUG|size chain: proto=({Hm},{Wm}) -> input=({ih},{iw}) -> raw=({h0},{w0})")

    # 生成掩码
    if boxes_xyxy.numel() == 0:
        print("无有效目标，输出空掩码。")
        union = np.zeros((h0, w0), dtype=np.uint8)
    else:
        protos = proto[0]  # (nm,Hm,Wm)
        if no_crop:
            # 不裁剪，直接选“最大掩码”作为前景
            masks_logit = (coef @ protos.reshape(nm, -1)).sigmoid().reshape(-1, Hm, Wm)  # (N,Hm,Wm)
            # 上采样到输入尺寸
            masks_up = F.interpolate(masks_logit.unsqueeze(1), size=(ih, iw), mode="bilinear", align_corners=False).squeeze(1)
            # 去 pad -> 回原图
            padw, padh = pad; x1, y1 = padw, padh; x2, y2 = iw - padw, ih - padh
            masks_crop = masks_up[:, y1:y2, x1:x2]
            masks_final = F.interpolate(masks_crop.unsqueeze(1), size=(h0, w0), mode="nearest").squeeze(1)
            # 取面积最大的那张或阈值融合
            areas = (masks_final > mask_thr).float().sum((1,2))
            m = masks_final[areas.argmax().item()]
            union = (m > mask_thr).cpu().numpy().astype(np.uint8) * 255
        else:
            masks_bin = process_masks(protos, coef, boxes_xyxy, (ih, iw), bin_thres=mask_thr)  # (N,ih,iw)
            padw, padh = pad; x1, y1 = padw, padh; x2, y2 = iw - padw, ih - padh
            masks_cropped = masks_bin[:, y1:y2, x1:x2]
            masks_final = F.interpolate(masks_cropped.unsqueeze(1).float(), size=(h0, w0), mode="nearest").squeeze(1).to(torch.uint8)
            union = (masks_final > 0).any(dim=0).cpu().numpy().astype(np.uint8) * 255

    # 保存
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    cv2.imwrite(f"{save_prefix}_mask.png", union)
    overlay = im0.copy()
    fg = np.zeros_like(overlay, dtype=np.uint8)
    fg[..., 1] = union
    overlay = cv2.addWeighted(overlay, 1.0, fg, 0.45, 0.0)
    cv2.imwrite(f"{save_prefix}_overlay.png", overlay)
    print(f"完成：\n  - mask: {save_prefix}_mask.png\n  - overlay: {save_prefix}_overlay.png")

    # === 只保留掩码区域 & 裁成最小外接矩形 ===
    # 1) 用掩码把原图以外的区域抠黑
    masked = cv2.bitwise_and(im0, im0, mask=union)  # union 是 0/255 单通道
    
    # 2) 计算掩码的最小外接矩形（轴对齐）
    ys, xs = np.where(union > 0)
    if ys.size == 0 or xs.size == 0:
        # 掩码为空的兜底：直接存一张全黑图，或你也可以选择不另存
        cv2.imwrite(f"{save_prefix}_masked_roi.png", np.zeros_like(im0))
    else:
        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())
    
        # （可选）给外接矩形留一点边距，比如 6 像素
        pad = 0  # 想留边就改成正值
        xmin = max(0, xmin - pad); xmax = min(w0 - 1, xmax + pad)
        ymin = max(0, ymin - pad); ymax = min(h0 - 1, ymax + pad)
    
        # 3) 按最小外接矩形裁图
        masked_roi = masked[ymin:ymax + 1, xmin:xmax + 1]
    
        # 4) 保存
        cv2.imwrite(f"{save_prefix}_masked_roi.png", masked_roi)
        # 如果你也想把相应的二值掩码一并裁出来，顺手保存：
        mask_roi = union[ymin:ymax + 1, xmin:xmax + 1]
        cv2.imwrite(f"{save_prefix}_mask_roi.png", mask_roi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="onnx 路径")
    ap.add_argument("--source", required=True, help="输入图片路径")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max_det", type=int, default=50)
    ap.add_argument("--save_prefix", type=str, default="runs/seg_vis/tongue")
    ap.add_argument("--mask_thr", type=float, default=0.45)
    ap.add_argument("--pre_topk", type=int, default=200)
    ap.add_argument("--center_prior", action="store_true")
    ap.add_argument("--no_crop", action="store_true")
    ap.add_argument("--debug", action="store_true", default=True)
    args = ap.parse_args()

    infer_onnx_seg(
        onnx_path=args.model,
        img_path=args.source,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        save_prefix=args.save_prefix,
        mask_thr=args.mask_thr,
        pre_topk=args.pre_topk,
        center_prior=args.center_prior,
        no_crop=args.no_crop,
        debug=args.debug
    )
