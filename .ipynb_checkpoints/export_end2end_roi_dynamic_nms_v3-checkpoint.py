# export_end2end_roi_dynamic_nms_v3.py
# 任意(H,W) -> [内部resize到640] -> YOLO分割 -> 掩码重建 -> NMS(ONNX) -> 回原图 -> 动态裁ROI
# 输出: roi_u8 (1,3,h_roi,w_roi)，变长尺寸，ROI外全黑。仅支持 batch=1（变长输出原因）。

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision.ops import nms as tv_nms  # 会导出为 ONNX NonMaxSuppression

IMGSZ = 640
MASK_THR = 0.72
PRE_TOPK = 400
MAX_DET  = 1

def xywh2xyxy(x):
    y = x.clone()
    y[...,0] = x[...,0] - x[...,2] / 2
    y[...,1] = x[...,1] - x[...,3] / 2
    y[...,2] = x[...,0] + x[...,2] / 2
    y[...,3] = x[...,1] + x[...,3] / 2
    return y

def collect_tensors(obj, acc=None):
    if acc is None:
        acc = []
    if isinstance(obj, torch.Tensor):
        acc.append(obj)
    elif isinstance(obj, (list, tuple)):
        for it in obj:
            collect_tensors(it, acc)
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_tensors(v, acc)
    return acc

def pick_pred_proto_from_outputs(outputs, imgsz, nc_hint=1):
    """
    自动从任意嵌套结构 outputs 中识别 pred(3D) 和 proto(4D)。
    规则：
      - pred: 3D，含一个“大维”(N>=1000)，另一维为C≈ 4 + nc + nm。
      - proto: 4D，两个空间维 ~ imgsz/4（80~320范围都接受），某一维 == expected_nm (= C-4-nc)；
               若形如 (B,Hm,Wm,nm) 会自动转成 (B,nm,Hm,Wm)。
    返回: pred(B,N,C)、proto(B,nm,Hm,Wm)、(C, expected_nm)
    """
    tensors = collect_tensors(outputs)
    # 候选 pred
    pred_cands = []
    for t in tensors:
        if t.dim() == 3:
            B, A, B2 = t.shape
            # 统一成 (B, N, C) 的评分：Nmax >= 1000
            if max(A, B2) >= 1000:
                pred_cands.append(t)
    if not pred_cands:
        raise RuntimeError("未在模型输出中找到 pred 候选(3D且N很大)")

    # 选 N 最大者
    def pred_score(t):
        return max(t.shape[1], t.shape[2])
    pred_raw = sorted(pred_cands, key=pred_score, reverse=True)[0]

    # 转成 (B,N,C)
    if pred_raw.shape[1] < pred_raw.shape[2]:
        pred = pred_raw.transpose(1, 2)
    else:
        pred = pred_raw

    B, N, C = pred.shape
    expected_nm = C - 4 - nc_hint
    if expected_nm <= 0:
        # 如果 nc_hint 不正确，先用 1 兜底再试一遍
        expected_nm = C - 4 - 1
        if expected_nm <= 0:
            raise RuntimeError(f"pred通道数异常，C={C} 无法得到合理的 nm")

    # 候选 proto
    proto_cands = [t for t in tensors if t.dim() == 4]
    # 优先：某个维度恰好等于 expected_nm（同时有两个近似空间维）
    def is_spatial_pair(h, w):
        lo, hi = imgsz // 8, imgsz // 2   # 大致在 80~320 之间
        return (lo <= h <= hi) and (lo <= w <= hi)

    best = None
    for t in proto_cands:
        Bp, D1, D2, D3 = t.shape
        # (B, nm, Hm, Wm)
        if D1 == expected_nm and is_spatial_pair(D2, D3):
            best = ("NCHW", t); break
        # (B, Hm, Wm, nm)
        if D3 == expected_nm and is_spatial_pair(D1, D2):
            best = ("NHWC", t); break
    if best is None:
        # 退而求其次：挑出“看起来像 proto”的：两个空间维在范围内，且通道维(不管在哪) <= 128
        scored = []
        for t in proto_cands:
            shape = t.shape
            perms = [("NCHW", shape[1], shape[2], shape[3]), ("NHWC", shape[3], shape[1], shape[2])]
            for tag, nm_like, h_like, w_like in perms:
                if is_spatial_pair(h_like, w_like) and nm_like <= 128:
                    scored.append((tag, t, nm_like))
        if not scored:
            raise RuntimeError("未在模型输出中找到 proto 候选(4D且空间维/通道数合理)")
        # 选通道最小的
        scored.sort(key=lambda x: x[2])
        best = (scored[0][0], scored[0][1])

    layout, proto_raw = best
    if layout == "NHWC":
        # (B,Hm,Wm,nm) -> (B,nm,Hm,Wm)
        proto = proto_raw.permute(0,3,1,2).contiguous()
    else:
        proto = proto_raw

    return pred, proto, (C, expected_nm)

class End2EndTongueROI_Dynamic_NMS(nn.Module):
    """
    输入:  (1,3,H,W) RGB, float32, [0,1]
    输出:  roi_u8 (1,3,h_roi,w_roi)  —— 变长尺寸，ROI 外全黑
    """
    def __init__(self, yolo_pt_path, nc=1):
        super().__init__()
        y = YOLO(yolo_pt_path)
        self.m = y.model
        self.m.export = True
        self.im = IMGSZ
        self.mask_thr = MASK_THR
        self.pre_topk = PRE_TOPK
        self.max_det = MAX_DET
        self.nc_hint = int(nc)

    def forward(self, x_raw):
        B, C, H0, W0 = x_raw.shape  # 仅支持 batch=1
        # 1) 前处理：resize 到 640×640（不 letterbox，导出更稳）
        x640 = F.interpolate(x_raw, size=(self.im, self.im), mode="bilinear", align_corners=False)

        # 2) 模型前向（可能返回嵌套结构）
        y = self.m(x640)

        # 3) 自动识别 pred / proto
        pred, proto, (Cc, expected_nm) = pick_pred_proto_from_outputs(y, self.im, nc_hint=self.nc_hint)

        # 统一 pred 为 (1,N,C)
        if pred.dim() != 3:
            raise RuntimeError(f"pred维度异常: {pred.shape}")

        Bp, N, Cc = pred.shape
        nm = proto.shape[1]
        Hm, Wm = proto.shape[2], proto.shape[3]

        # 若自动识别的 nm 与 expected_nm 不同，以 expected_nm 为准进行截断/选前
        if nm != expected_nm and expected_nm > 0 and expected_nm <= nm:
            proto = proto[:, :expected_nm, :, :]
            nm = expected_nm

        # 现在应满足 Cc = 4 + nc + nm（nc≈nc_hint）
        nc = Cc - 4 - nm
        if nc <= 0:
            # 仍异常就报错（此时多半是权重/实现不一致）
            raise RuntimeError(f"通道数不一致: C={Cc}, nm={nm}, 推断 nc={nc} ≤ 0；请检查权重/导出头结构。")

        # 4) 解析 boxes/score/coef
        boxes_xywh = pred[..., :4]               # (1,N,4)
        cls_logits  = pred[..., 4:4+nc]         # (1,N,nc)
        coef        = pred[..., 4+nc:]          # (1,N,nm)

        if nc == 1:
            score_cls = torch.sigmoid(cls_logits).squeeze(-1)      # (1,N)
        else:
            score_cls = torch.sigmoid(cls_logits).amax(dim=-1)     # (1,N)
        maskness = coef.abs().mean(dim=-1)                          # (1,N)
        score = torch.relu(score_cls - 0.5) + 1e-3
        score = score * (maskness / (maskness.amax(dim=1, keepdim=True) + 1e-6))  # (1,N)

        # === Center Prior（越靠中心权重越大）===
        # boxes_xywh: (1, N, 4) -> 取中心 (cx, cy)
        cxcy = boxes_xywh[:, :, :2].squeeze(0)               # (N,2)
        # 若是归一化坐标，先放大到 640 域
        if boxes_xywh.max() <= 1.2:
            cxcy = cxcy * float(self.im)                      # self.im=640
        
        # 以 640×640 的中心为参照计算权重
        dx = (cxcy[:, 0] - self.im / 2).abs() / (self.im / 2)
        dy = (cxcy[:, 1] - self.im / 2).abs() / (self.im / 2)
        center_w = (1 - 0.5 * (dx + dy)).clamp(0, 1)         # [0,1]
        # 分数融合：与 UI 保持同结构
        score = score.squeeze(0) * (0.5 + 0.5 * center_w)    # (N,)
        score = score.unsqueeze(0)                            # 还原回 (1,N)
        
        # 预筛 topK
        k = int(min(self.pre_topk, N))
        topk_score, topk_idx = torch.topk(score, k=k, dim=1)       # (1,k)
        idx_expand4 = topk_idx.view(1,k,1).expand(1,k,4)
        idx_expandM = topk_idx.view(1,k,1).expand(1,k,nm)
        boxes_topk = torch.gather(boxes_xywh, 1, idx_expand4).squeeze(0)  # (k,4)
        coef_topk  = torch.gather(coef,       1, idx_expandM).squeeze(0)  # (k,nm)
        score_topk = topk_score.squeeze(0)                                 # (k,)

        # xywh -> xyxy（640域）
        boxes_xyxy = xywh2xyxy(boxes_topk)
        boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, self.im-1)
        boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, self.im-1)

        # 5) NMS（torchvision.ops.nms -> ONNX NonMaxSuppression）
        keep = tv_nms(boxes_xyxy, score_topk, iou_threshold=0.5)
        keep = keep[:max(self.max_det, 1)]
        if keep.numel() == 0:
            keep = torch.tensor([0], dtype=torch.long)

        final_box  = boxes_xyxy.index_select(0, keep)   # (m,4)
        final_coef = coef_topk.index_select(0, keep)    # (m,nm)
        # 仅取 1 个实例
        final_box  = final_box[:1]                      # (1,4)
        final_coef = final_coef[:1]                     # (1,nm)

        # 6) 掩码重建 -> 回原图尺寸
        S = Hm * Wm
        proto_flat = proto.reshape(1, nm, S)                       # (1,nm,S)
        masks_logit = torch.bmm(final_coef.unsqueeze(1), proto_flat).squeeze(1)  # (1,S)
        mask_640 = F.interpolate(masks_logit.view(1,1,Hm,Wm),
                                 size=(self.im, self.im), mode="bilinear", align_corners=False).squeeze(1)  # (1,640,640)
        mask_orig = F.interpolate(torch.sigmoid(mask_640).unsqueeze(1),
                                  size=(H0, W0), mode="bilinear", align_corners=False).squeeze(1)           # (1,H0,W0)
        bin_mask = (mask_orig > MASK_THR).to(x_raw.dtype)          # (1,H0,W0)
        
        # === 用最终检测框在原图域再裁一次，去除框外的误检 ===
        scale_x = W0 / float(self.im)
        scale_y = H0 / float(self.im)
        fb = final_box.clone()                                # (1,4) in 640-domain
        fb[:, [0, 2]] = fb[:, [0, 2]] * scale_x
        fb[:, [1, 3]] = fb[:, [1, 3]] * scale_y
        
        ys = torch.arange(H0, device=bin_mask.device).view(1, H0, 1)
        xs = torch.arange(W0, device=bin_mask.device).view(1, 1, W0)
        x1 = fb[:, 0:1].view(1, 1, 1); y1 = fb[:, 1:2].view(1, 1, 1)
        x2 = fb[:, 2:3].view(1, 1, 1); y2 = fb[:, 3:4].view(1, 1, 1)
        
        rect = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)   # (1,H0,W0)
        bin_mask = (bin_mask > 0) & rect
        bin_mask = bin_mask.to(x_raw.dtype)

        # 7) 抠图 + 动态外接矩形 + 动态裁剪（NonZero + IndexSelect）
        x255 = (x_raw * 255.0).clamp(0,255)                        # (1,3,H0,W0)
        masked_full = x255 * bin_mask.unsqueeze(1)                 # (1,3,H0,W0)

        rows_on = (bin_mask.amax(dim=2) > 0)  # (1,H0)
        cols_on = (bin_mask.amax(dim=1) > 0)  # (1,W0)

        idx_rows = torch.nonzero(rows_on[0], as_tuple=False).squeeze(1)  # (hr,)
        idx_cols = torch.nonzero(cols_on[0], as_tuple=False).squeeze(1)  # (wr,)
        if idx_rows.numel() == 0 or idx_cols.numel() == 0:
            return torch.zeros(1,3,1,1, dtype=torch.uint8)

        roi_h = masked_full.index_select(2, idx_rows)  # (1,3,hr,W0)
        roi   = roi_h.index_select(3, idx_cols)        # (1,3,hr,wr)
        return roi.to(torch.uint8)

def export(pt_path: str, out_path="end2end_dynamic_roi_nms.onnx", nc=1):
    model = End2EndTongueROI_Dynamic_NMS(pt_path, nc=nc).eval()
    dummy = torch.zeros(1,3,480,640, dtype=torch.float32)  # 仅用于建图；实际推理任意H×W
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["image_rgb_f32_0_1"],
        output_names=["roi_rgb_u8"],
        opset_version=12,
        dynamic_axes={
            "image_rgb_f32_0_1": {0: "batch", 2: "in_h", 3: "in_w"},
            "roi_rgb_u8":        {0: "batch", 2: "roi_h", 3: "roi_w"},
        }
    )
    print("✅ 导出完成:", out_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="YOLOv12-seg 的 best.pt")
    ap.add_argument("--out", default="end2end_dynamic_roi_nms.onnx")
    ap.add_argument("--nc", type=int, default=1, help="类别数（舌象通常=1）")
    args = ap.parse_args()
    export(args.pt, args.out, nc=args.nc)
