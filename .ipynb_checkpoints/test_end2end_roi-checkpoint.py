import cv2, numpy as np, onnxruntime as ort

onnx_path = "/root/yolov12-main/runs/segment/train18/weights/end2end_dynamic_roi_nms.onnx"
img_path  = "/root/autodl-tmp/split_job/o/images/test/012155-001.jpg"

# 1) 加载 ORT（可选先尝试 CUDA，再回退 CPU）
sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider","CPUExecutionProvider"])

# 2) 读入任意尺寸BGR -> 转 RGB float32 [0,1]，满足模型输入约定
bgr  = cv2.imread(img_path)
rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
inp  = np.transpose(rgb, (2,0,1))[None, ...]  # (1,3,H,W)

# 3) 前向：直接得到“变长尺寸”的 ROI 图像（1,3,h_roi,w_roi），uint8
(roi_u8,) = sess.run(None, {"image_rgb_f32_0_1": inp})
print("ROI tensor shape:", roi_u8.shape)  # e.g. (1, 3, 812, 554)

# 4) 保存：RGB->BGR
roi_rgb = np.transpose(roi_u8[0], (1,2,0))    # (h_roi,w_roi,3)
roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("roi.png", roi_bgr)
