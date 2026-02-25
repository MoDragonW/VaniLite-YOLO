from ultralytics import YOLO
import json
import csv

if __name__ == '__main__':
    # 1. 加载模型（替换为你的实际模型路径）
    model = YOLO(r'D:\研究项目\基于轻量化设计的YOLO人参饮片检测\VaniLite-YOLO\trained_models\VaniLite-YOLO.pt')

    # 2. 执行验证并获取metrics对象
    metrics = model.val(
        split='test',
        data='renshen.yaml',
        verbose=True
    )

    # 3. 提取整体指标（核心：兼容所有版本的通用方式）
    # 从results_dict直接获取整体标量（和控制台all行完全一致，无索引错误）
    results_dict = metrics.results_dict
    precision = results_dict['metrics/precision(B)']  # 整体精确率（标量）
    recall = results_dict['metrics/recall(B)']  # 整体召回率（标量）
    map50 = results_dict['metrics/mAP50(B)']  # 整体mAP50（标量）
    map50_95 = results_dict['metrics/mAP50-95(B)']  # 整体mAP50-95（标量）
    f1_score = 2 * precision * recall / (precision + recall)  # 整体F1分数

    # 整理整体指标字典
    overall_metrics = {
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50-95": map50_95,
        "f1_score": f1_score
    }

    # 4. 提取类别级指标（通用方式，兼容所有版本）
    class_metrics = {}
    class_names = metrics.names  # 类别名称字典（如{0: 'White Ginseng', ...}）

    # 方式：从metrics.results中提取每个类别的详细指标
    # metrics.results是按类别排序的指标列表，顺序和class_names一致
    if hasattr(metrics, 'results') and len(metrics.results) > 0:
        for cls_idx, cls_name in class_names.items():
            # 确保索引不越界
            if cls_idx < len(metrics.results):
                cls_result = metrics.results[cls_idx]
                # 提取该类别的精确率、召回率、mAP50、mAP50-95
                class_metrics[cls_name] = {
                    "precision": cls_result.box.p,
                    "recall": cls_result.box.r,
                    "mAP50": cls_result.box.map50,
                    "mAP50-95": cls_result.box.map,
                    "f1_score": 2 * cls_result.box.p * cls_result.box.r / (cls_result.box.p + cls_result.box.r)
                }

    # 合并整体指标和类别指标
    all_metrics = {
        "整体指标": overall_metrics,
        "各分类详细指标": class_metrics
    }

    # 5. 保存指标到文件
    # 方式1：JSON文件（推荐，保留高精度数值）
    with open("results/val_metrics_precise.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)

    # 方式2：CSV文件（仅保存整体指标，避免类别指标的复杂结构）
    with open("results/val_metrics_precise.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(overall_metrics.keys())
        writer.writerow(overall_metrics.values())

    # 6. 打印验证（查看精确数值）
    print("\n✅ 整体精确指标（原始数值，非约数）：")
    for key, value in overall_metrics.items():
        print(f"{key}: {value}")

    # 可选：打印部分类别的详细指标
    if class_metrics:
        print("\n✅ 部分类别的详细精确指标：")
        # 选择你关注的类别
        target_classes = ["White Ginseng", "Korean ginseng", "red ginseng root"]
        for cls in target_classes:
            if cls in class_metrics:
                print(f"\n{cls}:")
                for k, v in class_metrics[cls].items():
                    print(f"  {k}: {v}")