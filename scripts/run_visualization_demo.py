import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_toy_segmentation_data import create_toy_segmentation_data
from src.visualization import save_visualization_set


DISCLAIMER_ZH = "本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。"
DISCLAIMER_EN = (
    "This project is intended only for medical image segmentation experiments and engineering workflow "
    "validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world "
    "medical decision-making."
)


def _ensure_data(data_dir):
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists() or not list(images_dir.glob("*.png")):
        create_toy_segmentation_data(data_dir)
    return images_dir, masks_dir


def _mock_prediction(mask):
    kernel = np.ones((5, 5), dtype=np.uint8)
    shifted = np.roll(mask, shift=4, axis=1)
    pred = cv2.dilate(shifted, kernel, iterations=1)
    return (pred > 127).astype(np.uint8) * 255


def _write_report(output_dir, paths):
    report_path = Path(output_dir) / "visualization_report.md"
    output_rows = "\n".join([f"| {name} | `{path}` |" for name, path in paths.items()])
    report_path.write_text(
        f"""# Visualization Overview / 可视化概述

中文：
本报告展示 toy segmentation demo 的可视化输出，包括预测叠加图、误检图、漏检图和并排对比图。

English:
This report shows visualization outputs from the toy segmentation demo, including prediction overlay, false-positive map, false-negative map, and side-by-side comparison.

## Output Files / 输出文件

| File Type / 文件类型 | Path / 路径 |
| --- | --- |
{output_rows}

## Interpretation Notes / 解释说明

中文：
红色叠加区域表示 mock prediction，误检图显示预测多出的区域，漏检图显示真实 mask 中未被预测覆盖的区域。

English:
The red overlay indicates the mock prediction. The false-positive map shows extra predicted regions, and the false-negative map shows ground-truth regions not covered by prediction.

## Current Limitations / 当前限制

中文：
本 demo 使用 toy mask 和 mock prediction，仅用于验证可视化流程，不代表真实医学数据上的模型表现。

English:
This demo uses toy masks and mock predictions only to validate the visualization workflow. It does not represent model performance on real medical data.

## Medical Disclaimer / 医学免责声明

中文：
{DISCLAIMER_ZH}

English:
{DISCLAIMER_EN}
""",
        encoding="utf-8",
    )
    return report_path


def run_visualization_demo(data_dir="examples/toy_segmentation_demo", output_dir="outputs/visualizations"):
    images_dir, masks_dir = _ensure_data(data_dir)
    image_path = sorted(images_dir.glob("*.png"))[0]
    mask_path = masks_dir / image_path.name
    image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    pred = _mock_prediction(mask)
    paths = save_visualization_set(image, mask, pred, output_dir, prefix="sample_001")
    report_path = _write_report(output_dir, paths)
    return Path(output_dir), report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run toy segmentation visualization demo.")
    parser.add_argument("--data-dir", default="examples/toy_segmentation_demo")
    parser.add_argument("--output-dir", default="outputs/visualizations")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir, report_path = run_visualization_demo(args.data_dir, args.output_dir)
    print(f"Visualization outputs saved to: {output_dir}")
    print(f"Visualization report saved to: {report_path}")


if __name__ == "__main__":
    main()
