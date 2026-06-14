import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from prettytable import PrettyTable
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info

"""
Clean paper-data validation entry.

This script is the source for paper tables:
- Precision (%)
- Recall (%)
- mAP50 (%)
- mAP50:95 (%)
- Params (M)
- GFLOPs
- FPS

It writes:
runs/val/<name>/paper_data.txt
"""

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp/weights/best.pt')
    parser.add_argument('--data', type=str, default='dataset/data.yaml')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/val')
    parser.add_argument('--plots', action='store_true', default=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    model = RTDETR(args.weights)
    result = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=args.plots,
        exist_ok=True,
    )

    if model.task == 'detect':
        length = result.box.p.size
        model_names = list(result.names.values())

        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        n_l, n_p, n_g, flops = model_info(model.model)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = [
            "GFLOPs",
            "Parameters",
            "前处理时间/一张图",
            "推理时间/一张图",
            "后处理时间/一张图",
            "FPS(前处理+模型推理+后处理)",
            "FPS(推理)",
            "Model File Size"
        ]
        model_info_table.add_row([
            f'{flops:.1f}',
            f'{n_p:,}',
            f'{preprocess_time_per_image / 1000:.6f}s',
            f'{inference_time_per_image / 1000:.6f}s',
            f'{postprocess_time_per_image / 1000:.6f}s',
            f'{1000 / all_time_per_image:.2f}',
            f'{1000 / inference_time_per_image:.2f}',
            f'{get_weight_size(args.weights)}MB'
        ])

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = [
            "Class Name",
            "Precision",
            "Recall",
            "F1-Score",
            "mAP50",
            "mAP75",
            "mAP50-95"
        ]

        for idx in range(length):
            model_metrice_table.add_row([
                model_names[idx],
                f"{result.box.p[idx]:.4f}",
                f"{result.box.r[idx]:.4f}",
                f"{result.box.f1[idx]:.4f}",
                f"{result.box.ap50[idx]:.4f}",
                f"{result.box.all_ap[idx, 5]:.4f}",
                f"{result.box.ap[idx]:.4f}",
            ])

        precision = result.results_dict['metrics/precision(B)']
        recall = result.results_dict['metrics/recall(B)']
        map50 = result.results_dict['metrics/mAP50(B)']
        map5095 = result.results_dict['metrics/mAP50-95(B)']
        f1 = np.mean(result.box.f1[:length])
        map75 = np.mean(result.box.all_ap[:length, 5])

        model_metrice_table.add_row([
            "all(平均数据)",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{map50:.4f}",
            f"{map75:.4f}",
            f"{map5095:.4f}",
        ])

        paper_summary = PrettyTable()
        paper_summary.title = "Paper Summary"
        paper_summary.field_names = [
            "Model",
            "Precision (%)",
            "Recall (%)",
            "mAP50 (%)",
            "mAP50:95 (%)",
            "Params (M)",
            "GFLOPs",
            "FPS"
        ]
        paper_summary.add_row([
            args.name,
            f"{precision * 100:.2f}",
            f"{recall * 100:.2f}",
            f"{map50 * 100:.2f}",
            f"{map5095 * 100:.2f}",
            f"{n_p / 1_000_000:.2f}",
            f"{flops:.1f}",
            f"{1000 / all_time_per_image:.2f}",
        ])

        print('-' * 20 + ' 论文上的数据以以下结果为准 ' + '-' * 20)
        print(model_info_table)
        print(model_metrice_table)
        print(paper_summary)

        save_path = result.save_dir / 'paper_data.txt'
        with open(save_path, 'w+', errors="ignore", encoding="utf-8") as f:
            f.write(str(model_info_table))
            f.write('\n\n')
            f.write(str(model_metrice_table))
            f.write('\n\n')
            f.write(str(paper_summary))
            f.write('\n')

        print('-' * 20, f'结果已保存至 {save_path}', '-' * 20)
