#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 runs/val_a640/*/metrics_summary.json 到一个 CSV。"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='runs/val_a640')
    p.add_argument('--out', type=str, default='runs/val_a640/summary.csv')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []
    for jf in sorted(root.glob('*/metrics_summary.json')):
        with open(jf, 'r', encoding='utf-8') as f:
            d = json.load(f)
        rows.append({
            '实验名': d.get('name', jf.parent.name),
            '模式': d.get('mode', ''),
            'tile': d.get('tile', ''),
            'target_overlap': d.get('target_overlap', ''),
            'Precision(%)': round(float(d.get('precision', 0)) * 100, 2),
            'Recall(%)': round(float(d.get('recall', 0)) * 100, 2),
            'F1(%)': round(float(d.get('f1', 0)) * 100, 2),
            'mAP50(%)': round(float(d.get('map50', 0)) * 100, 2),
            'mAP50:95(%)': round(float(d.get('map50_95', 0)) * 100, 2),
            'FPS': round(float(d.get('fps', 0)), 2),
            'Params(M)': round(float(d.get('params_m', 0)), 2),
            'GFLOPs': round(float(d.get('gflops', 0)), 1),
            '平均局部切片数/图': round(float(d.get('avg_local_tiles_per_image', 0)), 2),
            '平均总视图数/图': round(float(d.get('avg_views_per_image', 0)), 2),
        })
    if not rows:
        print(f'没有找到 {root}/实验名/metrics_summary.json')
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'汇总完成：{out}')
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
