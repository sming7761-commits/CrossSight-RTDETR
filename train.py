import os
# Keep fast CUDA path while avoiding cuDNN v8 frontend engine errors on some 4090/CUDA builds.
os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')

import argparse
import warnings

try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

"""
RT-DETR training entry for A+B+C experiments.

A: IS-A640-GLF inference module, evaluated by val_a640_native_oldstyle.py.
B: HF-GMF network module, selected by rtdetr-r18-hfgmf.yaml.
C: IS-WIoU-Lite loss, enabled by --iswiou.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    parser.add_argument('--data', type=str, default='dataset/data.yaml')
    parser.add_argument('--name', type=str, default='baseline_200_clean')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--pretrain', type=str, default='', help='正式主实验建议留空；仅调试时使用')
    parser.add_argument('--cache', action='store_true', help='显存/内存够时可开启数据缓存')
    parser.add_argument('--amp', dest='amp', action='store_true', default=False, help='启用 AMP 混合精度训练')
    parser.add_argument('--no-amp', dest='amp', action='store_false', help='关闭 AMP')

    # C innovation: IS-WIoU-Lite
    parser.add_argument('--iswiou', action='store_true', help='启用C创新点：IS-WIoU-Lite输入空间感知Wise-IoU损失')
    parser.add_argument('--iswiou-mix', type=float, default=0.70, help='IS-WIoU-Lite中WIoU与原始GIoU的混合比例')
    parser.add_argument('--iswiou-tau', type=float, default=0.10, help='IS-WIoU-Lite小目标尺度门控阈值')
    parser.add_argument('--iswiou-temp', type=float, default=0.04, help='IS-WIoU-Lite小目标尺度门控温度')
    parser.add_argument('--iswiou-ratio', type=float, default=1.0, help='Wise-IoU内部ratio参数，默认1.0')
    return parser.parse_args()


def set_iswiou_env(args):
    if args.iswiou:
        os.environ['RTDETR_USE_ISWIOU'] = '1'
        os.environ['RTDETR_ISWIOU_MIX'] = str(args.iswiou_mix)
        os.environ['RTDETR_ISWIOU_TAU'] = str(args.iswiou_tau)
        os.environ['RTDETR_ISWIOU_TEMP'] = str(args.iswiou_temp)
        os.environ['RTDETR_ISWIOU_RATIO'] = str(args.iswiou_ratio)
        print(f'[INFO] 启用 C 创新点 IS-WIoU-Lite，mix={args.iswiou_mix}, tau={args.iswiou_tau}, temp={args.iswiou_temp}, ratio={args.iswiou_ratio}')
    else:
        for k in ['RTDETR_USE_ISWIOU', 'RTDETR_ISWIOU_MIX', 'RTDETR_ISWIOU_TAU', 'RTDETR_ISWIOU_TEMP', 'RTDETR_ISWIOU_RATIO']:
            os.environ.pop(k, None)


if __name__ == '__main__':
    args = parse_args()
    set_iswiou_env(args)

    model = RTDETR(args.model)

    if args.pretrain:
        print(f'[WARN] 正在加载预训练/历史权重：{args.pretrain}')
        model.load(args.pretrain)

    train_kwargs = dict(
        data=args.data,
        cache=args.cache,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=0,
        project=args.project,
        name=args.name,
        amp=args.amp,
    )

    if args.resume:
        train_kwargs['resume'] = args.resume

    model.train(**train_kwargs)
