import argparse
import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp/weights/best.pt')
    parser.add_argument('--source', type=str, default='dataset/images/test')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--project', type=str, default='runs/detect')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--device', type=str, default='0')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = RTDETR(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        project=args.project,
        name=args.name,
        device=args.device,
        save=True,
       # show_labels=False,#只有框
       # show_conf=False,#只有框
       # line_width=2,#只有框
    )
