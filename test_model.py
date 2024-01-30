import argparse
import torch
from utils import load_model


def get_args_parser():
    parser = argparse.ArgumentParser('Test a model', add_help=False)
    parser.add_argument('--model_name', default='vit_s_none', type=str, help='Model identifier')
    parser.add_argument('--device', default='cuda', help='device to use for testing')

    return parser


def main(args):
    model = load_model(args.model_name)
    print('Model:', model)

    device = torch.device(args.device)
    model.to(device)  # move model to device


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    main(args)