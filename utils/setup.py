import argparse
import os
import json

import wandb

from parse_config import parse


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=False)

    return parser


def load_config(file):
    return json.load(file)


def setup():
    args = make_parser().parse_known_args()
    config = load_config(args.config)
    
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(login=os.getenv('WANDB_LOGIN'), name=args.run_name)
    
    torch.manual_seed(config['manual_seed'])

    return parse(config)
