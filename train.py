import argparse
import yaml

from trainer import get_trainer


def arg_parser():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--config',
                        type=str,
                        default='config/train_config.yml',
                        help='Configuration file to use.')

    return parser.parse_args()


if __name__ == '__main__':
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    arg = arg_parser()
    config = arg.config

    with open(config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    train_name = config['train']['trainer']
    trainer = get_trainer(train_name)(config)

    trainer.train()
