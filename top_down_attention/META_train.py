import argparse
from TRAIN import train_attention_layer
from TRAIN import train_last_layer


parser = argparse.ArgumentParser()
parser.add_argument('--option', dest='run')
args = parser.parse_args()


if __name__ == '__main__':
    if args.run == 'attn_layer':
        train_attention_layer.execute()
    elif args.run == 'last_layer':
        train_last_layer.execute()

