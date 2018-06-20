from io import open
import os
import argparse


parser = argparse.ArgumentParser(description='Split file into train, valid and test')

parser.add_argument(
    '--filename',
    type=str,
    default="",
    help='filename')

parser.add_argument(
    '--train',
    type=float,
    default=0.8,
    help='tain set proportion')

parser.add_argument(
    '--valid',
    type=float,
    default=0.1,
    help='valid set proportion')

parser.add_argument(
    '--test',
    type=float,
    default=0.1,
    help='test set proportion')


args = parser.parse_args()



f = open(args.filename, encoding='utf-8').read()
length = len(f)
tri = int(args.train * length)
vi = int((args.train + args.valid) * length)
tei = int((args.train + args.valid + args.test) * length)




with open(args.filename + ".train.txt", 'w') as file:
    file.write(f[0:tri])

with open(args.filename + ".valid.txt", 'w') as file:
    file.write(f[tri:vi])

with open(args.filename + ".test.txt", 'w') as file:
    file.write(f[vi:tei])
