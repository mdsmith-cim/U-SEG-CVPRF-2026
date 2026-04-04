#!/usr/bin/env python

import pickle as pkl
import argparse

"""
Strips backbone weights from existing model
"""

if __name__ == "__main__":

    argparse = argparse.ArgumentParser(description='Strips backbone weights from existing model')
    argparse.add_argument('input', type=str, help='Path to input model file')
    argparse.add_argument('output', type=str, help='Path to output model file')
    args = argparse.parse_args()

    print(f'Reading model from {args.input}')
    with open(args.input, 'rb') as f:
        obj = pkl.load(f)

    num_deleted = 0
    print('Deleting backbone entires...')
    for k in obj["model"].copy().keys():
        if "backbone" in k:
            del obj["model"][k]
            num_deleted += 1

    print(f'Deleted {num_deleted} backbone weight(s)')
    print(f'Writing model to {args.output}')
    with open(args.output, 'wb') as f:
        pkl.dump(obj, f)
    print('Done!')