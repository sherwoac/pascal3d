#!/usr/bin/env python

import argparse

import pascal3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camframe', action='store_true')
    args = parser.parse_args()

    camframe = args.camframe

    dataset = pascal3d.dataset.Pascal3DDataset('val', use_split_files=True)
    for i in range(len(dataset)):
        dataset.show_cad(i, camframe=camframe)


if __name__ == '__main__':
    main()
