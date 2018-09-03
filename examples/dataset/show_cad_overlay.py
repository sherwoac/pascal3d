#!/usr/bin/env python

import pascal3d


def main():
    data_type = 'val'
    dataset = pascal3d.dataset.Pascal3DDataset(data_type, use_split_files=True)
    for i in range(len(dataset)):
        print('[{dtype}:{id}] showing cad overlay'
              .format(dtype=data_type, id=i))
        dataset.show_cad_overlay(i)


if __name__ == '__main__':
    main()
