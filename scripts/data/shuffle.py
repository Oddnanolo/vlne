#!/usr/bin/env python3

# from https://github.com/usert5432/vlne_data_extractors_uboone

import argparse
import os

import h5py
import numpy as np

def parse_cmdargs():
    parser = argparse.ArgumentParser("Merge multiple hdf5 files")

    parser.add_argument(
        'input',
        help    = 'Input file',
        metavar = 'INPUT',
        type    = str,
    )

    parser.add_argument(
        'output',
        help    = 'Name of the output file',
        metavar = 'OUTPUT',
        type    = str,
    )

    parser.add_argument(
        '-s', '--seed',
        dest    = 'seed',
        default = 0,
        help    = 'Shuffle Seed',
        type    = int,
    )

    return parser.parse_args()

def shuffle_node(node, f_dst, indices):
    shuffled_values = node[:][indices]

    f_dst.create_dataset(
        node.name, data = shuffled_values, dtype = node.dtype,
        maxshape = (None, ), chunks = True
    )

def shuffle(dst_file, src_file, seed):
    indices = None
    rng     = np.random.default_rng(seed)

    with h5py.File(dst_file, 'w') as f_dst:
        with h5py.File(src_file, 'r') as f_src:
            for node in f_src.values():
                print("   - Shuffing node '%s'..." % node.name)

                if indices is None:
                    indices = np.arange(len(node))
                    rng.shuffle(indices)

                shuffle_node(node, f_dst, indices)

def main():
    cmdargs = parse_cmdargs()

    if os.path.exists(cmdargs.output):
        raise RuntimeError(
            "Output file '%s' exists. Will not overwrite." % (cmdargs.output)
        )

    shuffle(cmdargs.output, cmdargs.input, cmdargs.seed)

if __name__ == '__main__':
    main()
