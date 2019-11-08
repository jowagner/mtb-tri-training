#!/usr/bin/env python2

import random
import sys

import basic_dataset
import conllu_dataset

def get_splits(dataset, n, attempts = 100):
    retval = []
    if n == 1:
        retval.append(dataset)
    else:
        ds_size = dataset.get_number_of_items()
        print('request to split %d sentences with %d tokens into %d folds' %(
            len(dataset), ds_size, n
        ))
        n_1 = int(n/2)
        n_2 = n - n_1
        target_size = ds_size * n_1 / float(n)
        n_sentences = int(0.5 + len(dataset) * n_1 / n)
        best_candidate = None
        for i in range(attempts):
            sample = basic_dataset.Sample(
                dataset, random, n_sentences,
                with_replacement = False,
            )
            size = sample.get_number_of_items()
            deviation = abs(size - target_size)
            candidate = (deviation, random.random(), sample)
            if best_candidate is None or best_candidate > candidate:
                best_candidate = candidate
                print('new best candidate in attempt %d with deviation of %.3f tokens' %(
                    i+1, deviation
                ))
        half_1 = best_candidate[-1]
        half_2 = half_1.clone()
        half_2.set_remaining(random)
        print('splitting into %d sentences with %d tokens and %d sentences with %d tokens' %(
            len(half_1), half_1.get_number_of_items(),
            len(half_2), half_2.get_number_of_items(),
        ))
        splits_1 = get_splits(half_1, n_1, attempts)
        splits_2 = get_splits(half_2, n_2, attempts)
        for split in splits_1:
            retval.append(split)
        for split in splits_2:
            retval.append(split)
    return retval

def main():
    random.seed(42)
    n = int(sys.argv[1])
    dataset = conllu_dataset.ConlluDataset()
    dataset.load_file(sys.stdin)
    splits = get_splits(dataset, n)
    for i, split in enumerate(splits):
        f = open('fold-%03d.conllu' %i, 'wb')
        split.save_to_file(f)
        f.close()

if __name__ == "__main__":
    main()

