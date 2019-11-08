#!/usr/bin/env python2

import random
import sys

import basic_dataset
import conllu_dataset

ufeats_column = 5

def main():
    random.seed(42)
    dataset = conllu_dataset.ConlluDataset()
    dataset.load_file(sys.stdin)
    dropout = basic_dataset.SentenceDropout(random,
            [ufeats_column,],
            [1.0,],
    )
    dataset = basic_dataset.Sample(
        dataset, random,
        sentence_modifier = dropout,
        with_replacement = False,
        keep_order = True,
    )
    dataset.save_to_file(sys.stdout)

if __name__ == "__main__":
    main()

