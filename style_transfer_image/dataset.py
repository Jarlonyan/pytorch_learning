#coding=utf-8

import numpy as np
import collections
import random


def main():
    #idx_pairs = nlp_get_vocabulary()
    #print idx_pairs
    pd = ProcessData()
    pd.init_sample_table()
    res = pd.get_batch_pairs(3, 3)
    pos, neg = pd.get_pairs_by_neg_sample(res, 5)
    print "pos=",pos
    print "neg=",neg

if __name__ == '__main__':
    main()
