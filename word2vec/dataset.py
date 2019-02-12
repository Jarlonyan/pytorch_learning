#coding=utf-8

import numpy as np
import collections
import random

def nlp_generate_corpus():
    corpus = [
        'he is a king',
        'she is a queen',
        'he is a man',
        'she is a woman',
        'warsaw is poland capital',
        'berlin is germany capital',
        'paris is france capital',
    ]
    tokens = [x.split() for x in corpus]
    return tokens

def nlp_get_vocabulary():
    corpus = nlp_generate_corpus()
    vocabulary = []
    for sentence in corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
        #end-for
    #end-for
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    window_size = 2
    idx_pairs = []
    for sentence in corpus:
        indices = [word2idx[word] for word in sentence]
        for cenw_pos in range(len(indices)):
            for w in range(-window_size, window_size+1):
                ctx_pos = cenw_pos + w
                if ctx_pos < 0 or ctx_pos >= len(indices) or cenw_pos == ctx_pos:
                    continue
                ctx_idx = indices[ctx_pos]
                idx_pairs.append((indices[cenw_pos], ctx_idx))
            #end-for
        #end-for
    #end-for
    idx_pairs = np.array(idx_pairs)
    return idx_pairs, len(vocabulary)

class ProcessData(object):
    def __init__(self):
        corpus = [
           'he is a king',
           'she is a queen',
           'he is a man',
           'she is a woman',
           'warsaw is poland capital',
           'berlin is germany capital',
           'paris is france capital']
        words = []
        for line in corpus:
            words.extend([item.strip() for item in line.split()])
        word_freq = collections.Counter(words).most_common(10 - 1)
        self.word2id = dict()
        self.id2word = dict()
        self.word_freq = dict()
        self.word_pair_cache = collections.deque()
        idx = 0
        for w,cnt in word_freq:
            self.word2id[w] = idx
            self.id2word[idx] = w
            self.word_freq[idx] = cnt
            idx += 1

        self.corpus = []
        for line in corpus:
            sentence = []
            for w in line.split():
                w = w.strip()
                if w in self.word2id:
                    idx = self.word2id[w]
                    sentence.append(idx)
            self.corpus.append(sentence)
        #end-for
 

    def init_sample_table(self):
        self.sample_table = []
        pow_freq = np.array(list(self.word_freq.values()))**0.75
        words_pow = sum(pow_freq) 
        ratio = pow_freq / words_pow
        cnt_arr = np.round(ratio * 1000)
        for idx,cnt in enumerate(cnt_arr):
            self.sample_table += [idx]*int(cnt)
        self.sample_table = np.array(self.sample_table)

    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_cache) < batch_size:
            for _ in range(10000):
                idx = random.randint(0, len(self.corpus)-1) 
                word_ids = self.corpus[idx]
                for i,w in enumerate(word_ids):
                    for j,c in enumerate(word_ids[max(i-window_size,0): i+window_size]):
                        assert w < len(self.word2id)
                        assert c < len(self.word2id)
                        if i == j:
                            continue
                        self.word_pair_cache.append((w,c))
                #end-for
            #end-for 
        #end-while
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_cache.popleft()) 
        return batch_pairs
    
    def get_pairs_by_neg_sample(self, pos_word_pair, count):
        neg_word_pair = []
        for pair in pos_word_pair:
            i = 0
            neg_v = np.random.choice(self.sample_table, size=count)
            neg_word_pair += zip([pair[0]] * count, neg_v)
        return pos_word_pair, neg_word_pair



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