import sys
import time
import numpy as np
import scipy.cluster
from annoy import AnnoyIndex
from gensim.models.keyedvectors import KeyedVectors

kcluster = scipy.cluster.vq.kmeans

def debug_time(msg, init, now):
    print("{} [{}ms]".format(msg, int(round((now-init)*1000*1000))/1000.0), file=sys.stderr)

# Configuration Parameters
DATA_PATH = 'C:/_YaoYiheng/Projects/_CogPsychLab/NLP_AssocNetwork/data/'
VSM_BIN = "GoogleNews-vectors-negative300.bin"
VSM_LIMIT = 100000

# Load gensim model
load_t = time.time()
model = KeyedVectors.load_word2vec_format(f'{DATA_PATH}{VSM_BIN}', binary=True, limit=VSM_LIMIT)
dims = 300
lowercase_dic = {w.lower(): w for w in model.vocab.keys()}
debug_time(f"Loaded gensim model", load_t, time.time())

def load_para(text, k=10):
    text_li = text.split()
    para_vecs = []
    mean_vec = np.asarray([0 for i in range(dims)])
    features = []
    for word in text_li:
        try:
            para_vecs.append(model[word])
        except:
            try:
                para_vecs.append(model[word.lower()])
            except:
                #print("Word {} not found".format(word))
                continue
        mean_vec = np.add(mean_vec, para_vecs[-1])
        features.append(para_vecs[-1])
    features = np.asarray(features)
    max_v = -1
    mean_vec = np.true_divide(mean_vec, len(para_vecs))
    for i, o_v in enumerate(para_vecs):
        # print(o_v, mean_vec)
        # print(np.dot(o_v, mean_vec))
        cur = abs(np.dot(o_v, mean_vec))
        if (cur > max_v):
            max_v = cur
            closest_vec = text_li[i]

    for tk in range(1, 20):
        centroids, error = kcluster(features, tk)
        for c in centroids:
            cur_max = -1
            cur_word = None
            for i, p in enumerate(para_vecs):
                cur = abs(np.dot(p, c))
                if (cur > cur_max):
                    cur_max = cur
                    cur_word = text_li[i]
            print(cur_word)
        print(f"K: {tk} error: {error} score: {tk * tk * error}")

    return para_vecs, mean_vec, closest_vec

with open ('NASA_DataSets_Scrub.tsv', 'r') as fin:
    limit = 0
    fin.readline()
    for line in fin:
        data_line = line.split('\t')
        print(load_para(data_line[-2])[2])
        limit += 1
        if (limit > 0):
            break