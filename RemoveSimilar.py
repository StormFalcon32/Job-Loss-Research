import json
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from nltk import ngrams


def main():
    path = Path('C:/Data/Python/JobLoss')
    orig_data = []
    with open(path / 'Processed.json') as f:
        data = json.load(f)
        for tweet in data:
            orig_data.append(tweet[2])
#     orig_data = ['minhash is a probabilistic data structure for estimating the similarity between datasets',
#   'finhash dis fa frobabilistic fata ftructure for festimating the fimilarity fetween fatasets',
#   'minhash is a data structure for estimating the similarity between datasets',
#   'weights controls the relative importance between minizing false positive',
#   'wfights cfntrols the rflative ifportance befween minizing fflse posftive',
#   'unique'
# ]
    markers = [0 for _ in range(len(orig_data))]
    lsh = MinHashLSH(threshold=0.4, num_perm=128)
    minhashes = {}
    for c, i in enumerate(orig_data):
        print(c)
        minhash = MinHash(num_perm=128)
        for d in ngrams(i, 3):
            minhash.update(''.join(d).encode('utf-8'))
        lsh.insert(c, minhash)
        minhashes[c] = minhash
    for i in range(len(minhashes.keys())):
        result = lsh.query(minhashes[i])
        markers[i] = 1
        for j in result:
            if markers[j] != 1:
                markers[j] = 2
    final = [data[ind] for ind, val in enumerate(markers) if val != 2]
    with open(path / 'ProcessedSimilarRemoved.json', 'w') as f:
            json.dump(final, f)

if __name__ == '__main__':
    main()
