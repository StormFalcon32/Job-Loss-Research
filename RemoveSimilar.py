import json
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from nltk import ngrams

def main():
    path = Path('C:/Data/Python/JobLoss')
    orig_data = []
    ind_map = []
    ind = 0
    with open(path / 'Processed.json') as f:
        data = json.load(f)
        for tweet in data:
            if tweet['type'] != 'retweet':
                orig_data.append(tweet['orig_text'])
                ind_map.append(ind)
            ind += 1
            # orig_data.append(tweet['orig_text'])
    markers = [0 for _ in range(len(orig_data))]
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    minhashes = {}
    for c, i in enumerate(orig_data):
        # print(c)
        minhash = MinHash(num_perm=128)
        for d in ngrams(i, 5):
            minhash.update(''.join(d).encode('utf-8'))
        lsh.insert(c, minhash)
        minhashes[c] = minhash
    for i in range(len(minhashes.keys())):
        result = lsh.query(minhashes[i])
        if markers[i] == 2:
            continue
        markers[i] = 1
        for j in result:
            if markers[j] != 1:
                markers[j] = 2
    doc_set = set()
    similar_removed = [data[ind_map[ind]] for ind, val in enumerate(markers) if val != 2]
    final = []
    identicals = 0
    for line in similar_removed:
        doc = ' '.join(line['text'])
        if doc in doc_set:
            identicals += 1
            continue
        doc_set.add(doc)
        final.append(line)
    print(identicals)
    print(len(final))
    with open(path / 'ProcessedSimilarRemoved.json', 'w') as f:
            json.dump(final, f)

if __name__ == '__main__':
    main()
