import json
import multiprocessing as mp
from pathlib import Path

import Levenshtein

def check_ratio(similar_removed, try_tweet):
    for tweet in similar_removed:
        if Levenshtein.ratio(try_tweet, tweet[1]) > 0.9:
            return 1
    return 0

def main():
    path = Path('C:/Data/Python/JobLoss')
    orig_data = []
    with open(path / 'Processed.json') as f:
        data = json.load(f)
        for tweet in data:
            orig_data.append(tweet[2])
    similar_removed = []
    similar_removed.append((0, orig_data[0]))
    for i in range(1, len(orig_data)):
        print('%s / %s' % (i, len(orig_data)))
        pool = mp.Pool(mp.cpu_count())
        too_similar = 0
        try_tweet = orig_data[i]
        too_similar = pool.apply(check_ratio, args=(similar_removed, try_tweet))
        if not too_similar:
            similar_removed.append((i, try_tweet))
        pool.close()
    final = [data[ind[0]] for ind in similar_removed]
    with open(path / 'ProcessedSimilarRemoved.json', 'w') as f:
            json.dump(final, f)

if __name__ == '__main__':
    main()