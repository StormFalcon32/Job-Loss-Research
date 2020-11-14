import json
from pathlib import Path

import numpy as np
import pyLDAvis
from biterm.btm import oBTM
from biterm.utility import topic_summuary, vec_to_biterms
from sklearn.feature_extraction.text import CountVectorizer

chunksize = 2000
iterations = 10

def main(start, end, increment):
    path = Path('C:/Data/Python/JobLoss')
    data_words = []
    with open(path / 'Processed.json') as f:
        data = json.load(f)
        for tweet in data:
            data_words.append(' '.join(tweet[1]))
    vec = CountVectorizer()
    X = vec.fit_transform(data_words).toarray()
    vocab = np.array(vec.get_feature_names())
    biterms = vec_to_biterms(X)
    for k in range(start, end, increment):
        print('Model %s' % k)
        btm = oBTM(num_topics=k, V=vocab)
        for i in range(0, len(biterms), chunksize):
            print('%s / %s' % (i, len(biterms)))
            biterms_chunk = biterms[i : i + chunksize]
            btm.fit(biterms_chunk, iterations=iterations)
        topics = btm.transform(biterms)
        vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
        pyLDAvis.save_html(vis, str(path / ('Visualizations/BTMVisualization%s.html' % k)))

if __name__ == '__main__':
    main(11, 13, 1)
