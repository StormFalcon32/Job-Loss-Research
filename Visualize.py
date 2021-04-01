import json
import logging
import pickle
import sys
from pathlib import Path

import gensim
import pyLDAvis
import pyLDAvis.gensim
from gensim.test.utils import datapath


def main(start, end, increment):
    path = Path('C:/Data/Python/JobLoss')
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data_words = []
    with open(path / 'ProcessedSimilarRemoved.json') as f:
        data = json.load(f)
        for tweet in data:
            data_words.append(tweet['text'])
    corpus = pickle.load(open(path / 'Models/Corp.pkl', 'rb'))
    for k in range(start, end, increment):
        # load model
        file = datapath(path / ('Models/Model%s' % k))
        lda_model = gensim.models.ldamodel.LdaModel.load(file)
        # visualize
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, lda_model.id2word, sort_topics=False)
        # r'C:\Data\Python\JobLoss\Models\Visualizations\Visualization%s.html' % k
        pyLDAvis.save_html(vis, str(path / ('Visualizations/Visualization%s.html' % k)))


if __name__ == '__main__':
    main(11, 12, 1)
