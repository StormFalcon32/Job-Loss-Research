import json
import logging
import pickle
import sys
from pathlib import Path

import gensim
from gensim.test.utils import datapath

passes = 20
iterations = 400

def main(start, end, increment):
    path = Path('C:/Data/Python/JobLoss')
    # configure logging
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data_words = []
    with open(path / 'Processed.json') as f:
        data = json.load(f)
        for tweet in data:
            data_words.append(tweet[1])
    # create dictionary, corpus, and tdm
    id2word = gensim.corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    # view corpus
    # print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
    for k in range(start, end, increment):
        # train model
        lda_model = gensim.models.ldamulticore.LdaMulticore(
            corpus=corpus, id2word=id2word, num_topics=k, eta='auto', passes=passes, iterations=iterations, eval_every=None, workers=4, random_state=1)
        # save model
        file = datapath(path / ('Models/Model%s' % k))
        lda_model.save(file)
        pickle.dump(corpus, open(path / ('Models/Corp%s.pkl' % k), 'wb'))


if __name__ == '__main__':
    main(11, 12, 1)
