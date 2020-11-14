import json
import logging
import sys
from pathlib import Path

import gensim
import matplotlib.pyplot as plt
from gensim.test.utils import datapath


def main(start, end, increment):
    path = Path('C:/Data/Python/JobLoss')
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    coherence_values_cv = []
    data_words = []
    with open(path / 'Processed.json') as f:
        data = json.load(f)
        for tweet in data:
            data_words.append(tweet[1])
    for k in range(start, end, increment):
        # load model
        file = datapath(path / ('Models/Model%s' % k))
        lda_model = gensim.models.ldamodel.LdaModel.load(file)
        # coherence score
        coherence_model_cv = gensim.models.coherencemodel.CoherenceModel(
            model=lda_model, texts=data_words, coherence='c_v')
        coherence_cv = coherence_model_cv.get_coherence()
        coherence_values_cv.append(coherence_cv)
    x = range(start, end, increment)
    plt.plot(x, coherence_values_cv)
    plt.xlabel('Num Topics')
    plt.ylabel('Coherence Score (c_v)')
    plt.legend(('coherence_values'), loc='best')
    plt.grid()
    plt.xticks(x)
    plt.savefig(path / 'CoherenceScores/CoherenceCV.png')


if __name__ == '__main__':
    main(5, 25, 1)
