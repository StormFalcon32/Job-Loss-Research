import json
import pickle
from pathlib import Path

import gensim
from gensim.test.utils import datapath


def main(start, end, increment, top_x_docs):
    path = Path('C:/Data/Python/JobLoss')
    tweet_ids = []
    data_words = []
    orig_data = []
    with open(path / 'Processed.json') as f:
        data = json.load(f)
        for tweet in data:
            tweet_ids.append(tweet[0])
            data_words.append(tweet[1])
            orig_data.append(tweet[2])
    for k in range(start, end, increment):
        print('Topic %s' % k)
        # load model
        file = datapath(path / ('Models/Model%s' % k))
        lda_model = gensim.models.ldamodel.LdaModel.load(file)
        corpus = pickle.load(
            open(path / ('Models/Corp%s.pkl' % k), 'rb'))
        top_docs = [{} for _ in range(k)]
        doc_set = set()
        # find top documents for each topic
        for ind in range(len(corpus)):
            topic_vector = lda_model[corpus[ind]]
            doc = ' '.join(data_words[ind])
            if doc in doc_set:
                continue
            doc_set.add(doc)
            for topic_id, prob in topic_vector:
                top_docs[topic_id][ind] = prob
        for ind in range(len(top_docs)):
            topic = top_docs[ind]
            sorted_topic = sorted(topic.items(), key=lambda kv : kv[1], reverse=True)
            top_docs[ind] = sorted_topic
        topic_ind = 1
        for topic in top_docs:
            print('Topic %s' % topic_ind)
            topic_ind += 1
            num = 0
            for doc in topic:
                if num >= top_x_docs:
                    break
                print(tweet_ids[doc[0]])
                print(orig_data[doc[0]])
                print('-------')
                num += 1
            print('---------------------------')


if __name__ == '__main__':
    main(11, 12, 1, 10)
