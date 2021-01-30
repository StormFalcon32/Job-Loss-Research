import json
from pathlib import Path

import gensim
from gensim.test.utils import datapath


def topic_distribution(k):
    path = Path('C:/Data/Python/JobLossBackup1-21')
    data_words = []
    orig_data = []
    with open(path / 'Processed.json') as f:
        data = json.load(f)
        for tweet in data:
            data_words.append(tweet['text'])
            orig_data.append(tweet['orig_text'])
    # load model
    file = datapath(path / ('Models/Model%s' % k))
    lda_model = gensim.models.ldamodel.LdaModel.load(file)
    corpus = [lda_model.id2word.doc2bow(text) for text in data_words]
    topic_dist = [0 for _ in range(k)]
    print(len(corpus))
    for ind in range(len(corpus)):
        # print(ind)
        topic_vector = lda_model.get_document_topics(corpus[ind], minimum_probability=0, per_word_topics=True)
        max_prob = 0
        max_prob_topic = 0
        for topic_id, prob in topic_vector[0]:
            if prob > max_prob:
                max_prob = prob
                max_prob_topic = topic_id
        topic_dist[max_prob_topic] += 1
    topic_id = 1
    for num_tweets in topic_dist:
        print('Topic %s: %s tweets' % (topic_id, num_tweets))
        topic_id += 1



if __name__ == '__main__':
    topic_distribution(10)
