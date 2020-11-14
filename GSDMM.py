import json
import pickle
from pathlib import Path

import gensim
from numpy.core.fromnumeric import argmax
from gsdmm import MovieGroupProcess

path = Path('C:/Data/Python/JobLoss')

data_words = []
orig_data = []
with open(path / 'Processed.json') as f:
    data = json.load(f)
    for tweet in data:
        data_words.append(tweet[1])
        orig_data.append(tweet[2])

def train(topics):
    mgp = MovieGroupProcess(K=topics, alpha=0.1, beta=0.1, n_iters=30)
    vocab = set(word for doc in data_words for word in doc)
    n_terms = len(vocab)
    y = mgp.fit(data_words, n_terms)
    pickle.dump(mgp, open(path / ('Models/GSDMMModel.pkl'), 'wb'))
    pickle.dump(y, open(path / ('Models/GSDMMLabel.pkl'), 'wb'))
    print('Finished training')

def score():
    mgp = pickle.load(open(path / ('Models/GSDMMModel.pkl'), 'rb'))
    id2word = gensim.corpora.Dictionary(data_words)
    topics = [[word for word in topic] for topic in mgp.cluster_word_distribution]
    print('Starting scoring')
    coherence_model_cv = gensim.models.coherencemodel.CoherenceModel(topics=topics, texts=data_words, dictionary=id2word, coherence='c_v')
    coherence_cv = coherence_model_cv.get_coherence()
    print(coherence_cv)

def sort_key(e):
    return e[1]

def top_words(x):
    mgp = pickle.load(open(path / ('Models/GSDMMModel.pkl'), 'rb'))
    topics = []
    for topic in mgp.cluster_word_distribution:
        sorted_topic = sorted(topic.items(), key=lambda kv : kv[1], reverse=True)
        topics.append(sorted_topic)
    for topic in topics:
        print('-------')
        for i in range(x):
            print(topic[i])

def top_docs(x):
    num_topics = 11
    mgp = pickle.load(open(path / ('Models/GSDMMModel.pkl'), 'rb'))
    top_docs = [{} for _ in range(num_topics)]
    doc_set = set()
    ind = -1
    for doc in data_words:
        ind += 1
        doc_joined = ' '.join(doc)
        if doc_joined in doc_set:
            continue
        doc_set.add(doc_joined)
        probabilities = mgp.score(doc)
        top_docs[argmax(probabilities)][ind] = max(probabilities)
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
            if num >= x:
                break
            print(orig_data[doc[0]])
            print('-------')
            num += 1
        print('--------------------------------')

if __name__ == '__main__':
    # train(11)
    # score()
    top_docs(10)
