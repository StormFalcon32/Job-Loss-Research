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
    with open(path / 'ProcessedSimilarRemoved.json') as f:
        data = json.load(f)
        for tweet in data:
            tweet_ids.append(tweet['id'])
            data_words.append(tweet['text'])
            orig_data.append(tweet['orig_text'])
    corpus = pickle.load(open(path / 'Models/Corp.pkl', 'rb'))
    # id2word = gensim.corpora.Dictionary(data_words)
    for k in range(start, end, increment):
        print('Model %s' % k)
        # load model
        file = datapath(path / ('Models/Model%s' % k))
        lda_model = gensim.models.ldamodel.LdaModel.load(file)
        with open(path / 'Topics' / ('Model%sTopics.txt' % k), 'w') as f:
            f.write(str(lda_model.print_topics(k)))
        top_docs = [{} for _ in range(k)]
        # find top documents for each topic
        for ind in range(len(corpus)):
            # if tweet_ids[ind] == 1259245913798668288:
            #     print()
            #     for word_id in corpus[ind]:
            #         print(word_id, id2word[word_id[0]])
            # if 'animal' in data_words[ind]:
            #     print()
            #     for word_id in corpus[ind]:
            #         print(word_id, id2word[word_id[0]])
            topic_vector = lda_model.get_document_topics(corpus[ind], per_word_topics=True)
            for topic_id, prob in topic_vector[0]:
                top_docs[topic_id][ind] = prob
        for ind in range(len(top_docs)):
            topic = top_docs[ind]
            sorted_topic = sorted(topic.items(), key=lambda kv : kv[1], reverse=True)
            top_docs[ind] = sorted_topic
        f = open(path / 'Topics' / ('Model%sTopTweets.txt' % k), 'w', encoding='utf-8')
        topic_ind = 1
        write_lines = []
        for topic in top_docs:
            write_lines.append('Topic %s' % topic_ind)
            topic_ind += 1
            num = 0
            for doc in topic:
                if num >= top_x_docs:
                    break
                write_lines.append(str(tweet_ids[doc[0]]))
                write_lines.append(orig_data[doc[0]])
                write_lines.append('-------')
                num += 1
            write_lines.append('---------------------------')
        f.writelines([line + '\n' for line in write_lines])
        f.close()



if __name__ == '__main__':
    main(10, 11, 1, 10)
