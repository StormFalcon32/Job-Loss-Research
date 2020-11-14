import json
import re
import string
from pathlib import Path

import emoji
from nltk import TweetTokenizer
from nltk.corpus import stopwords


def remove_non_ascii(text): return ''.join(i for i in text if ord(i) < 128)

def remove_unicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)       
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    return text

def replace_at_user(text):
    """ Replaces "@user" with "atUser" """
    text = re.sub(r'@[^\s]+', 'atUser', text)
    return text

""" Replaces contractions from a string to their equivalents """
contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', r'\g<1> will'), (r'(\w+)n\'t', r'\g<1> not'), (r'(\w+)\'ve', r'\g<1> have'), (r'(\w+)\'s', r'\g<1> is'), (r'(\w+)\'re', r'\g<1> are'), (r'(\w+)\'d', r'\g<1> would'), (r'&', r'and'), (r'dammit', r'damn it'), (r'dont', r'do not'), (r'wont', r'will not') ]
def replace_contractions(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, _) = re.subn(pattern, repl, text)
    return text

def main():
    path = Path('C:/Data/Python/JobLoss/Tweets')
    stop_words = set(stopwords.words('english')) 
    data = []
    with open(path / 'AprilMay.jsonl', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    tokenizer = TweetTokenizer(preserve_case=False)
    processed = []
    for tweet in data:
        tweet_id = tweet['id']
        text = tweet['text']
        username = tweet['user']['id']
        timestamp = tweet['created_at']
        tweet_type = 'tweet'
        if tweet['is_quote_status']:
            tweet_type = 'quote'
        elif 'retweeted_status' in tweet:
            tweet_type = 'retweet'
        num_retweets = tweet['retweet_count']
        num_likes = tweet['favorite_count']
        hashtags = ''
        mentions = ''
        if 'extended_tweet' in tweet:
            text = tweet['extended_tweet']['full_text']
            hashtags = ' '.join([tweet['extended_tweet']['entities']['hashtags'][i]['text'] for i in range(len(tweet['extended_tweet']['entities']['hashtags']))])
            mentions = ' '.join([tweet['extended_tweet']['entities']['user_mentions'][i]['screen_name'] for i in range(len(tweet['extended_tweet']['entities']['user_mentions']))])
        if 'retweeted_status' in tweet:
            if 'extended_tweet' in tweet['retweeted_status']:
                text = tweet['retweeted_status']['extended_tweet']['full_text']
                hashtags = ' '.join([tweet['retweeted_status']['extended_tweet']['entities']['hashtags'][i]['text'] for i in range(len(tweet['retweeted_status']['extended_tweet']['entities']['hashtags']))])
                mentions = ' '.join([tweet['retweeted_status']['extended_tweet']['entities']['user_mentions'][i]['screen_name'] for i in range(len(tweet['retweeted_status']['extended_tweet']['entities']['user_mentions']))])
            else:
                text = tweet['retweeted_status']['text']
                hashtags = ' '.join([tweet['retweeted_status']['entities']['hashtags'][i]['text'] for i in range(len(tweet['retweeted_status']['entities']['hashtags']))])
                mentions = ' '.join([tweet['retweeted_status']['entities']['user_mentions'][i]['screen_name'] for i in range(len(tweet['retweeted_status']['entities']['user_mentions']))])
        orig_text = text
        text = remove_non_ascii(text)
        text = remove_unicode(text)
        text = replace_at_user(text)
        text = replace_contractions(text)
        text = emoji.demojize(text, delimiters=('', ''))
        # turn all whitespace into single space
        text = ' '.join(text.split())
        text = tokenizer.tokenize(text)
        text = [word if 't.co' not in word else 'URL' for word in text]
        text = [word for word in text if word not in string.punctuation]
        text = [word for word in text if word not in stop_words]
        row = [tweet_id, text, orig_text, username, timestamp, tweet_type, num_retweets, num_likes, hashtags, mentions]
        processed.append(row)
    with open(path.parent / 'Processed.json', 'w') as f:
        json.dump(processed, f)


if __name__ == '__main__':
    main()
