import json
import time
from pathlib import Path
import tweepy

path = Path('C:/Data/Python/JobLoss')
mentions = []
with open(path / 'Processed.json') as f:
    data = json.load(f)
    for tweet in data:
        mentions.append(tweet['mentions'])

def get_bios():
    with open(path / 'Auth.json') as f:
        auth_tokens = json.load(f)
    auth = tweepy.OAuthHandler(auth_tokens[0], auth_tokens[1])
    auth.set_access_token(auth_tokens[2], auth_tokens[3])
    api = tweepy.API(auth)
    bios = []
    for ind in range(len(mentions)):
        tweet_mentions = mentions[ind]
        tweet_bios = []
        for user_id in tweet_mentions:
            time.sleep(1)
            try:
                user = api.get_user(user_id)
                bio = user.description
                tweet_bios.append(bio)
            except tweepy.error.TweepError as e:
                print(e)
        bios.append(tweet_bios)
        print(ind)
    return bios

if __name__ == '__main__':
    start_time = time.perf_counter()
    bios = get_bios()
    with open(path / 'Bios.json', 'w') as f:
        json.dumps(bios, f)
    time_elapsed = time.perf_counter() - start_time
    print(time_elapsed)
