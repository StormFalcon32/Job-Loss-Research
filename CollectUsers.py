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

# loaded = []
# with open(path / 'Users.json', encoding='utf-8') as f:
#     for line in f:
#         loaded.append(json.loads(line.rstrip('\n|\r')))

def get_users(start_ind):
    with open(path / 'Auth.json') as f:
        auth_tokens = json.load(f)
    auth = tweepy.OAuthHandler(auth_tokens[0], auth_tokens[1])
    auth.set_access_token(auth_tokens[2], auth_tokens[3])
    api = tweepy.API(auth)
    for ind in range(start_ind, len(mentions)):
        tweet_mentions = mentions[ind]
        tweet_users = []
        for user_id in tweet_mentions:
            time.sleep(1)
            try:
                user = api.get_user(user_id)
                tweet_users.append(user._json)
            except tweepy.error.TweepError as e:
                print(e)
        with open(path / 'Users.jsonl', 'a') as f:
            f.write(json.dumps(tweet_users) + '\n')
        print(ind)

if __name__ == '__main__':
    start_time = time.perf_counter()
    get_users(0)
    time_elapsed = time.perf_counter() - start_time
    print(time_elapsed)
