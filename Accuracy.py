import json
import random
import time
from pathlib import Path

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

random.seed(100)

path = Path('C:/Data/Python/JobLoss')
ind_mapping_reverse = {}

with open(path / 'ProcessedSimilarRemoved.json') as f:
    data_words = json.load(f)
    ind = 0
    for tweet in data_words:
        ind_mapping_reverse[tweet['ind']] = ind
        ind += 1

def random_subset(x):
    with open(path / ('Locations/Topic[0, 12].txt'), 'r') as f:
        indices = f.read().split('\n')
    indices.pop()
    indices = random.sample(indices, x)
    tweets = []
    for ind in indices:
        ind = ind_mapping_reverse[int(ind)]
        tweet = data_words[ind]
        tweets.append([tweet['id'], ''])
    with open(path / 'Locations/Sample.txt', 'w') as f:
        json.dump(tweets, f, indent=4)

def single_tweet(tweet_id):
    data = []
    with open(path / 'Tweets/AprilMay.jsonl', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    # for getting tweet data from deleted tweets
    for tweet in data:
        if tweet['id'] == tweet_id:
            print(tweet)

def check_accuracy(x):
    correct = 0
    with open(path / 'Locations/Sample.txt') as f:
        sample = json.load(f)
    with open(path / 'Locations/Locations.json') as f:
        locations = json.load(f)
    with open(path / ('Locations/Topic[0, 12].txt'), 'r') as f:
        indices = f.read().split('\n')
        indices.pop()
        indices = random.sample(indices, x)
        sample_ind = 0
        for sample_ind in range(x):
            ind = indices[sample_ind]
            if ind in locations:
                tweet = locations[ind]
                tweet_states = set()
                if tweet['twitter'] :
                    abbrev = us_state_abbrev[tweet['twitter']]
                    tweet_states.add(abbrev)
                elif tweet['user']:
                    abbrev = us_state_abbrev[tweet['user']]
                    tweet_states.add(abbrev)
                elif tweet['bio']:
                    for location in tweet['bio']:
                        abbrev = us_state_abbrev[location]
                        tweet_states.add(abbrev)
                elif tweet['text']:
                    for location in tweet['text']:
                        abbrev = us_state_abbrev[location]
                        tweet_states.add(abbrev)
                elif tweet['tagged_users']:
                    for location in tweet['tagged_users']:
                        abbrev = us_state_abbrev[location]
                        tweet_states.add(abbrev)
                elif tweet['tagged_bios']:
                    for location in tweet['tagged_bios']:
                        abbrev = us_state_abbrev[location]
                        tweet_states.add(abbrev)
                if len(tweet_states) > 1:
                    print('Too much')
                else:
                    if list(tweet_states)[0] == sample[sample_ind][1]:
                        correct += 1
                    else:
                        print('Wrong')
            else:
                print('Not included')
            sample_ind += 1
    print(correct)


if __name__ == '__main__':
    # random_subset(100)
    # single_tweet(1265288097216036864)
    check_accuracy(100)

