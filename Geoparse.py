import csv
import json
import statistics
import time
from pathlib import Path

import gensim
import plotly.graph_objects as go
import spacy
from gensim.test.utils import datapath
from geopy.geocoders import Nominatim

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

path = Path('C:/Data/Python/JobLoss')
data_words = []
orig_words = []
ind_mapping = {}
ind_mapping_reverse = {}
ind_mapping_processed = {}

queried_locations = {}
ner = spacy.load('en_core_web_sm')
geolocator = Nominatim(user_agent='benchen9708@gmail.com')

with open(path / 'Processed.json') as f:
    data = json.load(f)
    ind = 0
    for tweet in data:
        ind_mapping_processed[tweet['ind']] = ind
        ind += 1
with open(path / 'ProcessedSimilarRemoved.json') as f:
    data = json.load(f)
    ind = 0
    for tweet in data:
        data_words.append(tweet['text'])
        orig_words.append(tweet['orig_text'])
        ind_mapping[ind] = tweet['ind']
        ind_mapping_reverse[tweet['ind']] = ind
        ind += 1

data = []
with open(path / 'Tweets/AprilMay.jsonl', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.rstrip('\n|\r')))

def get_topic(k, x):
    # load model
    file = datapath(path / ('Models/Model%s' % k))
    lda_model = gensim.models.ldamodel.LdaModel.load(file)
    corpus = [lda_model.id2word.doc2bow(text) for text in data_words]
    indices = []
    print(len(corpus))
    for ind in range(len(corpus)):
        topic_vector = lda_model.get_document_topics(corpus[ind], minimum_probability=0, per_word_topics=True)
        max_prob = 0
        max_prob_topic = 0
        for topic_id, prob in topic_vector[0]:
            if prob > max_prob:
                max_prob = prob
                max_prob_topic = topic_id
        if max_prob_topic in x:
            indices.append(ind_mapping[ind])
    print(len(indices))
    with open(path / ('Locations/Topic%s.txt' % x), 'w') as f:
        result = ''
        for ind in indices:
            result += str(ind) + '\n'
        f.write(result)

def twitter_location(tweet):
    output = ''
    if tweet['place']['full_name'] in queried_locations:
        output = queried_locations[tweet['place']['full_name']]
    else:
        try:
            time.sleep(1)
            location = geolocator.geocode(tweet['place']['full_name'], timeout=1000)
            if location.address != 'United States' and tweet['place']['full_name'] != 'U.S.':
                location = geolocator.reverse([location.latitude, location.longitude])
                if 'country_code' not in location.raw['address'] or location.raw['address']['country_code'] != 'us':
                    return output
                state = location.raw['address']['state']
                output = state
                queried_locations[tweet['place']['full_name']] = state
        except AttributeError as e:
            print(e)
            print(tweet['place']['full_name'])
        return output

def text_location(tweet_text, shingle_range):
    output = []
    for k_shingle in range(shingle_range, 0, -1):
        split_text = tweet_text.split(' ')
        had_ents = False
        for slice_ind in range(len(split_text)):
            slice_combine = ' '.join(split_text[slice_ind : slice_ind + k_shingle])
            doc = ner(slice_combine)
            for ent in doc.ents:
                had_ents = True
                if ent.label_ == 'GPE':
                    if ent.text in queried_locations:
                        output.append(queried_locations[ent.text])
                    else:
                        try:
                            time.sleep(1)
                            location = geolocator.geocode(ent.text, timeout=1000)
                            if location.address != 'United States' and ent.text != 'U.S.':
                                location = geolocator.reverse([location.latitude, location.longitude])
                                if 'country_code' not in location.raw['address'] or location.raw['address']['country_code'] != 'us':
                                    continue
                                state = location.raw['address']['state']
                                output.append(state)
                                queried_locations[ent.text] = state
                        except AttributeError as e:
                            print(e)
                            print(ent.text)
        if not had_ents:
            break
    return output

def user_location(user):
    output = ''
    if user['location']:
        if user['location'] in queried_locations:
            output = queried_locations[user['location']]
        else:
            try:
                time.sleep(1)
                location = geolocator.geocode(user['location'], timeout=1000)
                if location.address != 'United States' and user['location'] != 'U.S.':
                    location = geolocator.reverse([location.latitude, location.longitude])
                    if 'country_code' not in location.raw['address'] or location.raw['address']['country_code'] != 'us':
                        return output
                    state = location.raw['address']['state']
                    output = state
                    queried_locations[user['location']] = state
            except AttributeError as e:
                print(e)
                print(user['location'])
    return output

def bio_location(user, shingle_range):
    output = []
    bio = user['description']
    if bio:
        output = text_location(bio, shingle_range)
    return output

def get_locations(x, shingle_range):
    with open(path / ('Locations/Topic%s.txt' % x), 'r') as f:
        indices = f.read().split('\n')
    users = []
    with open(path / 'Users.jsonl') as f:
        for line in f:
            users.append(json.loads(line.rstrip('\n|\r')))
    indices.pop()
    locations_dict = {}
    for ind in indices:
        locations = {}
        locations['twitter'] = ''
        locations['user'] = ''
        locations['bio'] = []
        locations['text'] = []
        locations['tagged_users'] = []
        locations['tagged_bios'] = []
        # check for twitter location
        ind = int(ind)
        tweet = data[ind]
        if tweet['place']:
            locations['twitter'] = twitter_location(tweet)
        # ner primary user
        locations['user'] = user_location(tweet['user'])
        locations['bio'] = bio_location(tweet['user'], shingle_range)     
        # ner text
        tweet_text = orig_words[ind_mapping_reverse[ind]]  
        locations['text'] = text_location(tweet_text, shingle_range)
        # ner tagged users
        user_mentions = users[ind_mapping_processed[ind]]
        for user in user_mentions:
            tagged_user_location = user_location(user)
            if tagged_user_location:
                locations['tagged_users'].append(tagged_user_location)
            locations['tagged_bios'].extend(bio_location(user, shingle_range))
        if not locations['twitter'] and not locations['user'] and not locations['bio'] and not locations['text'] and not locations['tagged_users'] and not locations['tagged_bios']:
            continue
        locations_dict[ind] = locations
        print(ind)
    print(len(locations_dict))
    with open(path / 'Locations/Locations.json', 'w') as f:
        json.dump(locations_dict, f, indent=4)

def choropleth(map_type):
    with open(path / 'Locations/Locations.json') as f:
        locations = json.load(f)
    total = 0
    state_occurences = {}
    for state in us_state_abbrev:
        state_occurences[us_state_abbrev[state]] = 0
    print(len(locations))
    for ind in locations:
        tweet = locations[ind]
        has_state = False
        tweet_states = set()
        if tweet['twitter'] :
            abbrev = us_state_abbrev[tweet['twitter']]
            tweet_states.add(abbrev)
            has_state = True
        elif tweet['user']:
            abbrev = us_state_abbrev[tweet['user']]
            tweet_states.add(abbrev)
            has_state = True
        elif tweet['bio']:
            for location in tweet['bio']:
                abbrev = us_state_abbrev[location]
                tweet_states.add(abbrev)
                has_state = True
        elif tweet['text']:
            for location in tweet['text']:
                abbrev = us_state_abbrev[location]
                tweet_states.add(abbrev)
                has_state = True
        elif tweet['tagged_users']:
            for location in tweet['tagged_users']:
                abbrev = us_state_abbrev[location]
                tweet_states.add(abbrev)
                has_state = True
        elif tweet['tagged_bios']:
            for location in tweet['tagged_bios']:
                abbrev = us_state_abbrev[location]
                tweet_states.add(abbrev)
                has_state = True
        if has_state:
            for state in list(tweet_states):
                state_occurences[state] += 1
            total += len(tweet_states)
    if map_type == 'combined':
        not_paid_dict = {}
    with open(path / 'Locations/UIData.csv') as f:
        csv_reader = csv.reader(f)
        line = 0
        for row in csv_reader:
            if line == 0:
                line += 1
                continue
            if row[0] == 'US':
                continue
            state = us_state_abbrev[row[0]]
            initial_claims = int(row[1]) + int(row[2])
            if map_type == 'unpaid':
                first_payments = int(row[3]) + int(row[4])
                not_paid = initial_claims - first_payments
                state_occurences[state] = not_paid / initial_claims
            if map_type == 'complaints':
                state_occurences[state] = state_occurences[state] / initial_claims
            if map_type == 'combined':
                first_payments = int(row[3]) + int(row[4])
                not_paid = initial_claims - first_payments
                not_paid_dict[state] = not_paid / initial_claims
                state_occurences[state] = state_occurences[state] / initial_claims
            if map_type == 'divided':
                first_payments = int(row[3]) + int(row[4])
                not_paid = initial_claims - first_payments
                state_occurences[state] = state_occurences[state] / not_paid
                if not_paid < 0:
                    del state_occurences[state]
    if map_type == 'combined':
        not_paid_standard_dev = statistics.stdev(not_paid_dict.values())
        not_paid_mean = statistics.mean(not_paid_dict.values())
        not_paid_standard = {}
        for state in not_paid_dict:
            not_paid_standard[state] = (not_paid_dict[state] - not_paid_mean) / not_paid_standard_dev
        complaints_standard_dev = statistics.stdev(state_occurences.values())
        complaints_mean = statistics.mean(state_occurences.values())
        complaints_standard = {}
        for state in state_occurences:
            complaints_standard[state] = (state_occurences[state] - complaints_mean) / complaints_standard_dev
        for state in us_state_abbrev:
            state = us_state_abbrev[state]
            not_paid = not_paid_standard[state]
            complaints = complaints_standard[state]
            state_occurences[state] = complaints - not_paid
    if map_type == 'raw':
        fig = go.Figure(data=go.Choropleth(locations=list(state_occurences.keys()), z=list(state_occurences.values()), colorscale='RdPu', locationmode='USA-states'))
        fig.update_layout(title_text='Complaints', geo_scope='usa')
    if map_type == 'unpaid':
        fig = go.Figure(data=go.Choropleth(locations=list(state_occurences.keys()), z=list(state_occurences.values()), colorscale='RdPu', locationmode='USA-states'))
        fig.update_layout(title_text='Unpaid / Initial Claims', geo_scope='usa')
    if map_type == 'complaints':
        fig = go.Figure(data=go.Choropleth(locations=list(state_occurences.keys()), z=list(state_occurences.values()), zmin=0.00002, zmax=0.0004, colorscale='RdPu', locationmode='USA-states'))
        fig.update_layout(title_text='Complaints / Initial Claims', geo_scope='usa')
    if map_type == 'combined':
        fig = go.Figure(data=go.Choropleth(locations=list(state_occurences.keys()), z=list(state_occurences.values()), zmin=-3.5, zmax=3.5, colorscale='RdBu', locationmode='USA-states'))
        fig.update_layout(title_text='Complaints - Unpaid', geo_scope='usa')
    if map_type == 'divided':
        fig = go.Figure(data=go.Choropleth(locations=list(state_occurences.keys()), z=list(state_occurences.values()), zmin=0, zmax=0.001, colorscale='RdPu', locationmode='USA-states'))
        fig.update_layout(title_text='Complaints / Unpaid', geo_scope='usa')
    fig.show()
    print(total)
    states = [(orig_words[ind_mapping_reverse[int(i)]], locations[i]) for i in locations]
    with open(path / 'Locations/States.json', 'w') as f:
        json.dump(states, f, indent=4)


if __name__ == '__main__':
    start_time = time.perf_counter()
    # get_topic(15, [0, 12])
    # get_locations([0, 12], 4)
    choropleth('raw')
    time_elapsed = time.perf_counter() - start_time
    print(time_elapsed)
