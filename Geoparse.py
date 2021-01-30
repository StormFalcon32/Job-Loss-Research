import json
import time
from pathlib import Path

import gensim
import plotly.express as px
import spacy
from gensim.test.utils import datapath
from geopy.geocoders import Nominatim

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
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
    'Northern Mariana Islands':'MP',
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
mentions = []
ind_mapping = {}
ind_mapping_reverse = {}
with open(path / 'Processed.json') as f:
    data = json.load(f)
    ind = 0
    for tweet in data:
        data_words.append(tweet['text'])
        orig_words.append(tweet['orig_text'])
        mentions.append(tweet['mentions'])
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
        if max_prob_topic == x:
            indices.append(ind_mapping[ind])
    print(len(indices))
    with open(path / ('Locations/Topic%s.txt' % x), 'w') as f:
        result = ''
        for ind in indices:
            result += str(ind) + '\n'
        f.write(result)

def get_locations(x, shingle_range):
    queried_locations = {}
    ner = spacy.load('en_core_web_sm')
    geolocator = Nominatim(user_agent='benchen9708@gmail.com')
    with open(path / ('Locations/Topic%s.txt' % x), 'r') as f:
        indices = f.read().split('\n')
    with open(path / 'Bios.json') as f:
        bios = json.load(f)
    indices.pop()
    locations_dict = {}
    for ind in indices:
        has_locations = False
        locations = {}
        locations['twitter'] = ''
        locations['text'] = []
        locations['bios'] = []
        # check for twitter location
        ind = int(ind)
        tweet = data[ind]
        if tweet['place']:
            time.sleep(1)
            if tweet['place']['full_name'] in queried_locations:
                locations['twitter'] = queried_locations[tweet['place']['full_name']]
                has_locations = True
            else:
                try:
                    location = geolocator.geocode(tweet['place']['full_name'], timeout=1000)
                    if location.address == 'United States':
                        state = 'US'
                    else:
                        location = geolocator.reverse([location.latitude, location.longitude])
                        if location.raw['address']['country_code'] != 'us':
                            continue
                        state = location.raw['address']['state']
                    locations['twitter'] = state
                    has_locations = True
                    queried_locations[tweet['place']['full_name']] = state
                except AttributeError as e:
                    print(e)
                    print(tweet['place']['full_name'])
        # ner text
        tweet_text = orig_words[ind_mapping_reverse[ind]]
        for k_shingle in shingle_range:
            tweet_text = tweet_text.split(' ')
            for slice_ind in range(len(tweet_text)):
                slice_combine = ' '.join(tweet_text[slice_ind : slice_ind + k_shingle])
                doc = ner(slice_combine)
                for ent in doc.ents: 
                    if ent.label_ == 'GPE':
                        time.sleep(1)
                        if ent.text in queried_locations:
                            locations['text'].append(queried_locations[ent.text])
                            has_locations = True
                        else:
                            try:
                                location = geolocator.geocode(ent.text, timeout=1000)
                                if location.address == 'United States':
                                    state = 'US'
                                else:
                                    location = geolocator.reverse([location.latitude, location.longitude])
                                    if location.raw['address']['country_code'] != 'us':
                                        continue
                                    state = location.raw['address']['state']
                                locations['text'].append(state)
                                has_locations = True
                                queried_locations[ent.text] = state
                            except AttributeError as e:
                                print(e)
                                print(ent.text)
        # ner bio
        user_mentions = bios[ind_mapping_reverse[ind]]
        for bio in user_mentions:
            for k_shingle in shingle_range:
                bio = bio.split(' ')
                for slice_ind in range(len(bio)):
                    slice_combine = ' '.join(bio[slice_ind : slice_ind + k_shingle])
                    doc = ner(slice_combine)
                    for ent in doc.ents: 
                        if ent.label_ == 'GPE':
                            time.sleep(1)
                            if ent.text in queried_locations:
                                locations['bios'].append(queried_locations[ent.text])
                                has_locations = True
                            else:
                                try:
                                    location = geolocator.geocode(ent.text, timeout=1000)
                                    if location.address == 'United States':
                                        state = 'US'
                                    else:
                                        location = geolocator.reverse([location.latitude, location.longitude])
                                        if location.raw['address']['country_code'] != 'us':
                                            continue
                                        state = location.raw['address']['state']
                                    locations['bios'].append(state)
                                    has_locations = True
                                    queried_locations[ent.text] = state
                                except AttributeError as e:
                                    print(e)
                                    print(ent.text)
        if has_locations:
            locations_dict[ind] = locations
        print(ind)
    with open(path / 'Locations/Locations.json', 'w') as f:
        json.dump(locations_dict, f, indent=4)

def choropleth():
    with open(path / 'Locations/Locations.json') as f:
        locations = json.load(f)
    total = 0
    state_occurences = {}
    state_occurences_indices = []
    for ind in locations:
        tweet = locations[ind]
        has_state = False
        if tweet['twitter'] != 'US' and tweet['twitter'] != '':
            abbrev = us_state_abbrev[tweet['twitter']]
            if abbrev in state_occurences:
                state_occurences[abbrev] += 1
            else:
                state_occurences[abbrev] = 1
            total += 1
            has_state = True
        if tweet['text'].extend(tweet['bios']):
            for location in tweet['text'].extend(tweet['bios']):
                if location != 'US':
                    abbrev = us_state_abbrev[location]
                    if abbrev in state_occurences:
                        state_occurences[abbrev] += 1
                    else:
                        state_occurences[abbrev] = 1
                    total += 1
                    has_state = True
        if has_state:
            state_occurences_indices.append(ind)
    fig = px.choropleth(locations=list(state_occurences.keys()), locationmode='USA-states', color=state_occurences.values(), scope='usa')
    fig.show()
    print(total)
    states = [(orig_words[ind_mapping_reverse[int(i)]], locations[i]) for i in state_occurences_indices]
    with open(path / 'Locations/States.json', 'w') as f:
        json.dump(states, f, indent=4)


if __name__ == '__main__':
    start_time = time.perf_counter()
    # get_topic(10, 7)
    get_locations(7)
    # choropleth()
    time_elapsed = time.perf_counter() - start_time
    print(time_elapsed)
