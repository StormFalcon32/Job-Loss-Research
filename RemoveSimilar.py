import json
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from nltk import ngrams


def main():
    path = Path('C:/Data/Python/JobLoss')
    orig_data = []
    with open(path / 'Processed.json') as f:
        data = json.load(f)
        for tweet in data:
            orig_data.append(tweet[2])
#     orig_data = ['@PAAuditorGen today marks 60 days I’ve been unemployed.  Im still unable to file weekly certs in the PUA system, there are errors in my employment history I can’t correct, they have the wrong Claim Effective Date, the email is useless, and theres still nobody else to contact.',
#   '@JulieSuCA @CA_EDD Still trying to figure out what’s going on?! None of my unemployment is retroactive. I’ve been unemployed since March (have proof) and the Edd started my claim on April 26th. Haven’t gotten a hold of anyone to help. Please help',
#   'I got laid off yesterday #FuckCorona https://t.co/J57jaxL1Db',
#   'I got laid off today https://t.co/szCidUSGvS',
#   'I got laid off LMFAO https://t.co/bIE9HZ6lHn',
#   'just kidding i got laid off https://t.co/cvstd6oJAz',
#   'Before I got laid off, I never pictured a career in *Shia LeBeouf.*. #cah',
#   'Before I got laid off, I never pictured a career in *Having a vagina.*. #CardsAgainstHumanity',
#   'I spoke with 3 people whose landlords sexually harassed them when they struggled to pay rent during the pandemic. “He knows I don’t have a job. He knows I dont have anywhere to go — he’s preying on me,” one told mehttps://t.co/evyWHcphn9 https://t.co/4nU59t8iwW',
#   'Her Landlord Asked To Spend The Night With Her After She Lost Her Job And Couldn’t Afford Rent “He knows I don’t have a job. He knows I dont have anywhere to go — he’s preying on me.” https://t.co/PnH71wr8ZE',
#   'Her Landlord Asked To Spend The Night With Her After She Lost Her Job And Couldn’t Afford Rent “He knows I don’t have a job. He knows I dont have... - https://t.co/xBwQPyng5i #body #abs https://t.co/PVEOLvyLEx',
#   'Her Landlord Asked To Spend The Night With Her After She Lost Her Job And Couldn’t Afford Rent https://t.co/97sXbjuxiW via @ambiej “He knows I don’t have a job. He knows I don’t have anywhere to go.—He’s preying on me.” ��🏼🏼‍♀️'
#     ]
#     data = orig_data
    markers = [0 for _ in range(len(orig_data))]
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    minhashes = {}
    for c, i in enumerate(orig_data):
        print(c)
        minhash = MinHash(num_perm=128)
        for d in ngrams(i, 3):
            minhash.update(''.join(d).encode('utf-8'))
        lsh.insert(c, minhash)
        minhashes[c] = minhash
    for i in range(len(minhashes.keys())):
        result = lsh.query(minhashes[i])
        if markers[i] == 2:
            continue
        markers[i] = 1
        for j in result:
            if markers[j] != 1:
                markers[j] = 2
    final = [data[ind] for ind, val in enumerate(markers) if val != 2]
    # for tweet in final:
    #     print(tweet)
    #     print()
    with open(path / 'ProcessedSimilarRemoved.json', 'w') as f:
            json.dump(final, f)

if __name__ == '__main__':
    main()
