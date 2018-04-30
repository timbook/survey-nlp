import sys, os
import nltk
import pandas as pd
import numpy as np

import ngram

# k_vec = [int(i) for i in sys.argv[1:]]
k_vec = range(int(sys.argv[1]))

comps_df = pd.read_csv('output/complaint-topics.csv')
comps = comps_df['Consumer complaint narrative']

ex_sents = open('output/example_sents.txt').read().split('\n')
ex_sents = [s for s in ex_sents if s != '']

pp_df = pd.DataFrame({'sentence': ex_sents})

for k in k_vec:
    print("Entering loop %d" % k)
    comps_k = list(comps[comps_df.topic11 == k])
    print("There are %d rows in complaint %d" % (len(comps_k), k))

    # Break apart complaints that are multiple sentences long.
    comps_k = [
        sent 
        for sents in comps_k
        for sent in nltk.tokenize.sent_tokenize(sents)
    ]

    # Instantiate trigram model.
    mod = ngram.NGramModel(comps_k, 3)

    # Filter complaints into a format efficiently used in trigram.
    mod.filterSents([
        ('(19|20)[0-9]{2}', '_year_'),
        ('[^A-Za-z0-9 .!?]', ''),
        ('[Xx]{2,}', '_proper_noun_'),
        ('[\.]{3,}', '___')
    ])

    # Initiate N-grams with k = 0.05
    mod.makeNGrams(0.05)

    p_scores = []
    for s in ex_sents:
        p_scores.append(mod.perplexity(s))

    pp_df['topic_%d' % k] = p_scores

pp_df.to_csv('output/perlexity-table.csv', index=False)
