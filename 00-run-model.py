import sys
import numpy as np
import pandas as pd
import nltk
import re
from gensim import corpora, models
import string

n_top = int(sys.argv[1]) if len(sys.argv) > 1 else 10

compl = pd.read_csv('complaint-topics.csv', encoding='utf-8')
null_mask = pd.notnull(compl['Consumer complaint narrative'])
compl = compl[null_mask]
corpus = compl['Consumer complaint narrative'].tolist()

# Normalize text
sw = set(nltk.corpus.stopwords.words('english'))
sw = [re.sub('\'', '', s) for s in sw]
porter = nltk.PorterStemmer()
def filterComps(s):
    s = s.replace('credit card', 'credit_card')
    s = s.replace('student loan', 'student_loan')
    s = s.replace('do n\'t', 'dont')
    s = re.sub('[' + string.punctuation + ']', '', s).lower()
    s = re.sub('[x]+', '', s)
    s = nltk.tokenize.word_tokenize(s)
    s = [porter.stem(w) for w in s if (w not in sw and w.isalpha())]
    return(s)

print("Beginning normalizing...")
corp_filt = list(map(filterComps, corpus))

# Get text into usable format for LDA
print("Beginning doc2bow...")
dictionary = corpora.Dictionary(corp_filt)
doc_list = [dictionary.doc2bow(c) for c in corp_filt]

# LDA Topic Model
print("Beginning LDA...")
ldamodel = models.ldamodel.LdaModel(doc_list, num_topics=n_top, id2word=dictionary, passes=5)
ldamodel.show_topics()

# Get optimal topic predictions
def getMax(top):
    cats = [a for (a, b) in top]
    probs = [b for (a, b) in top]
    M = np.argmax(probs)
    return(cats[M])

print("Optimizing topic...")
topics = ldamodel.get_document_topics(doc_list)
opt_topics = list(map(getMax, topics))
# TODO: Output all topic probabilities.

print("Writing...")
compl['topic%02d' % n_top] = opt_topics
compl.to_csv("complaint-topics.csv", index=False, encoding='utf-8')
