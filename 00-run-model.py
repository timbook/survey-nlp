import sys, re, string
import nltk
import numpy as np
import pandas as pd
from gensim import corpora, models

if len(sys.argv) == 1:
    n_top = 10
else:
    n_top = [int(i) for i in sys.argv[1:]]

compl = pd.read_csv('../data/Consumer_Complaints.csv', encoding='utf-8')
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

for n in n_top:
    # LDA Topic Model
    print("Beginning LDA...")
    ldamodel = models.ldamodel.LdaModel(doc_list,
                                        num_topics=n,
                                        id2word=dictionary,
                                        passes=5)

    print("Saving model...")
    ldamodel.save('models/lda%02d' % n)

    # Get optimal topic predictions
    def getMax(top):
        cats = [a for (a, b) in top]
        probs = [b for (a, b) in top]
        M = np.argmax(probs)
        return(cats[M])

    print("Optimizing topic...")
    topics = ldamodel.get_document_topics(doc_list)
    opt_topics = list(map(getMax, topics))

    compl['topic%02d' % n] = opt_topics

print("Writing...")
compl.to_csv("output/complaint-topics.csv", index=False, encoding='utf-8')



