import re, nltk

class NGram:
    def __init__(self, sents, n_gram):
        self.sents = sents
        self.n_gram = n_gram

    def filterSents(self, replacements):

        start_tokens = ['START' + str(i) for i in range(self.ngram - 1)]
        corpus_out = []

        for sent in self.sents:
            sent = sent.lower()
            for (reg, repl) in replacements:
                sent = re.sub(reg, repl, sent)
            sent = nltk.tokenize.sent_tokenize(sent)
            for s in sent:
                s = nltk.tokenize.word_tokenize(s)
                if s[-1] in ['.', '?', '!']: s = s[:-1]
                corpus_out += start_tokens + s + ['END']

        self.corpus = corpus_out
