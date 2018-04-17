import re, nltk
import numpy as np

class NGramModel:
    def __init__(self, sents, n_gram):
        self.sents = sents
        self.n_gram = n_gram
        self.start_tokens = ['START' + str(i) for i in range(n_gram - 1)]
        self.grams = None
        self.corpus = None
        self.pdf = None

    def __repr__(self):
        s1 = "I am an %d-gram model." % self.n_gram

        if self.corpus is not None:
            s2 = "My corpus has been generated."
        else:
            s2 = "My corpus has not yet been generated."

        if self.pdf is not None:
            s3 = "My %d-grams are ready." % self.n_gram
        else:
            s3 = "My %d-grams are not yet ready." % self.n_gram

        return(s1 + '\n' + s2 + '\n' + s3 + '\n')

    def filterSents(self, replacements, print_out = True):
        corpus_out = []

        def filterOneSent(s, stokens):
            s = nltk.tokenize.word_tokenize(s)
            if len(s) > 0 and s[-1] in ['.', '?', '!']: s = s[:-1]
            return stokens + s + ['END']

        for sent in self.sents:
            sent = sent.lower()

            for (reg, repl) in replacements:
                sent = re.sub(reg, repl, sent)

            corpus_out += filterOneSent(sent, self.start_tokens)

        self.corpus = corpus_out
        if print_out:
            print("Corpus ready!")

    def makeNGrams(self, k = 0, print_out = True):
        self.grams = nltk.ngrams(self.corpus, self.n_gram)
        gfreq = nltk.FreqDist(self.grams)
        self.pdf = nltk.LidstoneProbDist(gfreq, k)
        print("N-grams ready!")

    def getNextWord(self, sent):

        def gramMatch(ng, sent, N):
            sent_trim = sent[-(N - 1):]
            ok_gram = [ng[i] == sent_trim[i] for i in range(N - 1)]
            return(all(ok_gram))

        ok_grams = [ng 
                    for ng in self.pdf.freqdist() 
                    if gramMatch(ng, sent, self.n_gram)]

        probs = list(map(self.pdf.prob, ok_grams))
        probs = np.array(probs)
        probs = probs / np.sum(probs)

        rnd_index = np.random.choice(range(len(probs)),
                                     size = 1,
                                     replace = True,
                                     p = probs)

        rnd_ng = ok_grams[int(rnd_index)]

        return(rnd_ng[-1])

    def genSent(self, max_sent = None):
        new_sent = self.start_tokens.copy()

        rnd_word = ''
        while rnd_word != "END":

            if max_sent is not None and len(new_sent) + (self.n_gram - 1) >= max_sent:
                rnd_word = "END"
            else:
                rnd_word = self.getNextWord(new_sent)

            new_sent.append(rnd_word)

        sent_out = [w for w in new_sent if not re.search("(START|END)", w)]
        sent_out = ' '.join(sent_out)

        return(sent_out)

    def perplexity(self, sent):
        sent = nltk.tokenize.word_tokenize(sent.lower())
        sent = self.start_tokens + sent + ['END']
        sent_grams = nltk.ngrams(sent, self.n_gram)

        def okGram(gr, sent_gram, N):
            ok = [gr[n] == sent_gram[n] for n in range(N - 1)]
            return(all(ok))

        gram_probs = []
        for sg in sent_grams:
            ok_grams = [gr for (gr, n) in self.pdf.freqdist().items() if okGram(gr, sg, self.n_gram)]
            probs = [self.pdf.prob(gr) for gr in ok_grams]
            try:
                gram_prob = self.pdf.prob(sg) / sum(probs)
            except ZeroDivisionError:
                return None
            else:
                gram_probs.append(gram_prob)

        log_pp = -1 / len(gram_probs) * np.sum(np.log(gram_probs))
        return np.exp(log_pp)








