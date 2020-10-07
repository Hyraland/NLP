import numpy as np 
from nltk.corpus import brown
import matplotlib.pyplot as plt

# implementing word2vec using negative sampling
def word2idx_wordcount(sentences, limit = 0):
    count = 0
    wordidx = {}
    wordcount = {}
    for sentence in sentences:
        for s in sentence:
            if s not in wordidx:
                wordidx[s] = count
                wordcount[s] = 1
                count += 1
            else:
                wordcount[s] += 1
    return wordidx, wordcount

def pfirst(wordcount, wordidx, alpha=0.75):
    pw = np.zeros(len(wordcount))
    for w, i in wordcount.items():
        pw[wordidx[w]] = i**alpha
    pw = pw/sum(pw)
    return pw

def pdrop(pw, threshold=1e-5):
    pd = np.zeros(len(pw))
    for i in range(len(pw)):
        pd[i] = 1- np.sqrt(threshold/pw[i])
    return pd

def subsample(senidx, pd):
    newsenidx = []
    for i in senidx:
        if np.random.rand() > pd[i]:
            newsenidx.append(i)
    return newsenidx

def sigmoid(a):
    return 1/(1+np.exp(-a))


class word2vec:

    def fit(self, sentences, D, window, epochs = 200):

        self.wordidx, self.wordcount = word2idx_wordcount(sentences)
        V = len(self.wordidx)
        self.W1 = np.random.randn(V,D)
        self.W2 = np.random.randn(D,V)

        pw = pfirst(self.wordcount, self.wordidx)
        pd = pdrop(pw)

        losses = []
        for epoch in range(epochs):
            loss = 0
            for sentence in sentences:
                senidx = np.array([self.wordidx[s] for s in sentence])
                senidx = subsample(senidx, pd)
                n = len(senidx)
                for i,mid in enumerate(senidx):
                    if i == 0:
                        context = senidx[i+1:i+1+window]
                    elif 0 < i <= window:
                        context = senidx[0:i] + senidx[i+1:i+1+window]
                    elif n-1 > i >= n-window:
                        context = senidx[i-window:i]+senidx[i+1:]
                    elif i == n-1:
                        context = senidx[i-window:i]
                    else:
                        context = senidx[i-window:i]+senidx[i+1:i+1+window]
                    loss += self.SGD(mid, context, label = 1)
                    neg = np.random.choice(V, p=pw)
                    loss += self.SGD(neg, context, label = 0)
            losses.append(loss)
            print("Epoch:", epoch, "training loss: ", loss)

    
    def SGD(self, mid, context, label, lr = 0.01):
        prob = sigmoid(self.W1[mid].dot(self.W2[:, context]))
        gW2 = np.outer(self.W1[mid], prob-label)
        gW1 = self.W2[:, context].dot(prob-label)

        self.W2[:, context] -= lr*gW2
        self.W1[mid] -= lr*gW1

        return -(label*np.log(prob) + (1-label)*np.log(1-prob)).sum()

def get_sentences():
  # returns 57340 of the Brown corpus
  # each sentence is represented as a list of individual string tokens
  return brown.sents()

if __name__ == '__main__':
    sentences = get_sentences()

    w2v = word2vec()
    w2v.fit(sentences, D = 20, window = 2)

    plt.imshow(w2v.W1)
    plt.title("word2vec matrix")
    plt.show()





