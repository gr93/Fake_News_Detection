import numpy as np
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from nltk.corpus import stopwords
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC


def remove_stop_words(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)

def clean_up_text(text):
    text = remove_stop_words(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def construct_labeled_sentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences


def get_word_embeddings(path,vector_dimension=300):
    data = pd.read_csv(path)

    missing_rows = []
    for i in range(len(data)):
        if data.loc[i, 'text'] != data.loc[i, 'text']:
            missing_rows.append(i)
    data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

    for i in range(len(data)):
        data.loc[i, 'text'] = clean_up_text(data.loc[i,'text'])

    x = construct_labeled_sentences(data['text'])
    y = data['label'].values

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.epochs)

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    xtrain = np.zeros((train_size, vector_dimension))
    xtest = np.zeros((test_size, vector_dimension))
    ytrain = np.zeros(train_size)
    ytest = np.zeros(test_size)

    for i in range(train_size):
        xtrain[i] = text_model.docvecs['Text_' + str(i)]
        ytrain[i] = y[i]
    j = 0
    for i in range(train_size, train_size + test_size):
        xtest[j] = text_model.docvecs['Text_' + str(i)]
        ytest[j] = y[i]
        j = j + 1

    return xtrain, xtest, ytrain, ytest


def plot_cmat(yte, ypred,title):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred,normalize=True)
    plt.title(title)
    plt.show()

if not (os.path.isfile('./xtr.npy') and os.path.isfile('./xte.npy') and os.path.isfile('./ytr.npy') and os.path.isfile('./yte.npy')):
    xtr,xte,ytr,yte = get_word_embeddings("datasets/train.csv")
    np.save('./xtr', xtr)
    np.save('./xte', xte)
    np.save('./ytr', ytr)
    np.save('./yte', yte)

xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')


bnb = BernoulliNB()
bnb.fit(xtr,ytr)
y_pred = bnb.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
print("Naive Bayes Accuracy = " + format((m-n)/m*100, '.2f') + "%")

# Draw the confusion matrix
plot_cmat(yte, y_pred, "Naive Bayes confusion matrix")


clf = LinearSVC ()
clf.fit(xtr, ytr)
y_pred = clf.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
print("SVM Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 88.42%

# Draw the confusion matrix
plot_cmat(yte, y_pred, "SVM confusion matrix")




