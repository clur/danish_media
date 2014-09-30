__author__ = 'claire'

import sqlite3 as lite
import codecs
import re

from sklearn.feature_extraction import text
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt


_STOP = [u'og', u'i', u'jeg', u'det', u'at', u'en', u'den', u'til', u'er', u'som', u'p\xe5', u'de', u'med', u'han',
         u'af', u'for', u'der', u'var', u'mig', u'sig', u'men', u'et', u'har', u'om', u'vi', u'min', u'havde', u'ham',
         u'hun', u'nu', u'over', u'da', u'fra', u'du', u'ud', u'sin', u'dem', u'os', u'op', u'man', u'hans', u'hvor',
         u'eller', u'hvad', u'skal', u'selv', u'her', u'alle', u'vil', u'blev', u'kunne', u'ind', u'n\xe5r', u'v\xe6re',
         u'dog', u'noget', u'ville', u'jo', u'deres', u'efter', u'ned', u'skulle', u'denne', u'end', u'dette', u'mit',
         u'ogs\xe5', u'under', u'have', u'dig', u'anden', u'hende', u'mine', u'alt', u'meget', u'sit', u'sine', u'vor',
         u'mod', u'disse', u'hvis', u'din', u'nogle', u'hos', u'blive', u'mange', u'ad', u'bliver', u'hendes',
         u'v\xe6ret', u'thi', u'jer', u's\xe5dan', u's\xe5']


def pull(query, db):
    '''
    return data from db accoring to sql query
    '''
    con = lite.connect(db)
    cur = con.cursor()
    cur.execute(query)
    col = cur.fetchall()
    cur.close()
    return col


def NMFtopic(data, pubnames, features):
    '''
    non negative matrix factorization topic modeling, print top words in each topic
    -digits removed
    -words must occur in more than 1 document or less than 10% of all documents
    '''
    data = [re.sub(u'\d+', ' ', d) for d in data]
    # vectorizer = text.TfidfVectorizer(max_features=features,stop_words=_STOP,ngram_range=(1,1),min_df=2,max_df=0.1)
    vectorizer = text.TfidfVectorizer(stop_words=_STOP, ngram_range=(1, 1), min_df=2, max_df=0.1)

    dtm = vectorizer.fit_transform(data)
    print 'Document Matrix:', dtm.shape
    feature_names = np.array(vectorizer.get_feature_names())
    print len(feature_names), 'features extracted\n'
    for ntopics in [5, 10, 15, 20, 25, 30]:
        print 'finding', ntopics, 'topics'
        nmf = decomposition.NMF(n_components=ntopics, random_state=1)
        doctopic = nmf.fit_transform(dtm)
        topic_words = []
        for topic in nmf.components_:
            word_idx = np.argsort(topic)[::-1][0:20]
            topic_words.append([feature_names[i] for i in word_idx])
        with codecs.open('topic_plots/' + str(ntopics) + '_topics_topwords', 'w', 'utf8') as f:
            f.write('Topics : top words:\n')
            for i, j in enumerate(topic_words):
                f.write(str(i) + ':')
                f.write(', '.join(j))
                f.write('\n')

        doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
        pubnames = np.asarray(pubnames)
        num_groups = len(set(pubnames))
        doctopic_grouped = np.zeros((num_groups, ntopics))
        for i, name in enumerate(sorted(set(pubnames))):
            doctopic_grouped[i, :] = np.mean(doctopic[pubnames == name, :], axis=0)
        doctopic = doctopic_grouped
        pubs = sorted(set(pubnames))
        # stacked_plot(doctopic,pubs)       #uncomment for plots, need folder called topic_plots/
        # heatmap_plot(doctopic,pubs)


def stacked_plot(doctopic, pubnames):
    # for plotting

    N, K = doctopic.shape
    ind = np.arange(N)
    width = 0.5
    plots = []
    height_cumulative = np.zeros(N)
    for k in range(K):
        color = plt.cm.coolwarm(float(k) / K, 1)
        if k == 0:
            p = plt.bar(ind, doctopic[:, k], width, color=color)
        else:
            p = plt.bar(ind, doctopic[:, k], width, bottom=height_cumulative, color=color)
        height_cumulative += doctopic[:, k]
        plots.append(p)
    plt.ylim((0, 1))
    plt.ylabel('Topics')
    plt.title('Topics in publications')
    plt.xticks(ind + width / 2, pubnames, rotation='vertical')
    plt.yticks(np.arange(0, 1, 10))
    topic_labels = ['#{}'.format(k) for k in range(K)]

    plt.legend([p[0] for p in plots], topic_labels, fontsize='x-small')
    # plt.tight_layout()
    plt.savefig('topic_plots/' + str(K) + '_topic_plot')
    plt.clf()


def heatmap_plot(doctopic, pubnames):
    # for plotting

    N, K = doctopic.shape
    topic_labels = ['#{}'.format(k) for k in range(K)]
    plt.pcolor(doctopic, norm=None, cmap='Blues')
    plt.yticks(np.arange(doctopic.shape[0]) + 0.5, pubnames)
    plt.xticks(np.arange(doctopic.shape[1]) + 0.5, topic_labels)
    plt.gca().invert_yaxis()
    plt.xticks(rotation=90)
    plt.colorbar(cmap='Blues')
    # plt.tight_layout()
    plt.savefig('topic_plots/' + str(K) + '_topic_heat')
    plt.clf()
    # o
    #


if __name__ == "__main__":
    # data=pull('Select * from dupetest where OV>0 and Article like "%nowden%"','INFO.db')
    # data=pull('Select * from dupetest where OV>0','INFO.db')
    data = pull(
        'Select * from info where (NI>0) and (Pub = "B.T." or Pub="Jyllands-Posten" or Pub="Berlingske" or Pub="Information" or Pub="Ekstra Bladet" or Pub="Politiken")',
        'INFO.db')
    # random.shuffle(data)

    articles = [d[3] for d in data]
    pubs = [d[2] for d in data]
    from collections import Counter

    print len(data), 'articles pulled from database from:'
    print Counter(pubs)
    import sys

    nfeatures = int(sys.argv[1])  # set number of features, default is 10000
    NMFtopic(articles, pubs, nfeatures)

