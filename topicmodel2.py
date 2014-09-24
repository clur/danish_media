__author__ = 'claire'

import sqlite3 as lite
import codecs
import os
import csv
import time

from sklearn.feature_extraction import text
from sklearn import decomposition
import numpy as np


_STOP=[u'og', u'i', u'jeg', u'det', u'at', u'en', u'den', u'til', u'er', u'som', u'p\xe5', u'de', u'med', u'han', u'af', u'for', u'der', u'var', u'mig', u'sig', u'men', u'et', u'har', u'om', u'vi', u'min', u'havde', u'ham', u'hun', u'nu', u'over', u'da', u'fra', u'du', u'ud', u'sin', u'dem', u'os', u'op', u'man', u'hans', u'hvor', u'eller', u'hvad', u'skal', u'selv', u'her', u'alle', u'vil', u'blev', u'kunne', u'ind', u'n\xe5r', u'v\xe6re', u'dog', u'noget', u'ville', u'jo', u'deres', u'efter', u'ned', u'skulle', u'denne', u'end', u'dette', u'mit', u'ogs\xe5', u'under', u'have', u'dig', u'anden', u'hende', u'mine', u'alt', u'meget', u'sit', u'sine', u'vor', u'mod', u'disse', u'hvis', u'din', u'nogle', u'hos', u'blive', u'mange', u'ad', u'bliver', u'hendes', u'v\xe6ret', u'thi', u'jer', u's\xe5dan',u's\xe5']


def pull(query, db):
    '''
    wrapper for sqlite
    return data from db accoring to sql query
    '''
    con = lite.connect(db)
    cur = con.cursor()
    cur.execute(query)
    col = cur.fetchall()
    cur.close()
    return col


def NMFtopic(data, term, ntopics=5, features=10000):
    """
    non negative matrix factorization topic modeling.
    This method generates topics specified by num_topics parameter.
    A folder called num_topics+'_topics' is created with subfolders for each topic in the current directory
    and articles pertaining to that topic are written to that folder.
    The top words in each topic are written to the num_topics+'_topics' folder.
    A .csv file is written to the num_topics+'_topics' folder with ['Artikel-id','Topic with highest share', 'Date', 'Publication']

    TFIDF vectors are used to represent the texts with the following settings:
    -the maximum number of features generated is set by the parameter features, the default value is 10,000.
    Note, the number of features can greatly decrease the speed of the function.
    -stop words given by the list _STOP are ignored
    -only unigrams are taken into account
    -words must occur in more than 1 document or less than 10% of all documents

    @param data: a list of utf-8 lists consisting of ids, publications, dates and articles
    @param ntopics: integer, the number of topics to find, default is 5
    @param features: integer, the number of features to use, default is 10000
    """
    vectorizer = text.TfidfVectorizer(max_features=features,stop_words=_STOP,ngram_range=(1,1),min_df=2,max_df=0.1)
    articles=[d[3] for d in data]
    dtm = vectorizer.fit_transform(articles)
    print 'document matrix:', dtm.shape
    feature_names = np.array(vectorizer.get_feature_names())
    print len(feature_names), 'features extracted'
    print 'finding', ntopics, 'topics...'
    nmf = decomposition.NMF(n_components=ntopics, random_state=1)
    doctopic = nmf.fit_transform(dtm)
    topic_words = []
    for topic in nmf.components_:
        word_idx = np.argsort(topic)[::-1][0:20]
        topic_words.append([feature_names[i] for i in word_idx])

    doctopic = doctopic / np.sum(doctopic, axis=1,
                                 keepdims=True)  # we will scale the document-component matrix such that the component values associated with each document sum to one.
    ids = np.asarray([d[0] for d in data])
    # make folder setup
    dir = term + '_' + str(ntopics) + '_topics'
    try:
        os.mkdir(dir)
    except:
        pass
    for subdir in range(ntopics):
        try:
            os.mkdir(dir + '/' + str(subdir))
        except:
            pass
    print 'writing top words to file'
    with codecs.open(dir + '/' + str(ntopics) + '_topics_topwords', 'w', 'utf8') as f:
        f.write('Topics : top words:\n')
        for i, j in enumerate(topic_words):
            f.write(str(i) + ':')
            f.write(', '.join(j))
            f.write('\n')
    print 'writing articles to folders'
    with open(dir + '/' + str(ntopics) + '_topics.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['Artikel', 'Topic with highest share', 'Date', 'Publication']
        header.extend(range(ntopics))
        writer.writerow(header)
        for i in range(len(doctopic)):
            topic = np.argsort(doctopic[i, :])[::-1][0]
            out = list([ids[i], topic, data[i][1], data[i][2]])
            out.extend(doctopic[i])
            writer.writerow(out)
            with codecs.open(dir + '/' + str(topic) + '/' + ids[i], 'w', 'utf8') as f:
                f.write(articles[i])



if __name__=="__main__":
    import sys
    start=time.time()
    term = sys.argv[1]  # term can be OV,NI,DS
    if term not in ['OV', 'NI', 'DS']:
        print 'Term must be OV, NI or DS'
        sys.exit(0)
    #pull the data
    data = pull(
        'Select * from dupetest where (' + term + '>0) and (Pub = "B.T." or Pub="Jyllands-Posten" or Pub="Berlingske" or Pub="Information" or Pub="Ekstra Bladet" or Pub="Politiken")',
        'INFO.db')
    # data=pull('Select * from dupetest where (DS>0) and (Pub = "B.T." or Pub="Jyllands-Posten" or Pub="Berlingske" or Pub="Information" or Pub="Ekstra Bladet" or Pub="Politiken")','INFO.db')
    # data=pull('Select * from dupetest where (OV>0) and (Pub = "B.T." or Pub="Jyllands-Posten" or Pub="Berlingske" or Pub="Information" or Pub="Ekstra Bladet" or Pub="Politiken")','INFO.db')
    pubs=[d[2] for d in data]
    from collections import Counter

    print len(data), 'articles pulled from database'
    c = Counter(pubs)
    for i in c.iteritems():
        print '%s : %s' % (i[0], i[1])

    ntopics = int(sys.argv[2])
    nfeatures = int(sys.argv[3])  # set number of features, default is 10000
    NMFtopic(data, term, ntopics, nfeatures)
    print 'took', round((time.time() - start) / 60, 2), 'minutes'
