__author__ = 'claire'

import glob
import sqlite3 as lite

import numpy as np


def sql_query(query, db):
    '''
    return data from db accoring to sql query
    '''
    con = lite.connect(db)
    cur = con.cursor()
    cur.execute(query)
    cur.close()


d = {}
# path_articles=glob.glob('~/Dropbox/CBS\ Reports/Cases/Topics/*/10_topics/*/Artikel*')
path_articles = glob.glob('/Users/claire/Dropbox/CBS Reports/Cases/Topics/*/10_topics/*/*')
print len(path_articles)
print path_articles[0].split('/')[7], path_articles[0].split('/')[9], path_articles[0].split('/')[10]
list = np.array([[a.split('/')[7], a.split('/')[9], a.split('/')[10]] for a in path_articles])

print list

con = lite.connect('/Users/claire/Dropbox/PycharmProjects/media/INFO.db')

with con:
    cur = con.cursor()

    for i in list:
        if i[0] == 'digital signatur':
            # insert into ds_topics column
            cur.execute("UPDATE info SET DS_Topic=? WHERE id=?", (i[1], i[2]))
        if i[0] == 'overva\xcc\x8agning':
            cur.execute("UPDATE info SET OV_Topic=? WHERE id=?", (i[1], i[2]))
        if i[0] == 'nemID':
            cur.execute("UPDATE info SET NI_Topic=? WHERE id=?", (i[1], i[2]))

