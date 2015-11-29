# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 claus

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
import os
from enron_read import process_mails
from optparse import OptionParser
import sys
from time import time
#
# MAIL_BODY = re.compile(r'\n\n(.+?)($|---)',re.DOTALL|re.MULTILINE)
# def read_enron_mails(users):
#     dataset = []
#     for user in users:
#         for root, dirs, files in os.walk(user, topdown=False):
#             for file in files:
#                 try:
#                     body = MAIL_BODY.search(open(root+'\\'+file).read()).group(1)
#                     dataset.append(body)
#                 except:
#                     print root,file
#     return dataset

N_CLUSTERS = 5

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")


(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

if not os.path.exists('model.pkl'):

    dataset = list([x[-1] for x in process_mails(glob.glob('maildir\\*'),ratio=0.05)])
    print 'Read %d mails' % len(dataset)
    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english', non_negative=True,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words='english',
                                           non_negative=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=opts.use_idf, analyzer='word', token_pattern='[a-zA-Z]{3,}')

    X = vectorizer.fit_transform(dataset)

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)


    if opts.n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print()


    ##############################################################################
    # Do the actual clustering

    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=N_CLUSTERS, init='k-means++', max_iter=100, n_init=1,
                    verbose=opts.verbose)

    print("Clustering sparse data with %s" % km)
    t0 = time()
    predicted = km.fit_predict(X)

    print("done in %0.3fs" % (time() - t0))
    label_counts = pd.Series(predicted).value_counts()

    print

    if not opts.use_hashing:
        print("Top terms per cluster:")

        if opts.n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in label_counts.index:
            print "Cluster %d (%d):" % (i, label_counts[i])
            for ind in order_centroids[i, :25]:
                print ' %s' % terms[ind],
            print

    pickle.dump((vectorizer, km), open('model.pkl','wb'))
    sys.exit(0)
else:
    print 'Loaded vectorizer and clusterer from model.pkl'
    vectorizer, km = pickle.load(open('model.pkl','rb'))
    # for i in range(km.cluster_centers_.shape[0]):
    #     plt.plot(sorted(km.cluster_centers_[i,:]),'.',label=i)
    #     print km.cluster_centers_[i,:].max()
    # plt.legend()
    # plt.show()
    terms = vectorizer.get_feature_names()
    for i in range(km.cluster_centers_.shape[0]):
        for ind in km.cluster_centers_[i,:].argsort()[::-1][:25]:
            print ' %s' % terms[ind],
        print

if not os.path.exists('df.pkl'):
    df = pd.DataFrame()
    for userdir in glob.glob('maildir\\[a-m]*'):
        user_data = list(process_mails([userdir],ratio=0.25))
        user_bodies = [x[-1] for x in user_data]
        Xuser = vectorizer.transform(user_bodies)
        predicted_user_labels = km.predict(Xuser)
        user_df = pd.DataFrame(index = [x[0] for x in user_data])
        user_df.index = pd.to_datetime(user_df.index)
        user_df['label'] = predicted_user_labels
        user_df['user'] = [x[1] for x in user_data]
        df = df.append(user_df)

    pickle.dump(df, open('df.pkl','wb'))
else:
    df = pickle.load(open('df.pkl','rb'))

df['month'] = df.index.month
df['year'] = df.index.year
# df['hourinweek'] = df.index.dayofweek*24 + df.index.hour
# grouped = df.groupby('user')
# for k,g in grouped:
#     if len(g) < 250: continue
#     plt.plot(np.histogram(g['hourinweek'],bins=range(168),normed=True)[0])
# plt.show()
grouped = df.groupby(['year','month'])
hist_df = {}
for key, group in grouped:
    hist = group['label'].value_counts()
    hist_df[key] = hist

hist_df = pd.DataFrame(hist_df).fillna(0).transpose()
hist_df_normed = hist_df.astype(float).div(hist_df.sum(axis=1), axis=0)
print hist_df.head(50)
hist_df.plot(kind='bar',stacked=True)
hist_df_normed.plot(kind='bar',stacked=True)
plt.show()

