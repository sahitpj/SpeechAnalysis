#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:59:13 2018

@author: Sahit
"""
from sklearn.cluster import KMeans

class Kmeans_cluster(object):
    def __init__(self, X_features, no_of_cluster=None, vectorizer, speeches_df):
        self.X_features = X_features
        self.number_of_cluster = no_of_clusters
        
    def cluster(self):
        km = Kmean(n_clusters=self.number_of_clusters)
        
        km.fit(X)

        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(number_of_clusters):
            top_words = [terms[ind] for ind in order_centroids[i, :7]]
            print("Cluster {}: {}".format(i, ' '.join(top_words)))
            
            
            