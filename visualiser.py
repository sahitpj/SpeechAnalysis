#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:03:34 2018

@author: Sahit
"""

import matplotlib.pyplot as plt 
from mpl_toolkits.basemap import Basemap
import numpy as np


class visualiser(object):
    def __init__(self, speeches_df):
        self.speeches_df = speeches_df
        
        
    def function_worldmap(self, speeches_df):
        from geopy.geocoders import Nominatim
        geolocator = Nominatim()
        
        fig = plt.figure(num=None, figsize=(12, 8) )
        m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c')
        m.drawcoastlines()
        #m.fillcontinents(color='tan',lake_color='lightblue')
        m.drawparallels(np.arange(-90.,91.,30.),labels=[True,True,False,False],dashes=[2,2])
        m.drawmeridians(np.arange(-180.,181.,60.),labels=[False,False,False,True],dashes=[2,2])
        m.drawmapboundary(fill_color='lightblue')
        count = 0
        for i in xrange(520):
            try:
                r = speeches_df.loc[i]
                location = geolocator.geocode(r['city'])
                longitude = location.longitude
                latitude = location.latitude  
                x,y = m(longitude, latitude)
                year_ = int(r['year'])
                if year_ == 2014: 
                    m.plot(x,y,marker='o',color='r',markersize=2)
                elif year_ == 2015:
                    m.plot(x,y,marker='o',color='m',markersize=2)
                elif year_ == 2016:
                    m.plot(x,y,marker='o',color='g',markersize=2)
                elif year_ == 2017:
                    m.plot(x,y,marker='o',color='y',markersize=2)
                count += 1
            except:
                None
                
        plt.title("PM Speeches")   
        plt.show() 
        
    def pie_chart(self, speeches_df, number_of_clusters=10):
        #For this example we take the number of clusters as 10, so that we can fix the labels and also the colors 
        sizes = [0]*number_of_clusters
        for i in xrange(520):
            sizes[speeches_df.loc[i, 'category']-1] += 1
            
        colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red', 'white', 'tan', 'green', 'magenta', 'pink']
        
        labels = ['Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5', 'Topic6', 'Topic7', 'Topic8', 'Topic9', 'Topic10']
        plt.pie(sizes,  labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
            
        
        plt.axis('equal')
        plt.show()
        
    def speech_timeline(self, speeches_df, number_of_clusters=10):
        distribution = np.zeros((number_of_clusters,4))
        for i in xrange(len(speeches_df['category'].values)):
            r = speeches_df.loc[i, 'category']
            p = speeches_df.loc[i, 'year']
            distribution[int(r)-1][int(p)-2014] += 1
            
            
        plt.figure(1)    
        k =[2014, 2015, 2016, 2017]
        for i in xrange(number_of_clusters):
            n = [ distribution[i][p] for p in xrange(4) ]
            plt.subplot()
            plt.plot(k, n, ) 
        plt.show()    
      

                