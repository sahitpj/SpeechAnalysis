#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:03:34 2018

@author: Sahit
"""

import matplotlib.pyplot as plt 
from mpl_toolkits.basemap import Basemap
import warnings
import matplotlib.cbook
import numpy as np


class map_visualiser(object):
    def __init__(self, speeches_df):
        self.speeches_df = speeches_df
        
        
    def function(self, speeches_df):
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
        
    def 
                