import pandas as pd
import re
import nltk
import numpy as np
import os
letters = 'abcdefghijklmnopqrstuvwxyz '
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans




def preprocess(x):
    x = re.sub('[^a-z\s]', '', x.lower())                  
    return ' '.join(x)     

folder = nltk.data.find(
    '/Users/Sahit/Documents/GitHub/BBC_WorkSpace/PM_Speech_Analysis/Speeches/Speeches_Modi_Demo')

path = '/Users/Sahit/Documents/GitHub/BBC_WorkSpace/PM_Speech_Analysis/Speeches/Speeches_Modi_Demo'
dirListing = os.listdir(path)
editFiles = []
for item in dirListing:
    if ".txt" in item:
        editFiles.append(path+'/'+item)
print len(editFiles)

words_in_speech = []
no_of_words = []

text = []


def simple_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    return words

def get_dictionary_word_list(filepath):
    # with context manager assures us the
    # file will be closed when leaving the scope
    with open(filepath) as f:
        # return the split results, which is all the words in the file.
        return f.read().split()

speeches_df  = []


def detect_language(line):
    maxchar = ord(max(line))
    if 65 <= maxchar <= 90:
        return 1
    elif 97 <= maxchar <= 122:
        return 1
    else:
        return 0

for filepath in editFiles:
    
    speech = {}
        
    place = re.search(r'_([a-zA-Z]+)_.txt$', filepath)
    date = re.search( r'/Users/Sahit/Documents/GitHub/BBC_WorkSpace/PM_Speech_Analysis/Speeches/Speeches_Modi_Demo/\d+_(\d+)_([a-zA-Z]+)_(\d+)', filepath)
    #print date.group(1), date.group(2), date.group(3)
    
    l = get_dictionary_word_list(filepath)
    
    h = []
    
    for i in xrange(len(l)):
        if detect_language(l[i]) == 1:
            oo = l[i].decode('utf-8')
            h.append(oo)

    number_of_words = len(h)
    
    words_in_speech.append(h)
    
    text.extend(l)
    try:
        speech['city'] = place.group(1)
    except:
        speech['city'] = 'NA'  
        
        
    date_ =  ' '.join([date.group(1), date.group(2), date.group(3)])

    day = date.group(1)
    month = date.group(2)
    year = date.group(3)
    
        
    speech['date'] = date_
    speech['day'] = day
    speech['month'] = month
    speech['year'] = year
    speech['data'] = h
    speech['category'] = 0 
    speech['word_count'] = number_of_words
    speeches_df.append(speech)
    #print speeches
    
modi_df = pd.DataFrame(speeches_df)   
    
import matplotlib.pyplot as plt 
from mpl_toolkits.basemap import Basemap
import warnings
import matplotlib.cbook

from geopy.geocoders import Nominatim
geolocator = Nominatim()



from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')


words = list(set(text))   

print 'Number of words are ' + str(len(words))
    
word_freq = Counter(text)

# Common words list in order of most common word first, however have still not completely removed 
# Hindi mis-read words
Common_words_list = sorted(word_freq, key=word_freq.get, reverse=True)


modi_df['data']=[" ".join(review) for review in modi_df['data'].values]


modi_df['clean_data'] = modi_df['data'].apply(preprocess)

modi_df['tokenised'] = [" ".join(tokenizer.tokenize(review)).lower() for review in modi_df['data'].values]

vectorizer = TfidfVectorizer(
    use_idf=True, tokenizer=simple_tokenizer, 
    max_features=5000,
    stop_words='english')
X = vectorizer.fit_transform(modi_df['tokenised'])


# X is now our featured documents with inverse frequencies as individual feature values.
def c_distance(a, b):
    if a.shape[0] == b.shape[0]:
        r = 0.
        a = np.array(csr_matrix(a).todense()).T
        b = np.array(csr_matrix(b).todense())
        for i in xrange(a.shape[0]):
            r += a[i][0]*b[i][0]
        return r/((np.linalg.norm(a)*np.linalg.norm(b)))
    else:
        print 'Dimension mismatch'

number_of_clusters = 10
km = KMeans(n_clusters=number_of_clusters)

from scipy.sparse import csr_matrix

km.fit(X)
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
centroids = []
for i in range(number_of_clusters):
    top_words = [terms[ind] for ind in order_centroids[i, :7]]
    centroids.append(' '.join(top_words))
    print("Cluster {}: {}".format(i, ' '.join(top_words)))
      
category_ = np.zeros((520, 1))
for i in xrange(520):
    score = 0
    for j in xrange(number_of_clusters):
        v1 = order_centroids[j, :].T
        v2 = X[i, :].T
        p = c_distance(v1, v2)
        if j == 0:
            score = p
            category = 1
        else:
            if  p > score:
                score = p
                category = j+1
    modi_df.loc[i, 'category'] = category
                
'''
fig = plt.figure(num=None, figsize=(40, 32) )
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
#m.fillcontinents(color='tan', lake_color='lightblue', ax=fig)
m.drawparallels(np.arange(-90.,91.,30.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,60.),labels=[False,False,False,True],dashes=[2,2])
m.drawmapboundary(fill_color='lightblue')

count = 0

for i in xrange(520):
    try:
        r = modi_df.loc[i]
        location = geolocator.geocode(r['city'])
        longitude = location.longitude
        latitude = location.latitude  
        x,y = m(longitude, latitude)
        year_ = int(r['year'])
        if year_ == 2014: 
            m.plot(x,y,marker='o',color='r',markersize=10)
        elif year_ == 2015:
            m.plot(x,y,marker='o',color='m',markersize=10)
        elif year_ == 2016:
            m.plot(x,y,marker='o',color='g',markersize=10)
        elif year_ == 2017:
            m.plot(x,y,marker='o',color='y',markersize=10)
        count += 1
    except:
        None
        
    print count
    

plt.title("PM Speeches")   
plt.show() 
 
'''
'''
sizes = [0]*10
for i in xrange(520):
    sizes[modi_df.loc[i, 'category']-1] += 1
    
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red', 'white', 'tan', 'green', 'magenta', 'pink']

labels = ['Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5', 'Topic6', 'Topic7', 'Topic8', 'Topic9', 'Topic10']
plt.pie(sizes,  labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
    

plt.axis('equal')
plt.show()

'''

distribution = np.zeros((number_of_clusters,4))
for i in xrange(len(modi_df['category'].values)):
    r = modi_df.loc[i, 'category']
    p = modi_df.loc[i, 'year']
    distribution[int(r)-1][int(p)-2014] += 1
    
    
plt.figure(1)    
k =[2014, 2015, 2016, 2017]
for i in xrange(number_of_clusters):
    n = [ distribution[i][p] for p in xrange(4) ]
    plt.subplot()
    plt.plot(k, n, ) 
plt.show()    
      

