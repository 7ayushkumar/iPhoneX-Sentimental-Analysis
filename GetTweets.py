
# coding: utf-8



import re
import tweepy

import csv

import pandas as pd



CONSUMER_KEY = 'c4cOOivpIx2u9i415HXLgqbxL'
CONSUMER_SECRET = 'ba7dQIGfAH0OMbkz6iVR5sq2e2iS8WBlzilnL09MZ5MQZAPcbU'
ACCESS_TOKEN = '60263710-8ccRGLIH55ENveaLcVvam0j4kScMnfN7nklNdyBSL'
ACCESS_TOKEN_SECRET = 'BGZ7AXmJ6dXIaQidtN2ZIUqMcTA22QZrzX3lnR4Rmmmmx'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth,wait_on_rate_limit=True)



status = "Tweeting through Python scripts, Check! #python #twitter #tweepy #datascience"
api.update_status(status=status)


tweets=[]


# In[39]:

for tweet in tweepy.Cursor(api.search,q="#iPhoneX",
                           lang="en",
                           since="2017-12-01").items():
    tweets.append(tweet)


# In[58]:

print("Number of tweets extracted: {}.\n".format(len(tweets)))
for tweet in tweets[:5]:
    print(tweet.text)
    print()


# In[59]:

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])


# In[61]:

data.head()


# In[74]:

print(tweets[0].id)
print(tweets[0].created_at)
print(tweets[0].source)
print(tweets[0].favorite_count)
print(tweets[0].retweet_count)
print(tweets[0].geo)
print(tweets[0].coordinates)
print(tweets[0].entities)


# In[75]:

data['len']  = np.array([len(tweet.text) for tweet in tweets])
data['ID']   = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])


# In[76]:

data.head()


# In[5]:

from textblob import TextBlob
import re


# In[78]:

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


# In[100]:

def analyze_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


# In[101]:

data['Sentiment'] = np.array([ analyze_sentiment(tweet) for tweet in data['Tweets'] ])


# In[16]:

data.head(10)


# In[103]:

FILE_PATH="~/Downloads/Sentimental Analysis/Tweets_Sentiment.csv"


# In[105]:

data.to_csv(FILE_PATH)


# In[6]:

data = pd.read_csv('Tweets_Sentiment.csv')


# In[7]:

data.head()


# In[8]:

data = data[['Tweets','Sentiment']]


# In[9]:

data.head()


# In[113]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets


# In[10]:

import nltk


# In[11]:

from nltk.corpus import stopwords


# In[12]:

from nltk.classify import SklearnClassifier


# In[ ]:

# how to install wordcloud


# In[ ]:

# conda install -c conda-forge wordcloud 


# In[15]:

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[16]:

data = pd.read_csv('Tweets_Sentiment.csv')
data = data[['Tweets','Sentiment']]


# In[17]:

from sklearn.model_selection import train_test_split # function for splitting data to train and test sets


# In[20]:

# Splitting the dataset into train and test set
train, test = train_test_split(data,test_size = 0.1)
# Removing neutral sentiments
train = train[train.Sentiment != -1]


# In[21]:

train_pos = train[ train['Sentiment'] == 1]
train_pos = train_pos['Tweets']
train_neg = train[ train['Sentiment'] == 0]
train_neg = train_neg['Tweets']


# In[22]:

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# In[24]:

print("Positive words")
wordcloud_draw(train_pos,'white')

print("Negative words")
wordcloud_draw(train_neg)







