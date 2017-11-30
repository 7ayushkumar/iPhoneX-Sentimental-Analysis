
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


# In[62]:

csvFile = open("Tweets.csv", 'a')


# In[63]:

csvWriter = csv.writer(csvFile)


# In[66]:

tweets=[]


# In[67]:

for tweet in tweepy.Cursor(api.sear
                           /-p;ch,q="#unitedAIRLINES",count=100,
                           lang="en",
                           since="2017-04-03").items():
    tweets.append(tweet)


# In[68]:

tweets_df = pd.DataFrame(vars(tweets[i]) for i in range(len(tweets)))


# In[65]:

csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])


# In[69]:

FILE_PATH="~/Downloads/Sentimental Analysis/Tweets.csv"


# In[70]:

tweets_df.to_csv(FILE_PATH)


# In[ ]:



