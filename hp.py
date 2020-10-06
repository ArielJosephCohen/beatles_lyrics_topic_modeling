import warnings
warnings.filterwarnings('ignore')
#imports
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from urllib.request import urlopen
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
import pickle
import operator
from wordcloud import WordCloud
from PIL import Image
import string
import re
from textblob import TextBlob
import math
import seaborn as sns

band='beatles'

url = f'https://www.lyricsfreak.com/b/{band}/'
html = urlopen(url) 
soup = BeautifulSoup(html, 'html.parser')

songs=soup.find_all('div',class_="lf-list__cell lf-list__title lf-list__cell--full")

replace_text="https://www.lyricsfreak.com"

song_links = []

for song in songs:
    sl=song.find_all('a')[0]['href'].replace('..','https://www.azlyrics.com/')
    sl=replace_text+sl
    song_links.append(sl)
    
song_names_lst=[]
song_names_dict={}

i=0

for song in song_links[0:-6]:
    song_names_lst.append(song.replace('https://www.lyricsfreak.com/b/beatles/','').split('_')[0].replace('+',' '))
    song_names_dict[i]=song.replace('https://www.lyricsfreak.com/b/beatles/','').split('_')[0].replace('+',' ')
    i+=1
    
def scrape_lyrics(index,form):
    
    url=song_links[index]
    html = urlopen(url) 
    soup = BeautifulSoup(html, 'html.parser')
    
    lyric=soup.find_all('div',class_="lyrictxt js-lyrics js-share-text-content")[0]
    
    song_lyrics=[]

    count=0

    for i in list(lyric):
        try:
            song_lyrics.append(i.replace('  ','')[1:])
        except:
             count=count+1
    
    blur=[lyr for lyr in song_lyrics if lyr!='']
    clear=[lyr for lyr in song_lyrics if lyr!='']
    
    if form == 'b':
        end = blur
    else:
        end = clear
        
    return end

df=pd.DataFrame()

song_links=song_links[0:-6]

df['index']=song_names_dict.keys()

df['song']=song_names_dict.values()

df['lyrics']=df['index'].apply(lambda x: scrape_lyrics(x,'c'))

df.drop('index',axis=1,inplace=True)

eng_stop=stopwords.words('english')

df.reset_index(inplace=True)

df.drop('index',axis=1,inplace=True)

df['lyrics_mush']=df.lyrics.apply(lambda x: ' '.join(x))

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df['clean_lyrics']=df.lyrics_mush.apply(lambda x: clean_text(x))

df.drop('lyrics_mush',axis=1,inplace=True)

df['no_stop_lyrics']=None
for i in range(len(df)):
    df['no_stop_lyrics'][i]=[lyric for lyric in df.clean_lyrics[i].split() if lyric not in eng_stop]
    
for i in range(len(df)):
    df.clean_lyrics[i]=' '.join(df['no_stop_lyrics'][i])
    
df.drop('lyrics',axis=1,inplace=True)

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df['polarity'] = df['clean_lyrics'].apply(pol)
df['subjectivity'] = df['clean_lyrics'].apply(sub)

def split_text(text, n):
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list

df['chunks']=df.no_stop_lyrics.apply(lambda x: split_text(x,10))

def sentiment_over_song(polar_subject,index_input,song_title=None):
    
    chunk_sent=[]
    style=''
    
    if song_title!=None:
    
        rl_nm=[w.lower() for w in song_title.split()]
        rl_nm=' '.join(rl_nm)
        index=df[df.song==rl_nm].index[0]
    
    else:
        
        index=index_input
        
    if polar_subject == 'p':
        style='polarity'
        for chunk in df.chunks[index]:
            chunk_sent.append(TextBlob(' '.join(chunk)).sentiment.polarity)
    else:
        style='subjectivity'
        for chunk in df.chunks[index]:
            chunk_sent.append(TextBlob(' '.join(chunk)).sentiment.subjectivity)

    name=[word.title() for word in df.song[index].split()]
    name=' '.join(name)
            
    y = chunk_sent
            
    x = range(0,len(df.chunks[index]))
    
    
    
    print(f'lyrics per chunk: {len(df.chunks[index])}')
    print('-'*20)
    
    with sns.axes_style({'axes.facecolor':'k','axes.edgecolor':'r'}):
        plt.figure(figsize=(15,8))
        sns.lineplot(x,y,color='gold')
        plt.title(f'Song: {name}')
        plt.xlabel('<-- time of song -->')
        plt.ylabel(f'<-- {style} -->')
        plt.rcParams.update({'font.size': 25})
        plt.tight_layout()
        plt.show()