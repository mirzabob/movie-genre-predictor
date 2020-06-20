import urllib
import requests
import json
import time
import itertools
import os
import tmdbsimple as tmdb
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pickle

#Setting the path where scraped folders to be saved
poster_folder='posters_final/'
if poster_folder.split('/')[0] in os.listdir('./'):
    print('Folder already exists')
else:
    os.mkdir('./'+poster_folder)
    
poster_folder

#Working on an example to scrap data
api_key = 'db3240ee08c956b119e4ecf0aa69199f'

tmdb.API_KEY = api_key
search=tmdb.Search()

def grab_poster_tmdb(movie):
    response = search.movie(query=movie)
    id=response['results'][0]['id']
    movie = tmdb.Movies(id)
    posterp=movie.info()['poster_path']
    title=movie.info()['original_title']
    url='image.tmdb.org/t/p/original'+posterp
    title='_'.join(title.split(' '))
    strcmd='wget -O '+poster_folder+title+'.jpg '+url
    os.system(strcmd)

def get_movie_id_tmdb(movie):
    response = search.movie(query=movie)
    movie_id=response['results'][0]['id']
    return movie_id

def get_movie_info_tmdb(movie):
    response = search.movie(query=movie)
    id=response['results'][0]['id']
    movie = tmdb.Movies(id)
    info=movie.info()
    return info

def get_movie_genres_tmdb(movie):
    response = search.movie(query=movie)
    id=response['results'][0]['id']
    movie = tmdb.Movies(id)
    genres=movie.info()['genres']
    return genres


# =============================================================================
# print (get_movie_genres_tmdb("The Matrix"))
# 
# info=get_movie_info_tmdb("The Matrix")
# print ("All the Movie information from TMDB gets stored in a dictionary with the following keys for easy access -")
# print (info.keys())
# 
# info=get_movie_info_tmdb("The Matrix")
# print (info['tagline'])
# =============================================================================



#Working with multiple movies
all_movies=tmdb.Movies()
top_movies=all_movies.popular()

print(len(top_movies['results']))
top20_movs=top_movies['results']

first_movie=top20_movs[0]
print ("Here is all the information you can get on this movie - ")
print (first_movie)
print ("\n\nThe title of the first movie is - ", first_movie['title'])




#Movie Title
for i in range(len(top20_movs)):
    mov=top20_movs[i]
    title=mov['title']
    print (title)
    if i==4:
        break



#Movie Genre    
for i in range(len(top20_movs)):
    mov=top20_movs[i]
    genres=mov['genre_ids']
    print (genres)
    if i==4:
        break    



#Genre list
genres=tmdb.Genres()
list_of_genres=genres.list()['genres']

#Genre id to name
Genre_ID_to_name={}
for i in range(len(list_of_genres)):
    genre_id=list_of_genres[i]['id']
    genre_name=list_of_genres[i]['name']
    Genre_ID_to_name[genre_id]=genre_name
    
for i in range(len(top20_movs)):
    mov=top20_movs[i]
    title=mov['title']
    genre_ids=mov['genre_ids']
    genre_names=[]
    for id in genre_ids:
        genre_name=Genre_ID_to_name[id]
        genre_names.append(genre_name)
    print(title,genre_names)
    if i==4:
        break

#Top 1000 movies
#all_movies=tmdb.Movies()
#top1000_movies=[]
#print('Pulling movie list, Please wait...')
#for i in range(1,51):
#    if i%15==0:
#        time.sleep(7)
#    movies_on_this_page=all_movies.popular(page=i)['results']
#    top1000_movies.extend(movies_on_this_page)
#len(top1000_movies)
#f3=open('movie_list.pckl','wb')
#pickle.dump(top1000_movies,f3)
#f3.close()
#print('Done!')


f3=open('movie_list.pckl','rb')
top1000_movies=pickle.load(f3)
f3.close()

#Generating all possible pairs
def list2pairs(l):
    pairs= list(itertools.combinations(l,2))
    #to take the duplicates
    for i in l:
        pairs.append((i,i))
    return pairs

#get genre list pairs
allPairs=[]
for movie in top1000_movies:
    allPairs.extend(list2pairs(movie['genre_ids']))


nr_ids=np.unique(allPairs)
visGrid=np.zeros((len(nr_ids),len(nr_ids)))
for p in allPairs:
    #print(np.argwhere(nr_ids==p[0]))
    visGrid[np.argwhere(nr_ids==p[0]),np.argwhere(nr_ids==p[1])]+=1
    if p[1]!=p[0]:
        visGrid[np.argwhere(nr_ids==p[1]),np.argwhere(nr_ids==p[0])]+=1
                
annot_lookup = []
for i in range(len(nr_ids)):
    annot_lookup.append(Genre_ID_to_name[nr_ids[i]])

sns.heatmap(visGrid, xticklabels=annot_lookup, yticklabels=annot_lookup)    


#Extract 100 movies per genre
movies=[]
baseyear=2020

print('Starting pulling movies from TMDB')
done_ids=[]
for g_id in nr_ids:
    baseyear-=1
    for page in range(1,6,1):
        time.sleep(0.5)
        
        url = 'https://api.themoviedb.org/3/discover/movie?api_key=' + api_key
        url += '&language=en-US&sort_by=popularity.desc&year=' + str(baseyear) 
        url += '&with_genres=' + str(g_id) + '&page=' + str(page)
        
        data= urllib.request.urlopen(url).read()
        
        dataDict=json.loads(data)
        movies.extend(dataDict["results"])
    done_ids.append(str(g_id))
print("Pulled movies for genres - "+','.join(done_ids))
        
#f6=open("movies_for_posters",'wb')
#pickle.dump(movies,f6)
#f6.close()

f6=open("movies_for_posters",'rb')
movies=pickle.load(f6)
f6.close()

movie_ids = [m['id'] for m in movies]
print("originally we had ",len(movie_ids)," movies")
movie_ids=np.unique(movie_ids)
print(len(movie_ids))
seen_before=[]
no_duplicate_movies=[]
for i in range(len(movies)):
    movie=movies[i]
    id=movie['id']
    if id in seen_before:
        continue
#         print "Seen before"
    else:
        seen_before.append(id)
        no_duplicate_movies.append(movie)
print("After removing duplicates we have ",len(no_duplicate_movies), " movies")

poster_movies=[]
counter=0
movies_no_poster=[]
print("Total movies : ",len(movies))
print("Started downloading posters...")
for movie in movies:
    id=movie['id']
    title=movie['title']
    if counter==1:
        print('Downloaded first. Code is working fine. Please wait, this will take quite some time...')
    if counter%300==0 and counter!=0:
        print("Done with ",counter," movies!")
        print("Trying to get poster for ",title)
    try:
        grab_poster_tmdb(title)
        poster_movies.append(movie)
    except:
        try:
            time.sleep(7)
            grab_poster_tmdb(title)
            poster_movies.append(movie)
        except:
            movies_no_poster.append(movie)
    counter+=1
print("Done with all the posters!")

print(len(movies_no_poster))
print(len(poster_movies))


#f=open('poster_movies.pckl','wb')
#pickle.dump(poster_movies,f)
#f.close()

f=open('poster_movies.pckl','rb')
poster_movies=pickle.load(f)
f.close()

# f=open('no_poster_movies.pckl','wb')
# pickle.dump(movies_no_poster,f)
# f.close()

f=open('no_poster_movies.pckl','rb')
movies_no_poster=pickle.load(f)
f.close()


#Builing the Dataset
movies_with_overviews=[]
for i in range(len(no_duplicate_movies)):
    movie=no_duplicate_movies[i]
    id=movie['id']
    overview=movie['overview']
    
    if(len(overview)==0):
        continue
    else:
        movies_with_overviews.append(movie)
    
len(movies_with_overviews)

genres=[]
all_ids=[]
for i in range(len(movies_with_overviews)):
    movie=movies_with_overviews[i]
    id=movie['id']
    genre_ids=movie['genre_ids']
    genres.append(genre_ids)
    all_ids.extend(genre_ids)
        
from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(genres)

genres[1]

print(Y.shape)
print(np.sum(Y,axis=0))

len(list_of_genres)

from sklearn.feature_extraction.text import CountVectorizer
import re

content=[]
for i in range(len(movies_with_overviews)):
    movie=movies_with_overviews[i]
    id=movie['id']
    overview=movie['overview']
    overview=overview.replace(',','')
    overview=overview.replace('.','')
    content.append(overview)

print(content[0])
print(len(content))

vectorize=CountVectorizer(max_df=0.95, min_df=0.005)
X=vectorize.fit_transform(content)

X.shape

f4=open('X.pckl','wb')
f5=open('Y.pckl','wb')
pickle.dump(X,f4)
pickle.dump(Y,f5)
f6=open('Genredict.pckl','wb')
pickle.dump(Genre_ID_to_name,f6)
f4.close()
f5.close()
f6.close()

f7=open('movies_with_overviews.pckl','wb')
pickle.dump(movies_with_overviews,f7)
f7.close()
#print(len(movies_with_overviews))
#X.shape[0]