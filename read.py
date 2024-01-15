import numpy as np
import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

#we will make tags using overview,genres, keywords, cast, crew. In order to do that we need to do data preprocessing, which include removing missing data and removing duplicate data

movies.dropna(inplace = True) 
#above line of code will drop the null columns 
movies.isnull().sum() 
#above line can be used to check is null columns 

#ast.literal_eval() this is used to convert list into dictionry 
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
#above function is used to get name of genres from every row
movies['genres'] = movies['genres'].apply(convert)
#This above line will change the genre for every row of the dataset
movies['keywords'] = movies['keywords'].apply(convert)
def convert3(obj):
    count=0
    L = []
    for i in ast.literal_eval(obj):
        if count < 4:
            L.append(i['name'])
            count+=1
        else:
            break
    return L
movies['cast'] = movies['cast'].apply(convert3)

def convert4(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(convert4)
#we will convert Overview from string to list because we need to merge it with genres, cast, crew in order to make a tag 
movies['overview'] = movies['overview'].apply(lambda x:x.split())

#We will remove space from the tags because of potential duplication issue that may arise due to spaces in between them  
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])

#we will make a new tag by adding, overview, genres, keywords, cast, crew
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

#creating a new dataframe with movie_id, title, tags
new_df = movies[['movie_id','title','tags']]
#converting the tags column from list to string 
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
#converting string to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
#steming is a technique to remove repeating words, use to remove words loving, loves, loved with love
#we will install natural learning library for it called nltk 
ps=PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


#we will convert tags into vector, which in turn will convert our movie into a vector, So if a person like one movie, then we can recommend more movies like it based on the closest vector
#we will use back of words technique to get this done
#In doing so, we will a make a string of all the tags most frequently used words, we will get 5000 such words.
#While make the vector for the first movie, we will add take the frequently used words in the first movies and tally their count in 5000 words which were taken from all the tags combined. this will make our first vector, we will do this repeatedly for every movie.
#we have to make sure to not include words like a, the, or, and etc
#for this we will use and external lib, 
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

#now we need to calculate distance between these arrays, the lesser the distance, more will be the similarity, between 2 different vector
#we will calculate the cosine distance between them, we will find the angle between them(because our vector has 4800+ dimensions 
#we will use sklearn.metrics.pairwise library to get the cosine_similarity 
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
#now we will make a function which will return 5 similar movies, based on input 
def recommend(movie):
    #we got the movie index by masking the movie title
    movie_index = new_df[new_df['title']==movie].index[0]
    #now will find similar movies, by finding similarity, then by doing sorting
    distance = similarity[movie_index]
    movie_list  = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
    return