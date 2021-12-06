# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:50:27 2021

@author: ziyad
"""
def recommend():        
    #string = input("Me: ") 
    ip=textF.get()
    msgs.insert(END,"You: " + ip)
    textF.delete(0,END)
    
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize 
    import math
    import re
    from collections import Counter
    
    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
        
    def text_to_vector(text):
        word = re.compile(r'\w+')    #breaks a line into words
        words = word.findall(text)
        return Counter(words) 
    
    for i in range(0,10866):
      
        X =dataset['Bag_of_words'][i]
        Y1 =ip
     
        # lowercase and tokenization(convert into list so that stopwords are removed easily)
        X_list = word_tokenize(X.lower())  
        Y_list = word_tokenize(Y1.lower()) 
    
        # remove stop words from the string 
        X_set = {w for w in X_list if not w in stopwords.words('english')}  
        Y_set = {w for w in Y_list if not w in stopwords.words('english')} 
        
        # get back to original form
        X=' '.join(X_set)
        Y=' '.join(Y_set)
        
        vector1 = text_to_vector(X)
        vector2 = text_to_vector(Y)
          
        dataset['cosine_sim'][i]=get_cosine(vector1,vector2)
    
    dataset1= dataset[['original_title','Bag_of_words','cosine_sim']]
    
    dataset1.sort_values(by=['cosine_sim'],inplace=True,ascending=False)     
    dataset1 = dataset1.drop(dataset1.columns[[1]] ,axis=1)
    dataset1 = dataset1.reset_index(drop=True) 
    msgs.insert(END,"Bot: " + dataset1['original_title'][0] +", "+ dataset1['original_title'][1] +", "+ dataset1['original_title'][2])
   
   
import pandas as pd
import nltk
from rake_nltk import Rake


data = pd.read_csv('D:\\Study Materials\\B.Tech\\7th Sem\\Project\\Major Project - Chatbot\Dataset New\\tmdb_movies_data.csv')
data = data.drop(data.columns[[0,1,3,4,9,16,19,20]] ,axis=1)
pd.options.mode.chained_assignment = None  # default='warn'

#fill up the null values
data ['cast']=data ['cast'].fillna("not available")
data['genres']=data['genres'].fillna("not available")
data['director']=data['director'].fillna("not available")
data['keywords']=data['keywords'].fillna("not available")
data['production_companies']=data['production_companies'].fillna("not available")
data ['overview']=data ['overview'].fillna("not available")
data ['homepage']=data ['homepage'].fillna("not available")
data ['release_year']=data ['release_year'].fillna("not available")

for i in range(0,10866):
    s=str(data['release_year'][i])
    data['release_year'][i]=s+'|'  

#split the multiple values separated by '|'    
data['genres'] = data['genres'].map(lambda x: x.split('|'))
data['cast'] = data['cast'].map(lambda x: x.split('|'))
data['director'] = data['director'].map(lambda x: x.split('|'))
data['keywords'] = data['keywords'].map(lambda x: x.split('|'))
data['production_companies'] = data['production_companies'].map(lambda x: x.split('|'))
data['release_year'] = data['release_year'].map(lambda x: x.split('|'))

#lowercase the data 
r = Rake()
for index, row in data.iterrows():
    data['cast'][index] = [x.lower() for x in row['cast']]
    data['genres'][index] = [x.lower() for x in row['genres']]
    data['director'][index] = [x.lower() for x in row['director']]
    data['keywords'][index] = [x.lower() for x in row['keywords']]
    data['production_companies'][index] = [x.lower() for x in row['production_companies']]
    r.extract_keywords_from_text(row['overview'])
    data['overview'][index] = list(r.get_word_degrees().keys())

#create a bag of words model 
data['Bag_of_words'] = ''
data['cosine_sim']=''
columns = ['genres', 'director', 'cast', 'keywords','production_companies', 'release_year']
for index, row in data.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    data['Bag_of_words'][index] = words
    data['cosine_sim'][index]=0
    
dataset= data[['original_title','Bag_of_words','cosine_sim']]

    
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import os
import spacy
nlp = spacy.load('en_core_web_sm')
from tkinter import Tk,Label,Frame,Scrollbar,Listbox,Button,Entry,BOTH,END,LEFT,RIGHT,X,Y
from PIL import ImageTk,Image


my_bot = ChatBot(name='MRBot', read_only= True, logic_adapters=['chatterbot.logic.BestMatch','chatterbot.logic.MathematicalEvaluation'])

File = os.listdir('D:\\Study Materials\\B.Tech\\7th Sem\\Project\\Major Project - Chatbot\\chatterbot-corpus-master\\chatterbot-corpus-master\\chatterbot_corpus\\data\\english\\')

list_trainer = ListTrainer(my_bot)

for i in File:
    chatbot_data = open('D:\\Study Materials\\B.Tech\\7th Sem\\Project\\Major Project - Chatbot\\chatterbot-corpus-master\\chatterbot-corpus-master\\chatterbot_corpus\\data\\english\\'+i,'r', encoding= 'utf-8').readlines()
    list_trainer.train(chatbot_data)
 
#my_bot.storage.drop()

main = Tk()
main.geometry("550x700")
main.title("MovieBot")
img = ImageTk.PhotoImage(Image.open("D:\\Study Materials\\B.Tech\\7th Sem\\Project\\Major Project - Chatbot\\Presentation\\2. Even Sem\\1607171615024-movie.jpg\\"))
photoL = Label(image=img)
photoL.pack(pady=5)


def ask():
    ip=textF.get()
    reply=my_bot.get_response(ip)
    msgs.insert(END,"You: " + ip)
    if (ip.strip()=='bye' or ip.strip()=='Bye'):
        msgs.insert(END,"Bot: Bye. See you again")
        textF.delete(0,END)
        return
    msgs.insert(END,"Bot: " + str(reply))
    textF.delete(0,END)
    
    
frame=Frame(main)
sc= Scrollbar(frame)
msgs=Listbox(frame,width=100,height=20)
sc.pack(side=RIGHT,fill=Y)
msgs.pack(side=LEFT,fill=BOTH,pady=10)
frame.pack()

textF=Entry(main,font=("Tahoma",14))
textF.pack(fill=X, pady=10)

btn=Button(main,text="Send",font=("Tahoma",16),command=ask)
btn.pack(side=LEFT,padx=50)
btn2=Button(main,text="Ask Movie",font=("Tahoma",16),command=recommend)
btn2.pack(side=RIGHT,padx=50)

main.mainloop()
