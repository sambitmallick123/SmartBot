import logging
logging.getLogger('tensorflow').disabled = True

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#import tensorflow as tf


from gtts import gTTS
import speech_recognition as sr
import os
import re
import webbrowser
import smtplib
import requests
import bs4
from weather import Weather
#import pyttsx3
import random
import sys
import wikipedia
import json
import pprint
import datetime

from textblob import TextBlob
from tensorflow.python.compiler.tensorrt import trt_convert as trt


import tweepy
from textblob import TextBlob

#consumer key, consumer secret, access token, access secret.
# Step 1 - Authenticate
consumer_key= "9F9eMYUbCF4IHMY4pKpJCTwio"
consumer_secret= "Kb77J6szyBVhmeA3hwk7ubOSi1Acn54kXxh7V7O12K99ThP1xa"

access_token="90417751-3Jy9wwyiSc85lZkYRG0LNowQUTztrJ26HiNvidEW9"
access_token_secret="oQ7v9UcLmpmW1HDLLNaxCtc5xpqw23CY3FhcvsqFCRnOy"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


now = datetime.datetime.now()

def talkToMe(audio):

    print(audio)
    for line in audio.splitlines():
        os.system("say " + audio)

def weather_data(query):
    res=requests.get('http://api.openweathermap.org/data/2.5/weather?'+query+'&APPID=a946124530bfdabc290682b3c1131b47&units=metric');
    return res.json();
def print_weather(result,city):
    return ("{}'s temperature: {}°C ".format(city,result['main']['temp']))+"\n"+("Wind speed: {} m/s".format(result['wind']['speed']))+"\n"+("Description: {}".format(result['weather'][0]['description']))+"\n"+("Weather: {}".format(result['weather'][0]['main']))

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

words = []
labels = []
docs_x = []
docs_y = []
with open("intents.json") as file:
    data = json.load(file)

    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)


##Change 1
model = tflearn.DNN(net)

#try:
##Change3
#model.load("model.tflearn")
#except:
#    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#    model.save("model.tflearn")


##Change 2
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(msg):
    #print("Start talking with the bot (type quit to stop)!")
    #talkToMe('Hi, I am Jarvis Assistant')
    while True:
        #inp = input("You: ")
        inp=msg
        if "quit" in inp.lower():
            exit()
        elif "wiki" in inp.lower():
            command =inp.lower().replace('wiki','')
            results = wikipedia.summary(command, sentences=2)
            talkToMe('According to wikipedia ')
            rslt=results
            #continue
        elif 'open website' in inp.lower():
            reg_ex = re.search('open website (.+)', inp.lower())
            if reg_ex:
                domain = reg_ex.group(1)
                url = 'https://www.' + domain+'.com'
                webbrowser.open(url)
                rslt=('Done!')
                #continue
            else:
                pass
        elif 'what\'s up' in inp.lower():
            rslt=('Just doing my thing')
            #continue

        elif 'weather' in inp.lower():
            city=inp.lower().replace('weather in','')
            query='q='+city;
            w_data=weather_data(query);
            rslt=(print_weather(w_data, city))
            #continue
        elif 'movie' in inp.lower():
            talkToMe('Opening movietickets.com for you')
            url = 'https://www.movietickets.com'
            webbrowser.open(url)
            rslt='Done'
            #continue
        elif 'translate' in inp.lower():
            #print('Opening sebokwiki for your search query')
            command =inp.lower().replace('translate','')
            #talkToMe('Input Text in english : ')
            command =(TextBlob(command))
            rslt=str((command.translate(to= 'ar'))+'\n'+(command.translate(to= 'fr'))+'\n'+(command.translate(to= 'es'))+'\n'+(command.translate(to= 'hi'))+'\n'+(command.translate(to= 'de')))
            #rslt="Translated to above languages"
            #continue
        elif 'youtube' in inp.lower():
            #print('Opening youtube for your search query')
            talkToMe('Opening youtube for your search query')
            command =inp.lower().replace('youtube','')
            url = 'https://www.youtube.com/results?search_query=' + command
            webbrowser.open(url)
            rslt=('Done!')
            #continue
        elif 'sentiment' in inp.lower():
            #print('Opening youtube for your search query')
            #talkToMe('Enter the keyword : ')
            command =inp.lower().replace('find sentiment value for : ','')
            #talkToMe('Input Text in english : ')
            #command =TextBlob(command)
            #command =input()
            output=TextBlob(command)
            rslt=str(output.sentiment)


            #####Pending
        elif 'twitter' in inp.lower():
            #print('Opening youtube for your search query')
            #talkToMe('Enter the keyword : ')
            command =inp.lower().replace('twitter feed for ','')
            #talkToMe('Input Text in english : ')
            #command =TextBlob(command)
            #command =input()
            public_tweets = api.search(command)
            rslt1=[]
            #rslt=str(public_tweets.text.encode('unicode-escape').decode('UTF-8'))
            for tweet in public_tweets:
                rslt1=(tweet.text.encode('unicode-escape').decode('UTF-8'))

            rslt=str(rslt1)
                #analysis = TextBlob(tweet.text)
                #print(analysis.sentiment)
               
            #continue



        elif 'joke' in inp.lower():
            res = requests.get(
                'https://icanhazdadjoke.com/',
                headers={"Accept":"application/json"}
                )
            if res.status_code == requests.codes.ok:
                rslt = (str(res.json()['joke']))
                #continue
            else:
                rslt=('oops!I ran out of jokes')
                #continue   


                '''
        elif 'email' in inp.lower():
            talkToMe('Who is the recipient?')
            recipient = input('email ID : ')
            talkToMe('What should I say?')
            content = input('Content of the email : ')

            #init gmail SMTP
            mail = smtplib.SMTP('smtp.gmail.com', 587)
            mail.ehlo()
            mail.starttls()
            mail.login('sambitmallick123@gmail.com', 'Konveect1990!')
            #mail.sendmail('Sambit Mallick', 'sambitmallick123@gmail.com', content)
            mail.sendmail('Test Mail',recipient, content)
            mail.close()
            talkToMe('Email sent.')
            continue
            '''


        elif 'time' in inp.lower():
            print ("Current date and time : ")
            rslt = (now.strftime("%Y-%m-%d %H:%M:%S"))
            #continue
        else:
            results = model.predict([bag_of_words(inp, words)])
            #results = model.predict([bag_of_words(inp, words)])
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    #print(tag)
                    rslt=(random.choice(responses))


        return rslt


#chat()
#Creating GUI with tkinter
import tkinter
from tkinter import *
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You : " + msg + '\n\n')
        ChatLog.config(foreground="brown", font=("Verdana", 12 ))
        #res = chatbot_response(msg)
        res=chat(msg)
        ChatLog.insert(END, "SmartBot : " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
base = Tk()
base.title("SmartBot")
base.geometry("480x600")
base.resizable(width=FALSE, height=FALSE)
#Create Chat window
ChatLog = Text(base, bd=0, bg="grey", height="8", width="50", font="Arial",)
#ChatLog.config(state=ENABLED)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
EntryBox = Text(base, bd=0, bg="grey",width="10", height="5", font="Arial")
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,bd=0, bg="blue", activebackground="green",fg='grey',command= send )

#Create the box to enter message

#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=466,y=6, height=420)
#EntryBox.place(x=128, y=401, height=90, width=265)
EntryBox.place(x=6, y=470, height=90, width=320)
ChatLog.place(x=6,y=6, height=456, width=455)
SendButton.place(x=330, y=481, height=90)
#SendButton.place(x=6, y=401, height=90)
base.mainloop()
