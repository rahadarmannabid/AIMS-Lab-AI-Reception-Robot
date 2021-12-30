#!/usr/bin/env python
# coding: utf-8

# In[7]:


#library for the chatbot system-3
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import IPython
import pyaudio
import wave
import subprocess, sys
from tkinter import *


# In[8]:


#encode text into numerical form
def sentence_encode(transcript_name):
    #model is used paraphase-mpnet-base-v2
    global model 
    #model= SentenceTransformer('bert-base-nli-mean-tokens')
    model= SentenceTransformer('paraphrase-mpnet-base-v2')
    #transcript_name is the reference database questions which are compared with users query
    global data
    data=pd.read_csv(transcript_name,header=None)
    a,b=data.shape
    #sentence is the variable where we save the encoded sententes from the database altogether 
    sentences=[]
    for i in range(0,a):
        corpus=data.iloc[i,b-2]
        sentences.append(corpus)
    #print(sentences)    
    sentence_embeddings = model.encode(sentences)
    #retuen the encoded sentence as a numpy variable 
    return sentence_embeddings


# In[9]:


# similarity check between database embaded sentence and users embedded query 
def get_input(sentence_embeddings,test):
    #encode users query into encoded from
    test_embeddings=model.encode(test)
    #checking the similarity 
    score=cosine_similarity(
    test_embeddings,
    sentence_embeddings)
    #s is the variable where we store all the scores from similarity checking
    s=score.tolist()[0]
    #best score 
    max_value = max(s)
    #best scores index number to find the corrusponing answer
    max_index= s.index(max_value)
    #print(max_index, max_value)
    answer_list=data.iloc[max_index,:].values
    #answer_str is the text format of the answer which computer will give
    answer_str = ''.join(answer_list)    
    print(answer_str.split("?")[1])
    #answer_text is convertated into the speech with googles gTTS model
    tts = gTTS(text=answer_str.split("?")[1], lang='bn')
    #saving the wav file of the gTTS
    tts.save("output.wav")


# In[10]:


def qus_ans_system():
    #recording procedure 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "input.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")
    

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    
    
    #test with recorded file; this record wav file must be PCM 16 bit integer, other than this format will show error
    recog= sr.Recognizer()
    #we need the users file to be saved in the backend because users voice data and noise will be checked for future research
    filename= "input.wav"
    #recognize the wav file as bangla
    with sr.AudioFile(filename) as source:
        audiofile=recog.listen(source)
        try:
            text=recog.recognize_google(audiofile, language='bn-BD')
            print(text)
            # google's recognition gives a string which we coverted into list format 
            test=[text]
            #database is transferd into sentence_encode function to encode
            sentence_embeddings=sentence_encode("transcript_domain.csv")
            #transfer the encoded database with users query into the get_input funtion where gTTS modules are availabe
            get_input(sentence_embeddings,test)
        except:
            print("check internet connectivity")
    #IPython.display.Audio("output.wav")
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, "output.wav"])


# In[ ]:


qus_ans_system()



