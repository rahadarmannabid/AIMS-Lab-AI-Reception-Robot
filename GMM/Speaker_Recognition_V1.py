#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentenceTransformer
#import speech_recognition as sr
#from gtts import gTTS
import playsound
import os
import IPython
import pyaudio
import wave
import subprocess, sys
import uuid
import csv
from csv import writer
from datetime import date

from predict import predict

import MakeModel as mm
from GetFiles import GetFiles
import sys


# In[2]:


#csv row data append 
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


# In[3]:


def entry_database(name,son):
    test_name=name
    test_so_dau_name=son
    data=pd.read_csv("details.csv")
    row, col = data.shape
    img_name=row+1
    test_spk_name =str(img_name)
    test_spk_name =test_name+'.wav'
    directory_img = '/home/shehan/Speaker_Recognition/GMM/dataset/train/'+ test_name+ '/'+test_spk_name
    todays_date = date.today()
    #os.rename(r'database/test_image.jpg',test_image_name)
    row_contents = [img_name,directory_img,test_name,test_so_dau_name,todays_date]
    append_list_as_row('details.csv', row_contents)


# In[4]:


def train(speaker_name):
    '''
    :param
    speaker_name : name of the speaker whose model is to be prepared
                   Actually it takes the folder name of the speaker's audio files.
    '''
    print("Training "+ speaker_name+"'s model")
    gf = GetFiles(dataset_path="dataset") #getting the training files of speaker
    pandas_frame = gf.getTrainFiles(flag="train", train_speaker_folder=speaker_name) #audios path pipelined in dataframe
    mm.makeModel(pandas_frame)
    print("Training finished.")


# In[5]:


def predict_person():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 10
    user_id=str(uuid.uuid1())
    print(user_id)
    #directory=input()
    # Parent Directory path
    #parent_dir = "/home/shehan/Speaker-Recognition/GMM/dataset/test/"
  
    #path = os.path.join(parent_dir, directory)
    #print(path)

    #os.mkdir(path)

    WAVE_OUTPUT_FILENAME = "/home/shehan/Speaker_Recognition/GMM/dataset/predict/"+ user_id+".wav"
    print(WAVE_OUTPUT_FILENAME)

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
    prediction=predict(WAVE_OUTPUT_FILENAME)
    
    print(prediction)
    if prediction[1]>=-23:
        print("Hello "+prediction[0])
    else:
        print("The person isn't registered in database. Please give me your name for registration")
        name=input()
        print("Now please give me your child's name")
        son=input()
        entry_database(name,son)
        print("please give your voice for 20 second")
        
        RECORD_SECONDS_FOR_TRAINING = 20
        # Parent Directory path
        parent_dir = "/home/shehan/Speaker_Recognition/GMM/dataset/train/"
  
        path = os.path.join(parent_dir, name)
        #print(path)

        os.mkdir(path)

        WAVE_INPUT_FILENAME = path+ "/"+name+".wav"
        #print(WAVE_INPUT_FILENAME)

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS_FOR_TRAINING)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(WAVE_INPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close() 
        train(speaker_name=name)
    


# In[6]:


predict_person()


# In[ ]:




