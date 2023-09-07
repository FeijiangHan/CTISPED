#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:06:16 2019

@author: agapi

Splits the data into training and validation sets. 
To run, paste the path for the data directory in the last line. 

e.g. 

If you want to run through bash, uncomment the last line, navigate to the folder the script is located 
and run the bash command with the data directory as argument. 
e.g. python3 splitTrainAndValid.py /home/agapi/darknet/data/img

CAUTION! Use only full paths, the code cannot load the images with relative paths.

"""



import random
import os
import subprocess
import sys
import random

def split_data_set(image_dir):

    f_val = open("valid.txt", 'w')
    f_train = open("train.txt", 'w')
    
    path, dirs, files = next(os.walk(image_dir))
    data_size = len(files)

    ind = 0
    data_valid_size = int(0.2 * data_size) #The 80% of the training data is for training and the 10% for validation.
    valid_array = random.sample(range(data_size), k=data_valid_size)
    
    for f in os.listdir(image_dir):
        if(f.split(".")[1] == "tiff"):
            ind += 1
            
            if ind in valid_array:
                f_val.write(image_dir+'/'+f+'\n')
            else:
                f_train.write(image_dir+'/'+f+'\n')
                
                
    #Shuffle the rows in the newly created .txts in order for the algorithm to train on mixed patients.
    with open('train.txt','r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open('train.txt','w') as target:
        for _, line in data:
            target.write( line )               
 

    with open('valid.txt','r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open('valid.txt','w') as target:
        for _, line in data:
            target.write( line )                          

split_data_set('/home/agapi/Desktop/12_37638_NIH&Decathlon_416_YOLO3_-200_300_20val_8000epochs/darknet/data/img')
#split_data_set(sys.argv[1])
#split_data_set(sys.argv[1])