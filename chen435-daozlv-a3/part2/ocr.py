# -*- coding: utf-8 -*-
#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2018)
#
'''
In order to facilitate debugging, the method of calling main function is changed. The path of train_img_fname,
train_txt_fname and test_img_fname files are given directly.You can run this py file directly for debugging.


There are 26 uppercase Latin characters, 26 lowercase characters, 10 numbers, spaces and 7 punctuation marks, (), -! "?" So set the hidden state of HMM to N = 69.
In terms of emission probability, we use a simple naive Bayesian classifier, so B is a unit matrix. The number of black
dots is calculated after each letter image is segmented to represent the observed value, which also has M = 69 observed values.
Read the text data in Part1 as text. txt. Calculate the parameters of pi, A and B, and establish the HMM model.

After the completion of the initial modeling, the effect is not satisfactory. The main reasons are as follows: 1.
The improper selection of features leads to one observation value corresponding to several states. 2. The model needs to be optimized.

Improvement: Change the way of calculating the eigenvalues, and take the points on the crossing of the letters as the eigenvalues.
The effect is worse after modeling.

Finally, the sum of points and all 0 points on the crossing is used as the feature to model, eliminating the situation that one
feature corresponds to several hidden states. Because of the interference of white points, the final test results are complemented
to the ideal. The next step is to prepare for image filtering (denoising direction is enhanced).


test-2-0.png accuracy result：
simple: c?!' -?-??!' 3-5-cc 3!-?? !!. !?-? - ?c-?ccc ?-!c !!. !?-?
viterbi: IX"X Z"""""X 375""X 30000 196 1960 ( ?"""""X Z""""""X Z"""
Final answer: Nos. 14-556. Argued April 28, 2015 - Decided June 26, 2015

'''



from PIL import Image, ImageDraw, ImageFont
import sys
import pickle
import numpy as np


CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


pi = {}

A = {}

B = {}
ww = []
pos = []
fre ={}
dp = []
pre = []
zz = {}
cx={}
cy={}





def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = [w for w in line.split(' ')]
        exemplars += tuple([ ' '.join(data[0::2]), ])
    return exemplars



def load_letters(fname):
    im = Image.open(fname)
    
    px = im.load()
    print px
    (x_size, y_size) = im.size
    print im.size
    print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        #result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
        px0=0
        for x in range(x_beg, x_beg+CHARACTER_WIDTH) :
            
            for y in range(0, CHARACTER_HEIGHT):
                if px[x, y] ==0:
                    px0+=1
        for x in range(x_beg, x_beg+CHARACTER_WIDTH) :
            if px[x, 13] ==0:
                    px0+=1
        for y in range(0, CHARACTER_HEIGHT):
            if px[x_beg+7, y] ==0:
                    px0+=1
        result.append(px0)
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' " #
    letter_images = load_letters(fname)
    #print letter_images
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def train_icf(data):
    #print pos,ww
    n = len(data)    
    #init
    for i in pos:
        pi[i]=0
        fre[i]=0
        A[i]={}
        B[i]={}
        for j in pos:
            A[i][j]=0
        for j in ww:
            B[i][j]=0    
    
    #cla
    for i in range(n):
        
        data[i].replace('``','"')
        data[i].replace('`',"'")
        if pi.has_key(data[i][0]):
            pi[data[i][0]]+=1   #
        for j in range(len(data[i])):
            if fre.has_key(data[i][j]) and fre.has_key(data[i][j-1]) :
                fre[data[i][j]]+=1
                if j>0:            
                    A[data[i][j-1]][data[i][j]]+=1
                B[data[i][j]][train_letters[data[i][j]]]+=1

    n = len(data)
    for i in pos:
        cx[i]=0
        cy[i]=0
        
        for j in pos:
            if(A[i][j]==0):
                cx[i]+=1
                A[i][j]=0.5
        for j in ww:
            if(B[i][j]==0):
                cy[i]+=1
                B[i][j]=0.5
    for i in pos:
        pi[i]=pi[i]*1.0/n
        for j in pos:
            A[i][j]=A[i][j]*1.0/(fre[i]+cx[i])
        for j in ww:
            B[i][j]=B[i][j]*1.0/(fre[i]+cy[i])
    #print B['s']
    f_pi = open("pi.txt", "wb")
    pickle.dump(pi,f_pi)
    f_pi.close()
    f_A = open("A.txt", "wb")
    pickle.dump(A,f_A)
    f_A.close()
    f_B = open("B.txt", "wb")
    pickle.dump(B,f_B)
    f_B.close()

def test_icf(sentence):
    num=len(sentence)
    dp=[{} for i in range(0,num)]
    pre=[{} for i in range(0,num)]
    for k in pos:
        for j in range(0,num):
            dp[j][k]=0
            pre[j][k]=""
    n=len(pos)
    for c in pos:
        if(B[c].has_key(sentence[0])):
            dp[0][c]=pi[c]*B[c][sentence[0]]*1000
        else:
            dp[0][c]=pi[c]*0.5*1000/(cy[c]+fre[c])
    for i in range(1,num):
        for j in pos:
            tt=0
            if(B[j].has_key(sentence[i])):
                tt=B[j][sentence[i]]*1000
            else:
                tt=0.5*1000/(cy[j]+fre[j])
            for k in pos:                    
                if(dp[i][j]<dp[i-1][k]*A[k][j]*tt):
                    dp[i][j]=dp[i-1][k]*A[k][j]*tt
                    pre[i][j]=k
    res=[0 for t in range(num)]
    MAX=""
    for j in pos:
        if(MAX=="" or dp[num-1][j]>dp[num-1][MAX]):
            MAX=j
##        if(dp[num-1][MAX]==0):
##            print "
##            continue
    i=num-1
    while(i>=0):
        res[i]=MAX
        MAX=pre[i][MAX]
        i-=1
    
    res=''.join(res)
    return res

#####
# main program

#(train_img_fname, train_txt_fname, test_img_fname) = ('courier-train.png','bc.train','test-2-0.png')  #'test-0-0.png'


(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)  #
print train_letters
train_data = read_data(train_txt_fname)   #
#print train_data[0]
for chas in train_letters:
    pos.append(chas)
    ww.append(train_letters[chas])
#print pos,ww
train_icf(train_data)



test_letters = load_letters(test_img_fname)
#print test_letters
sre=[]
for dd in test_letters:
    sre.append(pos[np.argmin(abs(dd-np.array(ww)))])
print 'simple:',''.join(sre)
testchar=test_icf(test_letters)
print 'viterbi:',testchar
file = open('test-strings.txt', 'r');
testxt=[]
for lin in file:
    testxt+=[lin,]
try:
    print 'Final answer:', testxt[int(test_img_fname.split('-')[1])]
except:
    pass
## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:

#print "\n".join([ r for r in train_letters['a'] ])

# Same with test letters. Here's what the third letter of the test data
#  looks like:

#print "\n".join([ r for r in test_letters[2] ])



