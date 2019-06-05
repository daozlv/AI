# -*- coding: utf-8 -*-
###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math
import pickle
import numpy as np
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

'''
1.In HMM, P (S1) is the parameter pi to indicate the probability of the first state being optional in the hidden state sequence,
P (Si+1 | Si) is the transition probability of the hidden state sequence, P (Wi | Si) is the observation probability corresponding to
the state, and P (Wi | Si) is the obfuscation matrix B. According to the theorem of large numbers,
The values of pi, A and B can be calculated according to the order of parts of speech and the corresponding words in the training data.
After reading the training data bc. train, the probability of the first part of speech in each sentence is pi, and the probability of
the 12*12 part of speech combination is A.
The probability of 12* word number combination is B, and the first step is completed.
2.Simple model is a simple clustering model. It only needs to count the words corresponding to each part
of speech in the training data.
3.Comple_mcmc is a Hidden Markov Chain, and the formula is not well understood, so we need to check the
relevant algorithms and data.After looking at the data, we found that Comple_mcmc and HMM are only different from A matrix, so we need to
recalculate A matrix and use MCMC to calculate and forecast.

The functions of this document include training and prediction. 1. After reading the training data, the pi,
A and B are calculated to indicate the completion of the training. The training parameters are saved by pickle
after the training is completed. When testing, there is no need to repeat the training, and the saved parameters
are read directly for testing.
2. In the test, the most probable sequence is predicted according to the incoming observations, pi, A, B values
and Viterbi algorithm.

In the construction of hmm, the first idea is to import the HMM integrated library for training.
Unit to better understand the HMM algorithm, and finally decided to find information, write the
whole process of HMM algorithm.

'''
#argv init
simplepre={}
pi = {}
A = {}
B = {}    #HMM argv

cA = {}   #mcmc A
cB = {}
ww = []
pos = [ 'adj' , 'adv', 'adp', 'conj','det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
fre ={}
dp = []
pre = []
zz = {}
cx={}
cy={}

##f_pi = open("pi.txt", "rb")
##pi=pickle.load(f_pi)
##f_pi.close()
##f_A = open("A.txt", "rb")
##A=pickle.load(f_A)
##f_A.close()
##f_B = open("B.txt", "rb")
##B=pickle.load(f_B)
##f_B.close()
##f_simple = open("simp.txt", "rb")
##simplepre=pickle.load(f_simple)
##f_simple.close()
##print pi


def list_write(f, v):
    for a in v:
        f.write(str(a))
        f.write(' ')
    f.write('\n')


class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        
        elif model == "Complex":
            return -999
        
        elif model == "HMM":

            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    
    def train(self, data):

        ########################################################
        
        
        ################################################################
        for i in pos:
            simplepre[i]=[]
        
        #tongji
        n = len(data)
        for i in range(0, n):
            word = data[i][0]          
            for pp in range(len(word)):
                if data[i][0][pp] not in simplepre[data[i][1][pp]]:
                    simplepre[data[i][1][pp]].append(data[i][0][pp])
            for danci in word: 
                if (danci not in ww):
                    ww.append(danci) 
        
        #init
        for i in pos:
            pi[i]=0
            fre[i]=0
            A[i]={}
            B[i]={}
            cA[i]={} #mcmc argv
            for j in pos:
                A[i][j]=0
                cA[i][j]={}
                for z in pos:
                    cA[i][j][z]=0
            for j in ww:
                B[i][j]=0
       
        
        #cla
        n = len(data)
        for i in range(n):
            
            pi[data[i][1][0]]+=1   
            for j in range(len(data[i][1])):
                fre[data[i][1][j]]+=1
                if j>0:            
                    A[data[i][1][j-1]][data[i][1][j]]+=1  # HMM A 
                if j>1:
                    cA[data[i][1][j-2]][data[i][1][j-1]][data[i][1][j]]+=1   #mcmc A
                B[data[i][1][j]][data[i][0][j]]+=1


 

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
                for z in pos:
                    cA[i][j][z]=cA[i][j][z]*1.0/fre[z]
            for j in ww:
                B[i][j]=B[i][j]*1.0/(fre[i]+cy[i])

        f_pi = open("pi.txt", "wb")
        pickle.dump(pi,f_pi)
        f_pi.close()
        f_A = open("A.txt", "wb")
        pickle.dump(A,f_A)
        f_A.close()
        f_B = open("B.txt", "wb")
        pickle.dump(B,f_B)
        f_B.close()
##        f_simple = open("simp.txt", "wb")
##        pickle.dump(simplepre,f_simple)
##        f_simple.close()



        pass
###############################################################

        

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):

        res=[]
        for danci in sentence:
            tu=0
            #print danci
            for i in pos:
                if danci in simplepre[i]:
                    res.append(i)
                    tu=1
                    break
            if tu==0:                  
                res.append(i)
        #print res      
        return res#[ "noun" ] * len(sentence)

    def complex_mcmc(self, sentence):
        '''The MCMC implementation algorithm still needs to be further studied,
        and has not yet been implemented.
        Figure C shows not a Markov chain, by looking for MCMCMC data, or do not know how to use MCMC to achieve.
        Then an algorithm is implemented according to my own understanding.

        '''
        num=len(sentence)
        xa = np.array([[1.0 / num for x in range(num)] for y in range(num)])
        x0=random.randint(0,11)
        count = 0
        samplecount = 0
        cx=''
        cxpre=0
        
        res=[0 for t in range(num)]
        for c in pos:
            if(B[c].has_key(sentence[1])):
                if pi[c]*B[c][sentence[1]]>cxpre:
                            cxpre= pi[c]*B[c][sentence[1]]
                            cx=c
        if cxpre!=0:
            res[1]=cx
            cxpre=0
            for c in pos:
                if B[c][res[1]]*pi[c]>cxpre:
                    cxpre=B[c][res[1]]*pi[c]
                    cx=c
            if cxpre!=0:
                res[0]=cx
        else:
            for c in pos:
                if(B[c].has_key(sentence[0])):
                    if pi[c]*B[c][sentence[0]]>cxpre:
                                cxpre= B[c][sentence[0]]
                                cx=c
                if pi[c]>cxpre:
                    cxpre=pi[c]
                    cx=c
            res[0]=cx
            cxpre=0
            for c in pos:
                if B[res[0]][c]>cxpre:
                    cxpre=B[res[0]][c]
                    cx=c
            res[1]=cx
                
        cx=[]
        cxpre=[]
        for i in range(2,num):
            for c in range(len(pos)):
                if(B[pos[c]].has_key(sentence[i])):
                    cx.append(c)
                    cxpre.append(B[pos[c]][sentence[i]])
                if cA[res[0]][res[1]].has_key(pos[c]):
                    cx.append(c)
                    cxpre.append(cA[res[0]][res[1]][pos[c]])
            res[i]=pos[cx[cxpre.index(max(cxpre))]]
            
            

##        for i in range(num):
##            res[i]=pos[random.randint(0,11)]
        return res

        #return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        num=len(sentence)
        dp=[{} for i in range(0,num)]
        pre=[{} for i in range(0,num)]
        #初始化概率
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
##            print 
##            continue
        i=num-1
        while(i>=0):
            res[i]=MAX
            MAX=pre[i][MAX]
            i-=1
        #print res
        return res


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

