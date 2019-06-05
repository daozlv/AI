# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:16:44 2018

@author:
"""

"""
KNN

(1) a description of how you formulated the problem:
    
     Using knn algorithm, we will find n high-dimensional point which is nearst to the point we want to classify.Then, we will find out the label of most point we picke out to be the predict label.

(2) a brief description of how your program works:
    
     This program including model_knn_train , model_knn_test and get_knn_train_result funtions. The other small funcions are called by them.

     The structure of this program is:
         
    	module_knn_train(train_file='train-data.txt',model_file='nearest_model.txt',k_range = 3,times_train = 1)
    		readTrainData(train_file)
    		generateTrainReport(imgTrainIds,imgTrainOrientations,imgTrainVectors,times_train,k_range)
    			trainKNN(trainIds,trainOrientations,trainVectors,k) 
    				testKNNAccurcy(partImgTrainIds,partImgTrainOrientations,partImgTrainVectors,partImgValIds,partImgValOrientations,partImgValVectors,k)
    					kNN_classify(imgTestVectors[i], imgTrainVectors, imgTrainOrientations, k)
    		save_train_result_to_excel.saveTrainResult()
    		generateTrainModelFile(model_file,bestK,imgTrainIds,imgTrainVectors,imgTrainOrientations)
            
    	model_knn_test(test_file='test-data.txt',model_file='nearest_model.txt')
    		readTrainModelFile(model_file)
    		readTrainData(test_file)
    		testKNNAccurcy(imgTrainIds,imgTrainOrientations,imgTrainVectors,imgTestIds,imgTestOrientations,imgTestVectors,bestK)
    			kNN_classify(imgTestVectors[i], imgTrainVectors, imgTrainOrientations, k)
    		generateTestOutputFile(outputFilename,predictSet) 
    		show_image_to_html.show_result_on_html(predictTrueSet,htmlTrueFile,predictFalseSet,htmlFalseFile)
            
    	get_knn_train_result()
    		module_knn_train(train_file ,model_file ,k_range ,ratio_train) 

(3) a discussion of any problems, assumptions, simplifications, and/or design decisions you made:
    
      We find that if the training set is large, our training will cost a lot of time.
      We design the function of 'get_knn_train_result()' to help us automaticly split the origin training set and get different training result and then generate the training result to excel file for analysis.
      Besides, we save the classfication result both right and wrong  in the html file to help us clearly find the patterns to the erros.
      
"""


import numpy as np 
import time
import save_train_result_to_excel 
import show_image_to_html
 



"""
@funciton:read data from dataset
@input:file address of dataset
@output:image ID,image orientation，image feature vector 
"""
def readTrainData(filename):  
    
    fr = open(filename)
    arrayLines = fr.readlines()
    numberLines = len(arrayLines)
    imgIds = []
    imgLabels = []
    imgDataSet = np.zeros((numberLines,192),dtype =int)
     
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split(' ')
        imgIds.append(listFromLine[0])
        imgLabels.append(listFromLine[1])
        imgDataSet[index,:] = listFromLine[2:]
        index = index + 1
        
    return imgIds,imgLabels,imgDataSet

""""
@funciton:kNN classfication
@input:taraget : feature vector of taget image:，dataset，label of dataset，k value
@output:result of classfication
"""
def kNN_classify(targetImg, dataSet, labels, k):
    
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(targetImg, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #sort the index
    sortedDistIndex = np.argsort(distances)     
    classCount={}          
    for i in range(k):
        voteLabel = labels[sortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key = lambda classCount:classCount[1], reverse=True)
    
    return sortedClassCount[0][0]


""""
@funciton:According to the training set and testing set,judge the accurcy in current k value
@input: trainset ，testset，k value
@output: accurcy
"""
def testKNNAccurcy(imgTrainIds,imgTrainOrientations,imgTrainVectors,imgTestIds,imgTestOrientations,imgTestVectors,k):
    predictTrueNum = 0
    predictFalseNum = 0
    
    predictSet=[]
    predictTrueSet = []
    predictFalseSet = []
    outputLines = []
    for i in range(len(imgTestIds)):
        guessOrientation = kNN_classify(imgTestVectors[i], imgTrainVectors, imgTrainOrientations, k)
        outputLine = [imgTestIds[i],guessOrientation]
        outputLines.append(outputLine)
        if guessOrientation == imgTestOrientations[i]:
            predictTrueNum = predictTrueNum + 1
            predictTrueSet.append([imgTestIds[i],imgTestOrientations[i],guessOrientation])
        else:
            predictFalseNum = predictFalseNum +1    
            predictFalseSet.append([imgTestIds[i],imgTestOrientations[i],guessOrientation])
        predictSet.append([imgTestIds[i],imgTestOrientations[i],guessOrientation])
    accurcy = predictTrueNum/len(imgTestIds)
    return accurcy,predictSet,predictTrueSet,predictFalseSet

"""
@function:According to the training set and k value, split the trainset to three set.One is validationset and two are training set.Then get the accuary 
@input:training set and k value
@output:分类准确度
"""
def trainKNN(imgTrainIds,imgTrainOrientations,imgTrainVectors,k):   

    
    #Split the trainset to three set.One is validationset and two are training set
    valDataSetSize = int(len(imgTrainIds)/3)
    accuracyAverage = 0
    for i in range(3):
        #image ID
        temp = imgTrainIds.copy()
        partImgValIds = temp[i*valDataSetSize:(i+1)*valDataSetSize]
        del temp[i*valDataSetSize:(i+1)*valDataSetSize]
        partImgTrainIds = temp
        #image orientation 
        temp = imgTrainOrientations.copy()
        partImgValOrientations = temp[i*valDataSetSize:(i+1)*valDataSetSize]
        del temp[i*valDataSetSize:(i+1)*valDataSetSize]
        partImgTrainOrientations = temp
        #image feature vector
        partImgValVectors = imgTrainVectors[i*valDataSetSize:(i+1)*valDataSetSize]    
        partImgTrainVectors = np.zeros((len(imgTrainVectors)-valDataSetSize,192),dtype =int)
        t = 0
        temp1 = partImgValVectors.tolist()
        temp2 = imgTrainVectors.tolist()
        for vector in temp2:
            if vector not in temp1:
                partImgTrainVectors[t,:] = vector
                t = t + 1
                
        accurcy,predictSet,predictTrueSet,predictFalseSet =  testKNNAccurcy(partImgTrainIds,partImgTrainOrientations,partImgTrainVectors,partImgValIds,partImgValOrientations,partImgValVectors,k)
        accuracyAverage = accuracyAverage + accurcy
    
    #average accuracy
    accuracyAverage = accuracyAverage/3  
    
    return accuracyAverage,predictTrueSet,predictFalseSet

"""
@function: Generate knn_model.txt,which including best k value and data from training set 
@input: best k value and data from training set 
@output: knn_model.txt
"""
def generateTrainModelFile(modelFilename,bestK,dataIds,dataSet, labels):
    file = open(modelFilename, 'w' )
    file.write(str(bestK))
    file.write('\n')
    for i in range(len(dataSet)):
        file.write(dataIds[i])
        file.write(' ')
        file.write(labels[i])
        file.write(' ')
        for element in dataSet[i]:
            file.write(str(element))
            file.write(' ')
        file.write('\n')
            
  
 
"""
@function:Generate training result from different k value and different training set 
@input:training set，number of different sizes of training set，range of k value
@output:result including: bestK,maxAccurcy,trainResult[splitTrainSetSize,k,accurcy,timeCost] 
"""
def generateTrainReport(imgTrainIds,imgTrainOrientations,imgTrainVectors,ratio_train,k_range): 
        
    accurcyDict ={}
    trainResult = []
 
    maxAccurcy = 0
    bestK = 1
    
    #according to the size of training set and number of different training set,generate training set which can be divided by 100 
    trainSetSize = int(np.floor(len(imgTrainIds)*ratio_train/100))*100
 
    #different size of training set 
    trainIds = imgTrainIds[0:trainSetSize]
    trainOrientations = imgTrainOrientations[0:trainSetSize] 
    trainVectors = imgTrainVectors[0:trainSetSize] 
    
    t0 = time.time()
    #get result
    for k in range(1,k_range+1):
        t1 = time.time()        
        accurcy,predictTrueSet,predictFalseSet = trainKNN(trainIds,trainOrientations,trainVectors,k) 
        t2 = time.time()
        timeCost = round(t2-t1,2)
        accurcy = round(accurcy,3)
        accurcyDict[k] = accurcy
        trainResult.append([trainSetSize,k,accurcy,timeCost])
        
        print('trainSetSize: %d , k: %d , accurcy: %.3f , time cost: %.2f\n'%(trainSetSize,k,accurcy,timeCost)) 
    
    sortedAccurcy = sorted(accurcyDict.items(), key = lambda accurcyDict:accurcyDict[1], reverse=True)  
    maxAccurcy  = sortedAccurcy[0][1]
    bestK = sortedAccurcy[0][0]
  
    timeCost = round(time.time()-t0)
 
    return trainSetSize,bestK,maxAccurcy,timeCost,trainResult
    

"""
@function:According to training set, get best k value.Then ouput the knn_model.txt 
@input: train_file='train-data.txt'
@output: model_file='nearest_model.txt'
"""    
def model_knn_train(train_file='train-data.txt',model_file='nearest_model.txt',k_range = 40,ratio_train = 0.5):
    #Analysis the training set
    imgTrainIds,imgTrainOrientations,imgTrainVectors = readTrainData(train_file)
    print('Read %s for testing successfully!!\n'%(train_file))
    
    #Generate the result in diffeent training set and different k value 
    trainSetSize,bestK,maxAccurcy,timeCost,trainResult = generateTrainReport(imgTrainIds,imgTrainOrientations,imgTrainVectors,ratio_train,k_range)
    print('In the case of k_range = %d, ratio_train = %.3f, training process has done!!'%(k_range,ratio_train))
    print('Result: best k value = %d, max accurcy = %.3f\n'%(bestK,maxAccurcy))

   
    #generate the parameter file
    generateTrainModelFile(model_file,bestK,imgTrainIds,imgTrainVectors,imgTrainOrientations)
    print('Generate %s in training successfully!!\n'%(model_file))

    return [trainSetSize,bestK,maxAccurcy,timeCost,trainResult]

 

"""
@funciton:read knn_model.txt
@input:model_file
@output:best k value, image ID ,image orientation,image feature vector 
"""
def readTrainModelFile(model_file):
    fr = open(model_file)
    arrayLines = fr.readlines()
    numberLines = len(arrayLines) -1 
    imgIds = []
    imgLabels = []
    imgDataSet = np.zeros((numberLines,192),dtype =int)
    bestK = 1
    
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split(' ')
        if len(listFromLine) == 1:
            bestK = int(listFromLine[0])
        else:
            imgIds.append(listFromLine[0])
            imgLabels.append(listFromLine[1])
            imgDataSet[index,:] = listFromLine[2:]
            index = index + 1
        
    return bestK,imgIds,imgLabels,imgDataSet

"""
@function:generate output.txt ，format： test/124567.jpg 180
@input:output_filename , predictSet[imgTestIds[i],imgTestOrientations[i],guessOrientation]
@output:output.txt
"""
def generateTestOutputFile(outputFilename,predictSet):
    file = open(outputFilename, 'w' )

    for predict in predictSet:
        file.write(predict[0])
        file.write(' ')
        file.write(predict[2])
        file.write('\n')
        

"""
@funciton:According to testing set and knn model file to generate the accuary  
@input:test_file='test-data.txt' , model_file='knn_model.txt'
@output: accurcy and the classfication result in the html file
"""
def model_knn_test(test_file='test-data.txt',model_file='nearest_model.txt'):
    
    #analysis the model file
    bestK,imgTrainIds,imgTrainOrientations,imgTrainVectors = readTrainModelFile(model_file)
    print('Read nearest_model.txt for testing successfully!!\n')
    
    #analysis the testing data
    imgTestIds,imgTestOrientations,imgTestVectors = readTrainData(test_file)
    print('Read test-data.txt for testing successfully!!\n')
    
    #get result using knn model
    accurcy,predictSet,predictTrueSet,predictFalseSet = testKNNAccurcy(imgTrainIds,imgTrainOrientations,imgTrainVectors,imgTestIds,imgTestOrientations,imgTestVectors,bestK)
    print('Testing have finished')
    print('Testing accurcy: %.3f\n'%(accurcy))
    
    #generate output.txt file 
    outputFilename = 'nearest_output.txt'
    generateTestOutputFile(outputFilename,predictSet) 
    print('Generate output.txt for testing successfully!!\n')
    
    
    #show the classfication result in the html file
    htmlTrueFile = 'knn_result_true.html'
    htmlFalseFile = 'knn_result_false.html'           
    show_image_to_html.show_result_on_html(predictTrueSet,htmlTrueFile,predictFalseSet,htmlFalseFile)
    print('Generate test_true_result.html and test_false_result.html for testing successfully!!\n')
    
"""
@funciton:In order to get training result under different parameters, we design this function to use  module_knn_train function 
@input: 
@output: results: ['trainingSetSize','bestK','maxAccuracy','timeCost'] in excel file
"""    
def get_knn_train_result():
    train_file='train-data.txt'
    model_file='nearest_model.txt'
    k_ranges= list(range(1,81,10))
    ratio_trains = list(range(2,12,4))
     
    
    titles = ['trainingSetSize','bestK','maxAccuracy','timeCost']
    trainResults = []
    for k_range in k_ranges:
        for ratio_train in ratio_trains:
            ratio_train = ratio_train/100
            print()
            print()
            print('start new training....\n')
            print('k_range',k_range,'ratio_train',ratio_train)
            print()
            result = model_knn_train(train_file ,model_file ,k_range ,ratio_train)       
            trainResults.append(result)
            
    trainResultsCopy = trainResults.copy()
    wantedResults =[]
    for i in range(len(trainResultsCopy)):
        wantedResult = trainResultsCopy[i][4]
        wantedResults.extend(wantedResult)
     
    trainResultFilename = 'nearest_train_result_training.xls'
    save_train_result_to_excel.saveTrainResult(trainResultFilename,titles,wantedResults)   