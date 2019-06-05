# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:29:39 2018

@author:
"""

"""
AdaBoost
(1) a description of how you formulated the problem:
     Using adaboost algorithm, we use the training set to generate several weak classifiers. In testing progress, we cascade weak classifiers to form a strong classifer. We add the predict result of weak classifiers to get a final predict result.

(2) a brief description of how your program works:
      This program including model_adaboost_train , model_adaboost_test and get_adaboost_train_result funtions. The other small funcions are called by them.

      The structure of this program is:
        	model_adaboost_train(train_file='train-data.txt',model_file='adaboost_model.txt',num_iteration=400,ratio_train = 1)
        		demo_forest.readFile(train_file,featNum,mode)
        		adaBoostTrainDS(dataArr,classLabels,num_iteration)
        			buildStump(dataArr,classLabels,D)
        				stumpClassify(dataMatrix,i,threshVal,inequal)
        		generate_adaboost_model(model_file,weakClassArr)
        		
        	model_adaboost_test(test_file='test-data.txt',model_file='adaboost_model.txt')
        		demo_forest.readFile(test_file,featNum,mode)
        		test_adaboost_accurcy(img_ID,img_test,labels_test,weakClassArr)
        		generate_adaboost_output_file(outputFilename,img_ID,predictList) 
        		show_image_to_html.show_result_on_html(predictTrueSet,htmlTrueFile,predictFalseSet,htmlFalseFile)
        		
        	get_adaboost_train_result()
        		model_adaboost_train(train_file,model_file,num_iteration,ratio_train)
        		save_train_result_to_excel.saveTrainResult(titles,trainResults)
	
(3) a discussion of any problems, assumptions, simplifications, and/or design decisions you made:

      We find the design of this wek classifier can only have two categories that is ture of false.
      But we have four categories need to classify. 
      In the training and testing progress, we set all labels to -1 if they aren't 0 degree. 
      So we get right format of labels. However, how can we get true testing result.? 
      We can only tell whether the orientation is 0 degree or not. 
      To solve this problem, we resize the feature vectors to square and rotate them for 90 degree or 180 degree or 270 degree. 
      Then we put the processed image to the classifier again to get the result. 
      If we get the predict label in the case of rotating 90 degree.That means this picture has rotated 90 degree before. 
      So we can get multiple classifications.
      
      
"""


import numpy as np
import demo_forest
import time
import show_image_to_html
import save_train_result_to_excel
   




"""
@funciton:According to the dimension and threshold value and the comparison direction,split the data into 1 or -1
@input:dataset and the dimension and threshold value and the comparison direction
@output: an array which value is 1 or -1
"""
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    
"""
@funciton:According to the dataset and labels and the wight vector , we get the best stump with minimal error
@input:the dataset and labels and the wight vector
@output: the best stump with minimal error
"""
def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

"""
@funciton:Core function of adaboost.In iteration,we have inital the weight vector,and update it to get different weak classifier.
@input:the dataset and labels and the number of iteration
@output: the weak classifier and the result  
""" 
def adaBoostTrainDS(dataArr,classLabels,numIt=400):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)   #init D to all equal
    aggClassEst = np.mat(np.zeros((m,1)))
    
    timeCost0 = time.time()
    for i in range(numIt):
        
        timeCost = time.time()
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
#        print("D:",D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
#        print("classEst: ",classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = np.multiply(D,np.exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
#        print("aggClassEst: ",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        
        timeCost = round(time.time() - timeCost,3)
        print("progress rate: %d/%d , total error: %.3f , timeCost: %.3fs"%(i,numIt,errorRate,timeCost))
        if errorRate == 0.0: break
    
    timeCost0 = time.time() - timeCost0
    return weakClassArr,aggClassEst,1-errorRate,timeCost0

"""
@funciton:Given the test data,get result of classification
@input: test data
@output: result of classification 
""" 
def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
 
    return aggClassEst
 
"""
@funciton:save adaboost model to text file
@input: the weak classifier and the file name
@output: model file
"""  
def generate_adaboost_model(model_file,weakClassArr):
    f=open(model_file,'w')
    for classifier in weakClassArr:
        f.writelines(str(classifier)+'\n')
    f.close()
    print('Adaboost classifier has been stored.')
    
"""
@funciton:the training function of adaboost model
@input: train_file, model_file, num_iteration, ratio_train
@output: list of [trainSetSize,num_iteration,errorRate,timeCost]
"""     
def model_adaboost_train(train_file='train-data.txt',model_file='adaboost_model.txt',num_iteration=300,ratio_train = 0.1):
    
    
    
    #get training set
    featNum = 64*3
    mode = 'rgb'         
    img,img_num,splitLabels,_ = demo_forest.readFile(train_file,featNum,mode)
     
    #according to ratio_train,get a part of training set
    trainSetSize = len(img)
    splitTrainSetSize = int(np.floor(trainSetSize*ratio_train/100)*100)
    split_img = img[0:splitTrainSetSize]
    print('trainSetSize',splitTrainSetSize)
      
    #analysis,get the training set and class label
    dataArr = []
    classLabels = []
    for example in split_img:
        dataArr.append(example[1:])
        if example[0] ==0:
            label = 1
        else:
            label = -1
        classLabels.append(label)
    
    #use adaboost to get a set of weak classifier
    weakClassArr,aggClassEst,errorRate,timeCost = adaBoostTrainDS(dataArr,classLabels,num_iteration)
    
    #generate model file
    generate_adaboost_model(model_file,weakClassArr)
 
    
    
#    return ['trainingSetSize':splitTrainSetSize,'num_iteration':num_iteration,'accuracy':1-errorRate,'timeCost':timeCost}
    return [splitTrainSetSize,num_iteration,errorRate,timeCost]

"""
@funciton:get adaboost model from text file
@input: model_file 
@output: the weak classifier of adaboost model
"""  
def read_adaboost_model(model_file):  
    
    weakClassArr=[]    
    f=open(model_file)
    lines=f.readlines()  
    for line in lines:
        weakClassArr.append(eval(line))
    f.close()
    
    return weakClassArr

"""
@funciton:generate adaboost outputfile
@input: outputFilename,img_ID,predictList
@output: ouput.txt
""" 
def generate_adaboost_output_file(outputFilename,img_ID,predictList):      
    outputFile=open(outputFilename,'w')   
    for i in range(len(img_ID)):
        outputFile.write(img_ID[i])
        outputFile.write(' ')
        outputFile.write(str(predictList[i]))
        outputFile.write('\n')
    print('Output file has been stored.')    
    
 
"""
@funciton: According to the adaboost model and testset to get test result
@input: img_ID,img_test,labels_test,weakClassArr
@output: accuracy,predictTrueSet,predictFalseSet,predictList
""" 
def test_adaboost_accurcy(img_ID,img_test,labels_test,weakClassArr):
    index = 0
    right_cnt = 0
    predictList=[] 
    predictTrueSet = []
    predictFalseSet = []
    
    for data in img_test:
        r= []
        g= []
        b= []
        for i in range(len(data)):
            if i%3 == 0:
                r.append(data[i])
            elif i%3 == 1:
                g.append(data[i])
            elif i%3 == 2:
                b.append(data[i])
        r = np.reshape(np.array(r),(8,8))
        g = np.reshape(np.array(g),(8,8))
        b = np.reshape(np.array(b),(8,8))
        
        rotation_predicts = []
        for i in range(0,4):        
            rot_r_line = np.reshape(np.rot90(r,i),(1,64))
            rot_g_line = np.reshape(np.rot90(g,i),(1,64))
            rot_b_line = np.reshape(np.rot90(b,i),(1,64))         
            data_test =[]
            for j in range(len(rot_g_line[0])):
                data_test.extend([rot_r_line[0][j],rot_g_line[0][j],rot_b_line[0][j]])
            rotation_predict = adaClassify(data_test,weakClassArr)
            rotation_predicts.append(rotation_predict)
        predict = rotation_predicts.index(max(rotation_predicts)) *90
        predictList.append(predict)
#        print('predict',predict)
#        print('true rotation:',labels_test[index])
        if(predict == labels_test[index]):
            right_cnt = right_cnt + 1
            predictTrueSet.append([img_ID[index],str(labels_test[index]),str(predict)])
        else:
            predictFalseSet.append([img_ID[index],str(labels_test[index]),str(predict)])
        index = index + 1
        
    accuracy = right_cnt/len(img_test)
    print('The accuracy is ',accuracy)
    
    return accuracy,predictTrueSet,predictFalseSet,predictList

"""
@funciton:the testing function of adaboost model
@input: train_file, model_file 
@output: testing result including output_adaboost.txt and result_adadboost_ture.html and result_adadboost_false.html file
"""   
def model_adaboost_test(test_file='test-data.txt',model_file='adaboost_model.txt'):
        
    #get testing set from file
    featNum = 64*3
    mode = 'rgb'
    dataArr_test,img_num_test,labels_test,img_ID = demo_forest.readFile(test_file,featNum,mode)
    print('Read testing dataset successfully!!\n')
    
    #analysis,get the testing set and class label
    img_test = []
    labels_test = []
    for example in dataArr_test:
        img_test.append(example[1:])
        labels_test.append(example[0])
        
    #read the model file
    weakClassArr = read_adaboost_model(model_file)   
    print('Read adaboost-model.txt for testing successfully!!\n')
    
    #use model and tesing set to get the testing result
    print('Using adaboost model, please wait...')
    accuracy,predictTrueSet,predictFalseSet,predictList = test_adaboost_accurcy(img_ID,img_test,labels_test,weakClassArr)
     
     
    #generate output.txt file 
    outputFilename = 'adaoost_output.txt'
    generate_adaboost_output_file(outputFilename,img_ID,predictList) 
    print('Generate output_adaboost.txt for testing successfully!!\n')
  
    
    #show the classfication result in the html file
    htmlTrueFile = 'adaboost_result_true.html'
    htmlFalseFile = 'adaboost_result_false.html'           
    show_image_to_html.show_result_on_html(predictTrueSet,htmlTrueFile,predictFalseSet,htmlFalseFile)
    print('Generate test_true_result.html and test_false_result.html for testing successfully!!\n')
     
    return accuracy
 
"""
@funciton:In order to get training result under different parameters, we design this function to use  module_knn_train function 
@input: 
@output: results: ['trainingSetSize','num_iteration','accuracy','timeCost'] in excel file
"""
def get_adaboost_train_result():
    train_file='train-data.txt'
    model_file='adaboost_model.txt'
    test_file='test-data.txt'
    num_iterations= list([50])
    ratio_trains = list(range(1,10,1))
     
    
    titles = ['trainingSetSize','num_iteration','accuracy','timeCost']
    trainResults = []
    for num_iteration in num_iterations:
        for ratio_train in ratio_trains:
            ratio_train = ratio_train/10
            print('start new training....')
            print('num_iteration',num_iteration,'ratio_train',ratio_train)
            result = model_adaboost_train(train_file,model_file,num_iteration,ratio_train)
            accuracy = model_adaboost_test(test_file,model_file) 
            result[2] = accuracy
            trainResults.append(result)   
    
          
    trainResultFilename = 'adaboost_train_result_training_size2.xls'
    save_train_result_to_excel.saveTrainResult(trainResultFilename,titles,trainResults)