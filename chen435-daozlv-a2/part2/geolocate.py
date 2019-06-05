import sys
import numpy as np
from collections import Counter
import math

stopwords = set(open("stopwords.txt","rb").read().decode(errors="ignore").split())

# creat the word vocab
def createVocabList(dataSet,max_words=10000):
    word_count = Counter()
    for sentence in dataSet:
        for w in sentence:
            #if not w in stopwords:
                word_count[w] += 1
    ls = word_count.most_common(max_words)
    word_dict = {w[0]: index for (index, w) in enumerate(ls)}
    return word_dict,[w[0] for w in ls]

#bag of words
def bagOfWords2VecMN(vocab,inputSet):
    returnVec=[0]*len(vocab)
    for word in set(inputSet):
        if word in vocab:
            returnVec[vocab[word]]=1
    return returnVec #vector

# Turn the sentence to vector
def _vectorize(vocab,data):
	vectors = []
	for sentence in data:
		vectors.append(bagOfWords2VecMN(vocab,sentence))
	return np.array(vectors)

def trainNB(trainMatrix, trainCategory):
    """
    we replace prob by log(prob), Multiplication   ==>   additions
    """

    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    labels = set(trainCategory)

    label_prob = {}
    for label in labels:
        label_prob[label] = math.log(trainCategory.count(label)/float(numTrainDocs))


    each_words_num = {}
    all_words_num = {}
    for label in labels:
        each_words_num[label] = np.zeros((numWords))
        all_words_num[label] = 0 
    # calc the word frequency
    for i in range(numTrainDocs):
        each_words_num[trainCategory[i]] += trainMatrix[i]
        all_words_num[trainCategory[i]] += sum(trainMatrix[i])
    #calc the probability
    label_words_prob={}
    for label in labels:
        # add one smooth
        label_words_prob[label] = np.log((each_words_num[label]+1) / (all_words_num[label]+numWords))
    
    return label_words_prob,label_prob


def testNB(testMatrix,label_words_prob,label_prob):
    
    p = {}
    testnum = len(testMatrix)
    test_result = []

    for i in range(testnum):
    	for label in label_prob.keys():
    		p[label] = sum(testMatrix[i]*label_words_prob[label])+label_prob[label]
    	L = sorted(p.items(), key = lambda d:d[1] , reverse=True)
    	test_result.append(L[0][0])
    return test_result

def load_data(path):
	"""
	load data. (train data and test data)
	"""
	sentences = []
	labels = []
	train_data = open(path,"rb").read().decode(errors="ignore").split("\n")
	for line in train_data:
		line = line.strip()
#		print (line)
		if line=="":
			continue
		else:
			try:
				line = line.split(",")
				location = line[0]
				sentence = ",".join(line[1:])
				labels.append(location)
				sentences.append(sentence.split())
			except:
				exit()
	return sentences,labels

def most_Relevant_words(label_words_prob,label_prob,trainMatrix,vocab):
	"""
	calc the most relevant words with each location
	"""
	pw = np.log(np.sum(trainMatrix,0)/np.float(np.sum(trainMatrix)))
	tops_words = {}
	for label in label_prob.keys():
		p_l_w = label_words_prob[label]+label_prob[label]-pw
		sorted_index = sorted(range(len(p_l_w)), key=lambda x: p_l_w[x],reverse=True)[:5]
		tops_words[label] = [vocab[index] for index in sorted_index]

	return tops_words


def main(train_file_path,test_file_path,out_path):

	print ("Load Data")

	train_data,train_label = load_data(train_file_path)
	test_data,test_label = load_data(test_file_path)

	print ("Creat Vocab")

	word_dict,vocab = createVocabList(train_data)

	print ("process data")

	trainMatrix  = _vectorize(word_dict,train_data)
	testMatrix = _vectorize(word_dict,test_data)

	print ("train NB model")

	label_words_prob,label_prob = trainNB(trainMatrix, train_label)

	print ("test NB model")
	test_predict = testNB(testMatrix,label_words_prob,label_prob)

	print ("top 5 words associated with each of the 12 locations:")
	tops_words = most_Relevant_words(label_words_prob,label_prob,trainMatrix,vocab)
	for label in tops_words.keys():
		print (label,":","/".join(tops_words[label]))

	fw = open(out_path,"w")
	for i in range(len(test_predict)):
		fw.write(test_predict[i]+","+test_label[i]+","+" ".join(test_data[i])+"\n")

	print ("acc:",sum(np.array(test_predict)==np.array(test_label))/float(len(test_label)))



if __name__ == '__main__':
	main(sys.argv[1],sys.argv[2],sys.argv[3])

    

        
