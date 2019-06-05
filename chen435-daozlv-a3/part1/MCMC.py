# -*- coding: utf-8 -*-

class MarkovChains:
    def __init__(self):
        self.count=0;
        self.arry=[];
    #list输入数字序列，输入其他内容的字典编码在外面完成
    def train(self,list):
        max = 0;
        for item in list:
            if item>max:
                max=item
        self.count=max+1
        for i in range(self.count):
            list2=[]
            for j in range(self.count):
                list2.append([0]*self.count)
            self.arry.append(list2)
        for j in range(list.__len__()-2):
            self.arry[list[j]][list[j+1]][list[j+2]]+=1;
        #print(self.arry)
    #预测返回值是统计数量，只能比较大小，不是概率
    def predict(self,list_previous,list_now,list_next):
        return (self.arry[list_previous][list_now][list_next])
 
markov=MarkovChains()
list=[24,26,24,7,14,12,14,12,24,26,24,26,24,26,24,26,24,26,24,26,24,26,24,21,12,7,12,14,24,26,24,26,24,26,24,14,12,7,12,14,12,14,24,26,24,7,26,7,24,26,24,26,24,26,24,14,12,14,24,21,7,21,24,21,24,14,12,7,24,21,24,7,12,7,12,7,26,24,26,24,26,24,26,14,7,12,14,24,26,24,26,24,26,24,26,24,21,24,26,24,21,24,21,26,24,21,24,21,24,21,19,21,19,12,21,7,24,21,24,26,16,21,16,7,12,7,12,7,21,24,21,24,21,24,21,24,21,24,21,24,7,12,7,21,7,16,24,26,24,26,21,24,21,24,26,24,21,24,14,12,7,14,7,24,17,20,7,20,19,20,19,20,7,13,7,20,7,76,14,12,14,12,16,12,16,28,29,28,27,29,28,14,19,12,7,24,12,14,35,24,7,38,39,32,42,32,35,38,42,38,24,26,14,39,7,14,12,7,12,14,12,7,32,39,32,39,26,39,14,36,28,36,40,36,7,36,7,12,7,36,28,19,36,32,28,12,19,7,12,7,19,7,19,32,7,32,7,13,22,12,19,12,19,14,32,25,22,25,13,22,7,13,7,25,7,22,25,7,25,1,2,5,4,2,4,2,3,2,3,1,2,7,6,3,2,3,2,3,2,3,2,3,7,3,47,50,47,50,47,50,46,50,47,50,47,51,47,35,21,12,7,14,12,7,35,41,35,24,21,14,20,29,33,29,31,29,31,29,31,33,29,7,13,7,17,7,31,29,31,29,31,29,31,20,31,29,31,29,13,7,20,17,7,13,31,29,43,42,43,42,45,42,44,46,40,43,31,40,31,45,43,45,31,44,41,44,42,41,42,41,12,35,14,21,7,40,41,35,41,43,42,43,40,19,12,19,14,40,36,28,36,40,33,37,33,28,33,28,33,40,33,40,36,7,14,12,7,14,28,19,28,7,28,7,58,59,58,55,58,55,59,45,26,32,39,42,45,32,7,12,14,12,14,7,59,58,59,7,57,54,49,54,49,52,49,52,49,59,52,49,59,60,27,30,27,30,27,34,16,39,32,26,43,7,14,7,12,14,48,53,56,48,56,53,43,7,59,60,59,60,49,59,60,59,60,59,60,59,60,59,60,59,60,55,45,59,7,14,12,7,19,28,33,15,18,15,18,7,12,14,33,29,23,29,25,7,15,38,14,24,18,15,18,15,18,15,18,15,18,15,10,9,18,15,18,15,18,15,7,8,7,8,7,18,15,18,7,8,7,8,18,7,18,15,18,15,18,15,18,15,18,8,7,8,7,11,9,10,9,7,12,7,12,35,21,61,62,63,61,50,63,66,63,66,67,65,63,70,68,69,67,68,71,64,63,64,72,73,74,73,74,64,63,64,75]
#生成隐马尔可夫需要的序列格式
"""strstr="["
for i in list:
    strstr=strstr+"["+str(i)+"],"
print(strstr)"""
 
markov.train(list)
probList=[]
for i in range(markov.count):
    probList.append(markov.predict(63,64,i))
print (str(probList))
print(markov.count)
print(probList.index(max(probList)))

