#!/bin/python
# put your group assignment problem here!
import sys
filename = sys.argv[1]
k = int(sys.argv[2])
m = int(sys.argv[3])
n = int(sys.argv[4])
studentls=[]
f = open(filename)
file=f.read().splitlines()
for line in file:
    name,size,prefer,notPrefer=line.split(" ")
    if prefer is not "_":
        prefer=prefer.split(",")
    else:
        prefer=[]
    if notPrefer is not "_":
        notPrefer=notPrefer.split(",")
    else:
        notPrefer=[]
    studentls.append([name,size,prefer,notPrefer,len(studentls)])
f.close()

print(studentls)
groupls = []
totalTime = 0

def costFunction(state):
    #k mins to grade each group
    totalTime = k*len(state)
    #1 mins for each student not in prefer groupsize
    for i in range(len(state)):
        for j in state[i]:
            if studentls[j][1] != len(state[i]) and studentls[j][1] != 0:
                totalTime +=1
    #n mins for each student not get their prefer groupmate
    for i in range(len(state)):
        for j in state[i]:
            for l in state[i]:
                if l != j:
                    if studentls[l][0] not in studentls[j][2]:
                        totalTime += n
    #m mins for each student group with person they are not prefer to group with
                    if studentls[l][0] in studentls[j][3]:
                        totalTime += m
    return totalTime

##initial state represent by name by number
init = []
for i in range(len(studentls)):
    init.append([i])


def successor(state):
    successor = []
    for i in range(0,len(state)-1):
	for j in range(i+1,len(state)):
            if len(state[i])+len(state[j])<=3:
                temp = state[:i]+state[i+1:j]+state[j+1:]
                temp.append(state[i]+state[j])
                successor.append(temp)
    return successor
def solve(init):
    minCost = costFunction(init)
    minState = init
    fringe = [[init,costFunction(init)]]
    while len(fringe)>0:
        a = fringe.pop()
        for i in successor(a[0]):
            fringe.append([i,costFunction(i)])
            if costFunction(i) < minCost:
                minCost = costFunction(i)
                print(minCost)
                minState = i
                print(minState)
    ##transfer minState into names
    output = []
    for i in minState:
        templs = []
        for j in i:
            templs.append(studentls[j][0])
        output.append(templs)
    return output
print(init)
print(costFunction(init))              
print(solve(init))
