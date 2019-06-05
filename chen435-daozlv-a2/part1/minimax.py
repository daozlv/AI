#coding=utf-8
INF = 100000000000 #positive infinite
N_INF = -INF #negative infinite
import random

def generate_value(Node,alpha,beta):
	#leaf 
	if len(Node[-1]) == 0: return Node[-2]
	#other
	children = Node[-1] #child set
	for child in children:
		if beta <= alpha: break
		c_value = generate_value(child,alpha,beta)
		if Node[1] == 'min' and c_value < beta:
			beta = c_value
		elif Node[1] == 'max' and c_value > alpha:
			alpha = c_value
	if Node[1] == 'min':
		return beta
	else:
		return alpha

#Alpha-Beta Prune Algorithm
def Alpha_Beta_Prune_Algorithm(Root):
	return generate_value(Root,N_INF,INF)

def get_next(Root):
	max_value = N_INF
	next_step_list = []
	for child in Root[-1]:
		value = generate_value(child,N_INF,INF)
		if value > max_value:
			next_step_list.clear()
			max_value = value
			next_step_list.append(child[0])
		elif value == max_value:
			next_step_list.append(child[0])

	length = len(next_step_list)
	index = random.randint(0,length-1)
	return next_step_list[index]