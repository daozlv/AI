import numpy as np 
import grade
import minimax
from board import *
INF = 100000000000 #positive infinite
N_INF = -INF #negative infinite

#construct Tree
def create_Tree(chessboard,current_deep,deep,node_type,action,pebble,other_pebble):

	n = chessboard.shape[1]
	#print ("current deep / deep: %d/%d"%(current_deep,deep))
	if current_deep == deep or check_win(chessboard,pebble) == True or check_win(chessboard,other_pebble) == True:
		value = grade.get_grade(chessboard,pebble)
		return [action,node_type,value,[]] #leaf node: action, node_type(max or min),value, child_tree.

	Root = [] #root of tree
	if node_type == 'max':
		Root = [action,node_type,N_INF,[]]
	else:
		Root = [action,node_type,INF,[]]
	
	actions = []

	# try all possible actions 
	for i in range(n):
		if drop(chessboard,n,i+1,pebble) is not None:
			actions.append(i+1)
		if rotate(chessboard,n,i+1) is not None:
			actions.append(-1*(i+1))

	for action in actions:
		if node_type == 'max':
			chessboard_new = run(chessboard,action,pebble)
			Root[-1].append(create_Tree(chessboard_new,current_deep+1,deep,'min',action,pebble,other_pebble))
		else:
			chessboard_new = run(chessboard,action,other_pebble)
			Root[-1].append(create_Tree(chessboard_new,current_deep+1,deep,'max',action,pebble,other_pebble))
	return Root
	
#calc the next step
def compute_next(chessboard,deep,pebble,other_pebble):
	Root = create_Tree(chessboard,1,deep,'max',0,pebble,other_pebble)
	next_step = minimax.get_next(Root)
	return next_step#return the action (move)