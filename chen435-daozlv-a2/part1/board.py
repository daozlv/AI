import numpy as np
import copy

def check(board,n):
	# check if the board is right
	if type(board) == str:
		if len(board) == n*(n+3):
			return True
		else:
			return False
	else:
		if board.shape == (n+3,n):
			return True
		else:
			return False

def ToArray(board,n):
	if not check(board,n):
		print ("you input a invalid board")
		exit()
	if type(board) == str:
		board = np.array(list(board)).reshape([n+3,n])
		return board
	else:
		return board

def ToStr(board):
	if type(board) != str:
		board = "".join(board.reshape(-1))
		return board
	else:
		return board
def LocTop(board,n,column):
	#string to array
	board = ToArray(board,n)
	row = n+3
	for i in range(n+3):
		if board[i][column-1] != ".":
			row = i
			break
	return row
def drop(board,n,column,pebble):
	board = ToArray(board,n)
	def check(board,column):
		# check if the player is allowed to drop a pebble into this column
		if board[0][column-1]!= ".":
			return False
		else:
			return True
	if check(board,column):
		new_board = copy.deepcopy(board)
		row = LocTop(board,n,column)
		new_board[row-1,column-1]=pebble
		return new_board
	else:
		return None
def rotate(board,n,column):
	board = ToArray(board,n)
	def check(board,column):
		# check if the player is allowed to rotate a pebble into this column
		if board[-1][column-1]!= ".":
			return True
		else:
			return False	
	if check(board,column):
		new_board = copy.deepcopy(board)
		bottom_value = new_board[-1,column-1]
		top_row = LocTop(board,n,column) 
		for i in range(n+2, top_row,-1): 
			new_board[i,column-1]=new_board[i-1,column-1]
		new_board[top_row,column-1] = bottom_value
		return new_board
	else:
		return None

def check_win(board,pebble):
	n = board.shape[1]
	def check_row():
		for i in range(n):
			if (board[i,:]==pebble).all():
				return True
		return False
	def check_column():
		for i in range(n):
			if (board[:n,i]==pebble).all():
				return True
		return False		
	def check_diag():
		
		flag = False
		for i in range(n):
			if board[i,i]!=pebble:
				flag = False
				break
			else:
				flag = True
		if flag:
			return True
		for i in range(n):
			if board[n-i-1,i]!=pebble:
				flag = False
				break
			else:
				flag = True
		return flag
	return (check_row()|check_column()|check_diag())
									
def run(board,action,pebble):

	n = board.shape[1]
	if action>0:
		board_new = drop(board,n,action,pebble) # pebble 
	else:
		board_new = rotate(board,n,-1*action) # pebble 

	return board_new


if __name__ == '__main__':
	tool = Board()
	board = tool.ToArray("..xxx.xooxooxxxoxo",3)
	print (board)
	print (tool.drop(board,3,3,"x"))
	print (tool.rotate(board,3,3))
	print (tool.check_win(board,"x"))
