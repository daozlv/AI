import sys
from board import *
"""
the heuristic evaluation function
we call the max number of player's pebbles in any row or in any column or in any diagonals "max_num"
the line have the most player's pebbles called "max_num_line"
the the squre has player's pebble called "filled_square"
the squre has other pebble called "other_filled_square"
the squre is empty called "unfilled_square"

The "max_num" is bigger, the score is larger.
if the square below unfilled_square is empty, you need at more than one drops to fill the square.
if the square above other_filled_square filled with other player's pebbles, you also need at least two rotates to put your pebbles into this square.

so we design the heuristic evaluation function is :

	s1 = max_num * 100

if the square in max_num_line is unfilled_square:
	
	s2 = -1 * empty_num * 50  # empty_num is number of unfilled_square below this square

if the square in max_num_line is other_filled_square:

	s3 = -1 * distance *150 + b  
	# distance is the square numuber between this square and filled_square or unfilled_square above
	# b = -50 if you find the square with your pebble above this square
	# b = -100  if you find the empty square  above this square

the score is :
	
	s = s1+\sum_{square in the max_num_line}(s2+s3)

for example:
	...
	..o
	x.o
	x.x
	ooo
	xox

	for pebble "o", the third column is a max_num_line

	max_num = 2*100 = 200

	there is only a empty square in third column.

	for this square:
		s2 = -1*1*50 
		s3 = 0
	so s = 150

	consider the third row

	s1 = 1*100 = 100 (only one "o")
	s2 = 2*(-50) = -100  (the second square)
	s3 = 2*-150-100 = -400

	s = -400


if a column is completed and it has no n continuous pebble (player's), you can't win the game througe rotate on this column,
so we set s = -1000

"""
def max_num_line(board,pebble):
	"""
	get the line with most player's pebble.
	and calculate the s1 score.
	"""
	n = board.shape[1]
	_max = 0
	_max_line = []
	
	#check row
	for i in range(n):
		num = sum(board[i,:]==pebble)
		if num>_max:
			_max_line = ["row_%d"%i]
			_max = num
		if num == _max:
			_max_line.append("row_%d"%i)
	
	#check column
	for i in range(n):
		num = sum(board[:n,i]==pebble)
		if num>_max:
			_max_line = ["col_%d"%i]
			_max = num
		if num == _max:
			_max_line.append("col_%d"%i)
	
	#check diag
	num=0
	for i in range(n):
		if board[i,i]==pebble:
			num+=1
	if num>_max:
		_max_line = ["diag_main"]
		_max = num
	if num == _max:
		_max_line.append("diag_main")

	num=0
	for i in range(n):
		if board[n-i-1,i]==pebble:
			num+=1
	if num>_max:
		_max_line = ["diag_acc"]
		_max = num
	if num == _max:
		_max_line.append("diag_acc")

	return set(_max_line),_max

def calc_s2(board,row,column):

	n = board.shape[1]
	num=0

	for i in range(row,n+3):
		if board[i,column] == ".":
			num+=1
		else:
			break

	return num*-50

def calc_s3(board,row,column,pebble):

	n = board.shape[1]
	num=0

	for i in range(row-1,0,-1):
		if board[i,column] == "." :
			num+=-100
			break
		elif board[i,column] == pebble:
			num+=-50
			break
		else:
			num+=-150
	else:
		for i in range(n+2,row-1,-1):
			if board[i,column] == pebble:
				num+=-50
			else:
				num+=-150

	return num


def check_for_rotate(board,column,pebble):
	"""
	if a column has no n continuous pebble,(empty square as filled_square)
	"""
	n = board.shape[1]
	continuous = 0
	pebble_num = sum((board[:,column]==".")|(board[:,column]==pebble))
	if pebble_num<n:
		return False
	for i in range(n+3):
		if board[i,column] == "." or board[i,column] == pebble:
			continuous+=1
		else:
			if continuous>=n:
				return True
			else:
				continuous = 0
	if continuous>0:
		for i in range(n+3):
			if board[i,column] == "." or board[i,column] == pebble:
				continuous+=1
			else:
				if continuous>=n:
					return True
				else:
					break
		return False		
		

def get_grade(board,pebble):

	_max_line,_max = max_num_line(board,pebble)

	s1 = _max*100

	scores = [] 

	n = board.shape[1]

#	print (board)
#	print (_max_line)

	for line in _max_line:

		l,num = line.split("_")

		s23 = 0 
		# calc the s2 s3  in row.
		if l == "row":
			num = int(num)
			for i in range(n):
				if board[num,i] == ".": # empty square
					s2 = calc_s2(board,num,i)
					s23 += s2
				elif board[num,i] != pebble:
					s3 = calc_s3(board,num,i,pebble)
					s23 += s3

		# calc the s2 s3  in column.
		if l == "col":
			num = int(num)
			for i in range(n):
				if board[i,num] == ".": # empty square
					s2 = calc_s2(board,i,num)
					s23 += s2
				elif board[i,num] != pebble:
					s3 = calc_s3(board,i,num,pebble)
					s23 += s3

			if not check_for_rotate(board,num,pebble):
				s23 = -1000

		# calc the s2 s3  in diagonals.
		if l == "diag":
			if num == "main":
				for i in range(n):
					if board[i,i]==".":
						s2 = calc_s2(board,i,i)
						s23 += s2
					elif board[i,i] != pebble:
						s3 = calc_s3(board,i,i,pebble)
						s23 += s3
			if num == "acc":
				for i in range(n):
					if board[n-i-1,i]==".":
						s2 = calc_s2(board,n-i-1,i)
						s23 += s2
					elif board[i,i] != pebble:
						s3 = calc_s3(board,n-i-1,i,pebble)
						s23 += s3

		scores.append(s23)

	s = s1 + max(scores)

	return s



if __name__ == '__main__':
	n = int(sys.argv[1])
	pebble = sys.argv[2]
	board = sys.argv[3]

	board = ToArray(board,n)
	print(board)
	print (get_grade(board,pebble))






