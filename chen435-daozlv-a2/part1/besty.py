import sys
from board import *
import compute


if __name__ == '__main__':


	n = int(sys.argv[1])
	pebble = sys.argv[2]
	board_string = sys.argv[3]
	seconds = float(sys.argv[4])

	# check the board
	if check(board_string,n):
		pass
	else:
		print ("you have input a wrong board")
		exit()

	if pebble == "x":
		other_pebble = "o"
	else:
		other_pebble = "x"
	board = ToArray(board_string,n) # turn string  such as "...x..o.ox.oxxxo.o"  to array
	action = compute.compute_next(board,2,pebble,other_pebble)
	board_new = ToStr(run(board,action,pebble))
	print (action,board_new)
	action = compute.compute_next(board,4,pebble,other_pebble)
	board_new = ToStr(run(board,action,pebble))
	print (action,board_new)

