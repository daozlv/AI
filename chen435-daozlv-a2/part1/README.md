# Besty game
This is a program for automatically playing besty game.
the process is:
1.read the board ,check it , and save it in a array.
2.construct a Tree with N layer. the leaf node is at Nst layer or it is a "win" or "fail" node.
3.get the best action through minimax slgorithm.
4.run the best action and update the board

## board.py
we read the current board (string) and stored it in a numpy array for calculation.
we define a "drop" and a "rotate" function to execute drop and rotate respectively.
We difine a check_win function to judge if the player win the game.

## compute.py 
construct the Tree 

## minimax.py
get the best action through minimax algorithm with alpha-beta search

## grade.py : heuristic evaluation function
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