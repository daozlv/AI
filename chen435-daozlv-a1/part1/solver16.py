#!/bin/python
# solver16.py : Circular 16 Puzzle solver
# Based on skeleton code by D. Crandall, September 2018
#
from Queue import PriorityQueue
from random import randrange, sample
import sys
import string


# shift a specified row left (1) or right (-1)
def shift_row(state, row, dir):
    change_row = state[(row*4):(row*4+4)]
    return ( state[:(row*4)] + change_row[-dir:] + change_row[:-dir] + state[(row*4+4):], ("L" if dir == -1 else "R") + str(row+1) )

# shift a specified col up (1) or down (-1)
def shift_col(state, col, dir):
    change_col = state[col::4]
    s = list(state)
    s[col::4] = change_col[-dir:] + change_col[:-dir]
    return (tuple(s), ("U" if dir == -1 else "D") + str(col+1) )

# check the distance to goal
def distance(state):
    dis = 0
    for i in range (len(state)):
        if i+1 != state[i]:##current and goal place is not match
            dis += 1
    return dis
##            curCol = (state[i])//4
##            curRow = state[i]%4
##            if curRow !=0:
##                curCol +=1
##            goalCol = (i+1)//4
##            goalRow = (i+1)%4
##            if goalRow !=0:
##                goalCol +=1
##            dis+=abs(curCol - goalCol) + abs(curRow - goalRow)
##            if abs(curCol - goalCol) >2:
##                dis -= 2
##            if abs(curRow - goalRow) >2:
##                dis -= 2
##    return dis/4

# pretty-print board state
def print_board(row):
    for j in range(0, 16, 4):
        print '%3d %3d %3d %3d' % (row[j:(j+4)])

# return a list of possible successor states
def successors(state):
    return [ shift_row(state, i, d) for i in range(0,4) for d in (1,-1) ] + [ shift_col(state, i, d) for i in range(0,4) for d in (1,-1) ] 

# just reverse the direction of a move name, i.e. U3 -> D3
def reverse_move(state):
    return state.translate(string.maketrans("UDLR", "DURL"))

# check if we've reached the goal
def is_goal(state):
    return sorted(state) == list(state)
    
# The solver! - using A* right now
def solve(initial_board):
    fringe = PriorityQueue()
    fringe.put((-distance(initial_board),initial_board, "", 0, distance(initial_board)))
    statels=[initial_board]
    minstep = distance(initial_board)
    while not fringe.empty():
        top_element = fringe.get()
        ((pri, state, route_so_far, steps, dis)) = top_element
        ##print(pri)
        for (succ, move) in successors(state):
            if is_goal(succ):
                return( route_so_far + " " + move ) 
            if succ not in statels:
                fringe.put((steps+1+distance(succ), succ, route_so_far + " " + move,steps+1,distance(succ)))
    return False

# test cases


 
start_state = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        start_state += [ int(i) for i in line.split() ]

if len(start_state) != 16:
    print "Error: couldn't parse start state file"

print "Start state: "
print_board(tuple(start_state))

print "Solving..."
route = solve(tuple(start_state))

print "Solution found in " + str(len(route)/3) + " moves:" + "\n" + route
