# -*- coding: utf-8 -*-
#!/usr/bin/python2
###################################
# CS B551 Fall 2018, Assignment #3
# D. Crandall
#
# There should be no need to modify this file, although you 
# can if you really want. Edit pos_solver.py instead!
#
# To get started, try running: 
#
#   python ./label.py bc.train bc.test.tiny
#
'''
bc.test.tiny predict result:

So far scored 1 sentences with 17 words.
                   Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
         1. Simple:       82.35%                0.00%
            2. HMM:       88.24%                0.00%
        3. Complex:       29.41%                0.00%
----
                  Simple     HMM Complex desperately ,    nick flashed one  hand up   ,    catching poet's neck in   the  bend of   his  elbow .   
0. Ground truth  -999.00 -999.00 -999.00 adv         .    noun verb    num  noun prt  .    verb     noun   noun adp  det  noun adp  det  noun  .   
      1. Simple  -999.00 -999.00 -999.00 adv         x    noun verb    noun noun adp  x    noun     noun   noun adp  det  noun adp  det  noun  .   
         2. HMM  -999.00 -999.00 -999.00 adv         .    pron verb    num  noun prt  .    verb     noun   noun adp  det  noun adp  det  noun  .   
     3. Complex  -999.00 -999.00 -999.00 .           .    pron pron    num  num  num  .    .        .      .    .    det  det  det  det  det   det 

==> So far scored 2 sentences with 35 words.
                   Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
         1. Simple:       77.14%                0.00%
            2. HMM:       91.43%                0.00%
        3. Complex:       28.57%                0.00%
----
                  Simple     HMM Complex the  air  hose was  free !    !   
0. Ground truth  -999.00 -999.00 -999.00 det  noun noun verb adj  .    .   
      1. Simple  -999.00 -999.00 -999.00 det  noun noun verb adj  .    .   
         2. HMM  -999.00 -999.00 -999.00 det  noun noun verb adj  .    .   
     3. Complex  -999.00 -999.00 -999.00 pron noun .    verb verb verb verb

==> So far scored 3 sentences with 42 words.
                   Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
         1. Simple:       80.95%               33.33%
            2. HMM:       92.86%               33.33%
        3. Complex:       28.57%                0.00%


'''


from pos_scorer import Score
from pos_solver import *
import sys

# Read in training or test data file
#
def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]

    return exemplars


####################
# Main program
#

if len(sys.argv) < 3:
    print("Usage: \n./label.py training_file test_file")
    sys.exit()

(train_file, test_file) = sys.argv[1:3]

#(train_file, test_file) = ('bc.train','bc.test.tiny')#bc.train#bc.test.tiny#'bc.test'

print("Learning model...")
solver = Solver()
train_data = read_data(train_file)

#After training, pi, A, B are saved. Next test reads parameters directly without repetitive training. In solver, pickle reads parameters directly.
solver.train(train_data)


##############
print("Loading test data...")
test_data = read_data(test_file)

print("Testing classifiers...")
scorer = Score()

Algorithms = ("Simple", "HMM", "Complex")
Algorithm_labels = [ str(i+1) + ". " + Algorithms[i] for i in range(0, len(Algorithms) ) ]
for (s, gt) in test_data:

    outputs = {"0. Ground truth" : gt}
        
    # run all algorithms on the sentence
    for (algo, label) in zip(Algorithms, Algorithm_labels):
        outputs[label] = solver.solve( algo, s) 

    # calculate posteriors for each output under each model
    posteriors = { o: { a: solver.posterior( a, s, outputs[o] ) for a in Algorithms } for o in outputs }
    Score.print_results(s, outputs, posteriors, Algorithms)
        
    scorer.score(outputs, gt)
    scorer.print_scores()
    
    print("----")
    
