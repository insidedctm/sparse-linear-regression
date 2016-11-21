'''
Created on 6 Nov 2016

@author: robineast1
'''

import numpy as np
from numpy.random import uniform, normal

NUM_ROWS = 800
NUM_COLS = 5
coefs = np.array([1.0, 2.2, 5.5, 1.8, 10.9])
A_filename = "../../data/A.txt"
b_filename = "../../data/b.txt"

rawA = np.round(uniform(0,1, NUM_ROWS * NUM_COLS),2)
sparseA = np.multiply(rawA, rawA < 0.6)
A = np.reshape(sparseA, (NUM_ROWS, NUM_COLS))
b = np.round(np.dot(A,coefs) + normal(0,2,NUM_ROWS), 2)

fA = open(A_filename, 'w')
fb = open(b_filename, 'w')
for i in range(0, NUM_ROWS):
    for j in range(0, NUM_COLS):
        if A[i,j] != 0.0:
            fA.write('%d,%d,%.2f\n' % (i,j,A[i,j]) )
    fb.write('%d,%.2f\n' % (i,b[i]))

fA.close()
fb.close()
