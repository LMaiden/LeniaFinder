''' Global math function not included in standard libraries '''
###################################################################################
''' imports '''
import numpy as np

###################################################################################
''' Maths Function '''

def bell(x ,m ,s):
    return np.exp(-((x-m)/s)**2 /2)
###################################################################################
''' Metrics Function, 
The expected input will always be a N Dimension numpy array (Number of frame, ...)'''

'''Takes an array and returns a list of masses for each frame'''
def mass (arr):
    res = []
    for frame in arr:
        res.append(np.sum(frame))
    return res

''' Takes an array and return the entropy gain'''
def entropy (arr):
    res = []
    for frame in arr:
        total = np.sum(frame)
        en = frame / total
        en = en[en>0]
        res.append(-np.sum(en * np.log(en)))
    return res





