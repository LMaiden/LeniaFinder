import numpy as np
import json
import os
import multiprocessing
import time
import glob
import codecs

import Settings as S

################### Maths Function ####################

def bell(x ,m ,s):
    return np.exp(-((x-m)/s)**2 /2)



################### JSON FUNCTIONS ####################

def Save_tojson(entity, filename, folder):
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as fp:
        json.dump(entity, fp)
        
def Save_tojson_array(array, filename, folder):
    """
    Save a NumPy array or a list to a JSON file in the specified folder.
    Appends the array to the file without overwriting existing content.
    """
    filepath = os.path.join(folder, filename)
    with open(filepath, 'a') as fp:
        for a in array:
            a = a.tolist() 
            json.dump(a, fp)
            fp.write('\n')
        fp.write('\n')
        
def Load_json(filename, folder):
    filepath = os.path.join(folder, filename)
    with open(filepath, 'r') as fp:
        entity = json.load(fp)
    return entity

def clear_data_folder():
    filepath = S.data_fp
    files = glob.glob(os.path.join(filepath, '*.json'))
    for f in files:
        os.remove(f)
        
def read_data(filepath):
    """
    Read data from a JSON file and return it as a NumPy array.
    Handles files with multiple JSON objects per line.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return np.array(data)
    
    
######################## Data Treatment Funciton ####################


# Mass, Center of Mass, Speed, angular speed, entropy. Mean and std

def mass(world):
    mass = []
    for chan in world:
        mass.append(np.sum(np.asarray(chan)))
    mass = np.asarray(mass)
    return np.sum(mass)

def com(world, mass):
    """
    Calculate the center of mass for each channel in the world.
    Returns a list of center of mass coordinates
    """
    com= [0, 0, 0]
    
    if mass == 0:
        return com
    
    for z in range(len(world)):
        for y in range(len(world[z])):
            for x in range(len(world[z][y])):
                if world[z][y][x] > 0:
                    com[0] += x * world[z][y][x]
                    com[1] += y * world[z][y][x]
                    com[2] += z * world[z][y][x]
    
    return com/ mass

def velocity(coord ):
    velocity = np.zeros(len(coord)-5)
    for i in range(0, len(coord)-3, 3):
        velocity[i] = np.sqrt((coord[i+3] - coord[i])**2 + (coord[i+4] - coord[i+1])**2 + (coord[i+5] - coord[i+2])**2)
    return velocity

def angvelocity():
    return