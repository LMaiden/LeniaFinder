import numpy as np
import json
import os
import multiprocessing
import time
import glob
import codecs
import math

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
        
def clear_elite_folder():
    filepath = S.elite_fp
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

#################### Fitness Selection Functions ####################

def selector(performance, n_elites):
    """
    Selects the top n_elites entities based on their performance metrics.
    Returns a list of (entity, score) tuples.
    """
    def compute_score(metrics):
        return (10*metrics['avg_mass']) - ( metrics['std_mass']) -  metrics['avg_velocity']

    sorted_entities = sorted(
        performance.items(),
        key=lambda item: (item[1]['std_mass'], item[1]['avg_velocity'])
    )[:n_elites]

    result = [(entity, compute_score(metrics)) for entity, metrics in sorted_entities]
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return sorted_result


#Mutator Functions

def mutate(world, mutation_rate=0.1):
    """
    Mutate the world by randomly changing some values based on the mutation rate.
    Returns a new mutated world.
    """
    mutated_world = np.copy(world)
    for z in range(len(mutated_world)):
        for y in range(len(mutated_world[z])):
            for x in range(len(mutated_world[z][y])):
                if np.random.rand() < mutation_rate:
                    mutated_world[z][y][x] += np.random.normal(0, 0.1)  # Add a small random value
                    mutated_world[z][y][x] = max(mutated_world[z][y][x], 0)  # Ensure non-negativity
    return mutated_world

def crossover(parent1, parent2):
    """    Perform crossover between two parent worlds to create a new world.
    Returns a new world that is a combination of the two parents.
    """
    child = np.copy(parent1)
    for z in range(len(child)):
        for y in range(len(child[z])):
            for x in range(len(child[z][y])):
                if np.random.rand() < 0.5:  # Randomly choose genes from either parent
                    child[z][y][x] = parent2[z][y][x]
    return child
