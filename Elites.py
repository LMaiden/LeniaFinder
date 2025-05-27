#################
import numpy as np
import Settings as S
import Lenia
import random
import os
import glob
from datetime import datetime, timedelta

################## Fun #################

import Functions as F

################# MAP ELITES #################

def init_population(number):
    population = {}
    for i in range(number):
        name = f"evo_{i}"
        R = Lenia.pattern["aquarium"]["R"]
        T = Lenia.pattern["aquarium"]["T"]
        #TODO : generate random cells, kernels experimentation is for way later
        kernels= Lenia.pattern["aquarium"]["kernels"].copy()
        cells = Lenia.pattern["aquarium"]["cells"].copy()
        population[name] = {
            "name": name,
            "R": R,
            "T": T,
            "cells": cells,
            "kernels": kernels
        }

    return population

def map_elites(population):
    
    if not os.path.exists(S.data_fp):
        os.makedirs(S.data_fp)
    F.clear_data_folder()
    
    elite_map = init_population(population)
    for specimen in elite_map.keys():
        
        
        world = Lenia.run_world_execute(elite_map[specimen])
        
    
    Evos =[]
    keys = []
    
    for files in glob.glob(os.path.join(S.data_fp, '*.json')):
        Evos.append(F.read_data(files))
        keys.append(os.path.basename(files).split(".")[0])
        print(keys)
        
    metrics = {}
    for i in range(len(Evos)):
        
        metrics[str(keys[i])] = {}
        print(metrics.keys())
        
        mass = np.array([])
        com = np.array([])
        
        for j in range(len(Evos[i])):
            
            mass = np.append(mass, F.mass(Evos[i][j : j+3]))
            com = np.append(com, F.com(Evos[i][j : j+3], mass[-1]))
            
        metrics[str(keys[i])]["mass"] = mass
        metrics[str(keys[i])]["avg_mass"] = np.mean(mass)
        metrics[str(keys[i])]["std_mass"] = np.std(mass)
        metrics[str(keys[i])]["center_of_mass"] = com
        
        velocity = F.velocity(com)
        
        metrics[str(keys[i])]["velocity"] = velocity
        metrics[str(keys[i])]["avg_velocity"] = np.mean(velocity)
        metrics[str(keys[i])]["std_velocity"] = np.std(velocity)
        
    return


###################

if __name__ == "__main__":
    #TODO : run the map elites
    population = 10
    map_elites(population)