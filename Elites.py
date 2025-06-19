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
import shutil

################# MAP ELITES #################

def init_population(number):
    population = {}
    for i in range(number):
        name = f"evo_{i}"
        R = Lenia.pattern["aquarium"]["R"]
        T = Lenia.pattern["aquarium"]["T"]
        kernels= Lenia.pattern["aquarium"]["kernels"].copy()
        cells = [[[random.random() for _ in range(54)] for _ in range(54)] for _ in range(3)]
        population[name] = {
            "name": name,
            "R": R,
            "T": T,
            "cells": cells,
            "kernels": kernels
        }

    return population

def new_generation(parents, generation, initial_population):
    new_pop = {}
    
        
    for i in range(initial_population):
        name = f"evo_{i}_{generation}"
        R = Lenia.pattern["aquarium"]["R"]
        T = Lenia.pattern["aquarium"]["T"]
        kernels= Lenia.pattern["aquarium"]["kernels"].copy()
        cells  = F.crossover(F.mutate(parents[random.randint(0, len(parents)-1)]),
                                 F.mutate(parents[random.randint(0, len(parents)-1)]))
        cells = cells[0 : 3][0 : 54][0 : 54]
        newcells = np.zeros((3, 54, 54))
        if random.random() < 0.05:
            for i in range(len(newcells)):
                for j in range(len(newcells[i])):
                    for k in range(54):
                        if random.random() < 0.1:
                            newcells[i][j][k] = max(0, min(cells[i][j][k] + random.random(), 1))
                        else:
                            newcells[i][j][k] = cells[i][j][k]
        else:
            newcells = [[[random.random() for _ in range(54)] for _ in range(54)] for _ in range(3)]
        
        new_pop[name] = {
            "name": name,
            "R": R,
            "T": T,
            "cells": newcells,
            "kernels": kernels
        }

    
        
        #Crossover and mutation
        
    return new_pop
        

def run_elites_lenia(pop):
    
    if not os.path.exists(S.data_fp):
        os.makedirs(S.data_fp)
    F.clear_data_folder()
    
    elite_map = pop
    for specimen in elite_map.keys():
        
        
        world = Lenia.run_world_execute(elite_map[specimen], False)
        
    
    Evos =[]
    keys = []
    
    for files in glob.glob(os.path.join(S.data_fp, '*.json')):
        Evos.append(F.read_data(files))
        keys.append(os.path.basename(files).split(".")[0])
        
    metrics = {}
    for i in range(len(Evos)):
        
        metrics[str(keys[i])] = {}
        
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
        
    return metrics


def MAP_ELITES(sample_size, n_elites = 10, iterations=100):
    
    if not os.path.exists(S.elite_fp):
        os.makedirs(S.elite_fp)
    F.clear_elite_folder()
    
    Map = [("test_entity", -100000000000000000000) for _ in range(n_elites)]
    
    for i in range(iterations):
        
        if i == 0:
            population = init_population(sample_size)
            
        else:
            parents = []
            for files in glob.glob(os.path.join(S.data_fp, '*.json')):
                indata= F.read_data(files)
                temp2 = []
                for k in range(3):
                    temp2.append(indata[k][0 : 54][0 : 54])
                parents.append(temp2)
                
            population = new_generation(parents, i, sample_size) #Mutation and crosover step
            
        
        performance = run_elites_lenia(population)
        choose_perf = F.selector(performance, n_elites)
        
        
        
        for entity, score in choose_perf:
            replaced = False
            for elite, escore in Map:
                    if score > escore and not replaced:
                        Map.remove((elite, escore))
                        Map.append((entity, score))
                        Map = sorted(Map, key=lambda x: x[1], reverse=True)
                        print(f"Entity {entity} with score {score} replaced elite {elite} with score {escore}")
                        replaced = True
            
            
            src_file = os.path.join(S.data_fp, f"{entity}.json")
            print(src_file)
            dst_dir = os.path.join(S.elite_fp)
            
            
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            
            if os.path.exists(src_file) and (entity,score) in Map:
                print(f"Saving {entity} to elite folder")
                entity_data = F.read_data(src_file)
                entity_data = entity_data[0 : 3]
                F.Save_tojson_array(entity_data, f"{entity}.json", dst_dir)
                
            for f in glob.glob(os.path.join(S.elite_fp, '*.json')):
                if os.path.basename(f) not in [e[0] + ".json" for e in Map]:
                    print(f"Removing {f} from elite folder")
                    os.remove(f)
                
        Map =  sorted(Map, key=lambda x: x[1], reverse=True)
    
    return Map
    

###################

if __name__ == "__main__":
    #TODO : run the map elites
    MAP_ELITES(sample_size=25)