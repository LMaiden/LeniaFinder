#################
import numpy as np
import Settings as S
import Lenia
import random
import os
import glob

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
        timeout = 10  # Set the timeout in seconds
        world = Lenia.run_world_execute(elite_map[specimen])
        # TODO: Evaluate the world
            #TODO : Deserialize the world
            #TODO : Get metrics
        #TODO : MAP

        
    return


###################

if __name__ == "__main__":
    #TODO : run the map elites
    population = 10
    map_elites(population)