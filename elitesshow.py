import Lenia
import Functions as F
import Settings as S
import os
import glob

def create_elite(cells):
    
        name = f"elite"
        R = Lenia.pattern["aquarium"]["R"]
        T = Lenia.pattern["aquarium"]["T"]
        kernels= Lenia.pattern["aquarium"]["kernels"].copy()
        
        
        # Refit
        newcells = [[[0 for _ in range(54)] for _ in range(54)] for _ in range(S.Channel_size)]
        for i in range(S.Channel_size):
            for j in range(9, len(cells[i])):
                for k in range(9, len(cells[i][j])):
                    newcells[i][j-10][k-10] = cells[i][j][k]
        
        return {
            "name": name,
            "R": R,
            "T": T,
            "cells": newcells,
            "kernels": kernels
        }


def show_elites():
    
    Evos =[]
    keys = []
    
    for files in glob.glob(os.path.join(S.elite_fp, '*.json')):
        Evos.append(F.read_data(files))
        keys.append(os.path.basename(files).split(".")[0])
        
    for i in range(len(Evos)):
        print("now showing elite: ", keys[i])
        Lenia.run_world_execute(create_elite(Evos[i]), True)
    
    return
        
        
if __name__ == "__main__":
    show_elites()