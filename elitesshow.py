
import Functions as F
import Settings as S
import os
import glob
import LeniaTest as LT
import SLE

def create_elite_test(cells):
    
        name = f"elite"
        R = SLE.pattern["aquarium"]["R"]
        T = SLE.pattern["aquarium"]["T"]
        kernels= SLE.pattern["aquarium"]["kernels"].copy()
        
        
        # Refit
        newcells = [[[0 for _ in range(59*2)] for _ in range(59*2)] for _ in range(S.Channel_size)]
        for i in range(S.Channel_size):
            for j in range(9, len(cells[i])):
                for k in range(9, len(cells[i][j])):
                    newcells[i][j-10][k-10] = cells[i][j][k]
                    newcells[i][54+j-10][k-10+54] = cells[i][j][k]
        
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
        LT.run_world_execute(create_elite_test(Evos[i]), True)
    
    return
        
        
if __name__ == "__main__":
    show_elites()