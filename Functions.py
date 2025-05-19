import numpy as np
import json
import os
import multiprocessing
import time
import glob

import Settings as S

def bell(x ,m ,s):
    return np.exp(-((x-m)/s)**2 /2)

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
        print(f"Saved {filename} to {filepath}")
        
def Load_json(filename, folder):
    filepath = os.path.join(folder, filename)
    with open(filepath, 'r') as fp:
        entity = json.load(fp)
    return entity

def clear_data_folder():
    filepath = S.data_fp
    files = glob.glob(os.path.join(filepath, '*.json'))
    for f in files:
        print(f"Deleting {f}")
        os.remove(f)