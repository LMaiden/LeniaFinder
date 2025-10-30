''' central hub for moving data in and outside of Lenia'''

###################################################################################
''' imports '''

import os
import glob
import json

###################################################################################
''' custom imports '''


###################################################################################
''' global variables   '''

path = os.path.dirname(os.path.abspath(__file__))
out_folder  = os.path.join(path, 'dataout')
out_file = 'Lenia_out'

###################################################################################

def dump_frame(frame,iter, file = out_file ,folder = out_folder):
    
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"frame_{str(iter)}.json")
    with open(filepath, 'a') as fp:
        for a in frame:
            a = a.tolist()
            json.dump(a , fp)
            fp.write('\n')
        fp.write('\n')
        
        
def get_frames(folder = out_folder):
    files = sorted(glob.glob(os.path.join(folder, 'frame_*.json')))
    frames = []
    for f in files:
        with open(f, 'r') as fp:
            frame = []
            for line in fp:
                if line.strip():
                    a = json.loads(line)
                    frame.append(a)
            frames.append(frame)
    return frames[1:] #We have a bug at the start, we have way too much data in it, its inconsequential so this duct tape solution is enough




def clear_dataout_folder(folder = out_folder):
    files = glob.glob(os.path.join(folder, '*.json'))
    for f in files:
        os.remove(f)
        
        
    