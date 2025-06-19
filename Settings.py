# Lenia Parameters

size = 64;  
mid = size // 2;  
scale = 1;  
cx, cy = 10, 10
frame_id = 0

### filepaths
elite_fp = r'C:\Users\gweno\Documents\Homework\ResearchEC\elites'
data_fp = r'C:\Users\gweno\Documents\Homework\ResearchEC\data'
hof_fp = r'C:\Users\gweno\Documents\Homework\ResearchEC\Hof_entities'

### Spaghetti

import numpy as np

Channel_size = 3 #Equals to the number of channels in your evoling  -> np.array(Lenia.pattern["aquarium"]["cells"]).shape