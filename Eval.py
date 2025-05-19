import numpy as np

def Eval_mass(world):
    total_mass = 0
    alive = 0
    Mcenter = 0
    for z in range(len(world)):
        for y in range(len(world[z])):
            for x in range(len(world[z][y])):
                total_mass += world[z][y][x]
                if world[z][y][x] > 0:
                    alive += 1
                Mcenter += world[z][y][x] * np.array([x, y, z])
            
    return (total_mass, (total_mass/alive), (Mcenter/total_mass))

def Eval_velocity(world):
    return

