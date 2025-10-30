##NEAT implementation

###################################################################################
''' imports '''
import matplotlib.pyplot as plt
import numpy as np
import neat
import gc
###################################################################################
''' custom imports'''

import Settings
import Settings as Settings
import Lenia.lenia_fitness as fit
import Visualizer as vs


###################################################################################
''' globals '''

GRID_SIZE_X = Settings.DIM_X
GRID_SIZE_Y = Settings.DIM_Y
GRID_SIZE_C = Settings.DIM_C

###################################################################################

def create_FFNeat(genome, config):
    return neat.nn.FeedForwardNetwork.create(genome, config)

def get_weight(cppn, x1, y1, z1, x2, y2, z2):
    inputs = [x1, y1, z1, x2, y2, z2]
    return cppn.activate(inputs)[0]

def generate_pattern(cppn, grid_size_x, grid_size_y, grid_size_c):
    pattern = np.zeros((grid_size_x, grid_size_y, grid_size_c))
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            for z in range(grid_size_c):
                if x < grid_size_x - 1:
                    w = get_weight(cppn, x/grid_size_x, y/grid_size_y, z/grid_size_c,
                                   (x+1)/grid_size_x, y/grid_size_y, z/grid_size_c)
                    pattern[x, y, z] = w
    return pattern

def evaluate_genomes(genomes, config) -> None:
    
    for genome_id, genome in genomes:
        nn = create_FFNeat(genome, config)
        pattern = generate_pattern(nn, GRID_SIZE_X-10, GRID_SIZE_Y-10, GRID_SIZE_C)
        genome.fitness = fit.fitness_main(pattern)
        
        # Memory management
        plt.close('all')
        gc.collect()


def main():
    config_path = 'neat_config'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    winner = pop.run(evaluate_genomes, 10)
    
    res = create_FFNeat(winner, config)
    pattern = generate_pattern(res, GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_C)
    vs.plt_show(pattern)
    
if __name__ == "__main__":
    main()
    
    
    





