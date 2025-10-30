import matplotlib
# matplotlib.use('Agg')


import optuna
import neat
import numpy as np
import Settings
import Lenia.lenia_fitness as fit
import Visualizer as vs
import matplotlib.pyplot as plt
import gc

GRID_SIZE_X = Settings.DIM_X
GRID_SIZE_Y = Settings.DIM_Y
GRID_SIZE_C = Settings.DIM_C

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

def evaluate_genome(genomes, config) -> float:
    for genome_id, genome in genomes:
        nn = create_FFNeat(genome, config)
        pattern = generate_pattern(nn, GRID_SIZE_X-10, GRID_SIZE_Y-10, GRID_SIZE_C)
        genome.fitness = fit.fitness_main(pattern)
        
        # Memory management
        plt.close('all')
        gc.collect()

def run_neat(trial):
    # Suggest hyperparameters
    params = {
        'pop_size': trial.suggest_int('pop_size', 10, 50),
        'weight_mutate_rate': trial.suggest_float('weight_mutate_rate', 0.1, 0.9),
        'weight_replace_rate': trial.suggest_float('weight_replace_rate', 0.0, 0.5),
        'bias_mutate_rate': trial.suggest_float('bias_mutate_rate', 0.1, 0.9),
        'response_mutate_rate': trial.suggest_float('response_mutate_rate', 0.0, 0.5),
        'conn_add_prob': trial.suggest_float('conn_add_prob', 0.1, 0.9),
        'conn_delete_prob': trial.suggest_float('conn_delete_prob', 0.1, 0.5),
        'node_add_prob': trial.suggest_float('node_add_prob', 0.0, 0.5),
        'node_delete_prob': trial.suggest_float('node_delete_prob', 0.0, 0.5),
        'enabled_mutate_rate': trial.suggest_float('enabled_mutate_rate', 0.0, 0.2),
        'compatibility_threshold': trial.suggest_float('compatibility_threshold', 1.0, 5.0),
    }

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'neat_config')
    config.pop_size = params['pop_size']
    config.genome_config.weight_mutate_rate = params['weight_mutate_rate']
    config.genome_config.weight_replace_rate = params['weight_replace_rate']
    config.genome_config.bias_mutate_rate = params['bias_mutate_rate']
    config.genome_config.response_mutate_rate = params['response_mutate_rate']
    config.genome_config.conn_add_prob = params['conn_add_prob']
    config.genome_config.conn_delete_prob = params['conn_delete_prob']
    config.genome_config.node_add_prob = params['node_add_prob']
    config.genome_config.node_delete_prob = params['node_delete_prob']
    config.genome_config.enabled_mutate_rate = params['enabled_mutate_rate']
    config.compatibility_threshold = params['compatibility_threshold']

    pop = neat.Population(config)
    winner = pop.run(evaluate_genome, 5)
    return winner.fitness

def objective(trial):
    return run_neat(trial)

if __name__ == "__main__":
    gc.enable()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=7200)  # 20 trials, 1 hour timeout

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Fitness): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
