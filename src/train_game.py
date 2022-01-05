import subprocess
import os
import neat

class Player:
    def __init__(self, net: neat.nn.FeedForwardNetwork) -> None:
        self.net = net
        # TODO: Each player has a perceived game state


def sim_battle(net1: neat.nn.FeedForwardNetwork, net2: neat.nn.FeedForwardNetwork) -> float:
    # TODO
    # - Probably have each player on its own thread, battle running on main thread
    pass

def eval_genomes(pop1: neat.Population, pop2: neat.Population) -> None:
    genomes1 = list(iteritems(pop1.population))
    genomes2 = list(iteritems(pop2.population))
    
    for genome_id1, genome1 in genomes1:
        genome1.fitness = 0.0
    
    for genome_id2, genome2 in genomes2:
        genome2.fitness = 0.0
    
    for genome_id1, genome1 in genomes1:
        net1 = neat.nn.FeedForwardNetwork.create(genome1, pop1.config)
        for genome_id2, genome2 in genomes2:
            net2 = neat.nn.FeedForwardNetwork.create(genome2, pop2.config)
            battle_value = sim_battle(net1, net2)
            genome1.fitness += battle_value
            genome2.fitness -= battle_value
    
    for genome_id1, genome1 in genomes1:
        genome1.fitness /= len(genomes2)
    
    for genome_id2, genome2 in genomes2:
        genome2.fitness /= len(genomes1)

def update_pop(pop: neat.Population) -> bool:
    # Gather and report statistics.
    best = None
    for g in itervalues(pop.population):
        if best is None or g.fitness > best.fitness:
            best = g
    pop.reporters.post_evaluate(pop.config, pop.population, pop.species, best)

    # Track the best genome ever seen.
    if pop.best_genome is None or best.fitness > pop.best_genome.fitness:
        pop.best_genome = best

    if not pop.config.no_fitness_termination:
        # End if the fitness threshold is reached.
        fv = pop.fitness_criterion(g.fitness for g in itervalues(pop.population))
        if fv >= pop.config.fitness_threshold:
            pop.reporters.found_solution(pop.config, pop.generation, best)
            return True

    # Create the next generation from the current generation.
    pop.population = pop.reproduction.reproduce(pop.config, pop.species,
                                                  pop.config.pop_size, pop.generation)

    # Check for complete extinction.
    if not pop.species.species:
        pop.reporters.complete_extinction()

        # If requested by the user, create a completely new population,
        # otherwise raise an exception.
        if pop.config.reset_on_extinction:
            pop.population = pop.reproduction.create_new(pop.config.genome_type,
                                                           pop.config.genome_config,
                                                           pop.config.pop_size)
        else:
            raise CompleteExtinctionException()

    # Divide the new population into species.
    pop.species.speciate(pop.config, pop.population, pop.generation)

    pop.reporters.end_generation(pop.config, pop.population, pop.species)

    pop.generation += 1
    return False

def train(pop1: neat.Population, pop2: neat.Population, n: int = 0) -> None:
    if (pop1.config.no_fitness_termination or pop2.config.no_fitness_termination) and (n <= 0):
        raise RuntimeError("Cannot have no generational limit with no fitness termination")
    
    k = 0
    while n <= 0 or k < n:
        k += 1
        
        pop1.reporters.start_generation(pop1.generation)
        pop2.reporters.start_generation(pop2.generation)
        
        eval_genomes(pop1, pop2)
        
        if update_pop(pop1) or update_pop(pop2):
            break
    
    if pop1.config.no_fitness_termination:
        pop1.reporters.found_solution(pop1.config, pop1.generation, pop1.best_genome)
    
    if pop2.config.no_fitness_termination:
        pop2.reporters.found_solution(pop2.config, pop2.generation, pop2.best_genome)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_train_game.ini')
    
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the populations, which are the top-level objects for a NEAT run.
    pop1 = neat.Population(config)
    pop2 = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop1.add_reporter(neat.StdOutReporter(True))
    pop1.add_reporter(neat.StatisticsReporter())
    #pop1.add_reporter(neat.Checkpointer(5))
    pop2.add_reporter(neat.StdOutReporter(True))
    pop2.add_reporter(neat.StatisticsReporter())
    #pop2.add_reporter(neat.Checkpointer(5))

    # Train for up to n generations.
    train(pop1, pop2, 10)

    # Display the winning genomes.
    print('\nBest genome of pop 1:\n{!s}'.format(pop1.best_genome))
    print('\nBest genome of pop 2:\n{!s}'.format(pop2.best_genome))
