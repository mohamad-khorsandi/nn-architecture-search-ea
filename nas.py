import os
import random
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np

import utils.arc_utils
from utils import arc_utils
from utils.evolution_utils import parent_selection, recombination, mutation


class NAS:
    def __init__(self, iteration, population_size, parent_pool_size, p_rec, p_mut, mut_step):
        assert parent_pool_size <= population_size
        assert parent_pool_size % 2 == 0

        self.iteration = iteration
        self.population_size = population_size
        self.parent_pool_size = parent_pool_size
        self.p_rec = p_rec
        self.p_mut = p_mut
        self.population = list()
        self.gen_rand_population()
        self.mut_step = mut_step

        self.fitness_hist = list()
        self.best_hist = list()
        self.duration = float()
        self.searched_arcs = list()

        self.res_dir = 'res_dir'
        dir_exists = os.path.exists(self.res_dir)
        if dir_exists:
            shutil.rmtree(self.res_dir)
        os.mkdir(self.res_dir)

    def run(self):
        start = time.time()
        for i in range(self.iteration):
            print('epoch----------------------', i)
            self.recode_statistics()
            parent_pool = parent_selection(self.population, self.parent_pool_size)
            children = []

            for j in range(0, self.parent_pool_size - 1, 2):
                p1, p2 = parent_pool[j], parent_pool[j + 1]
                for rec_result in recombination(p1, p2, self.p_rec):
                    mut_result = mutation(rec_result, self.p_mut, self.mut_step)
                    children.append(mut_result)

            print('\npopulation')
            self.show_arc_list(self.population)
            self.survival_selection(children)

        end = time.time()
        self.duration = end - start

    def gen_rand_population(self):
        for i in range(self.population_size):
            rand_arc = arc_utils.get_random_arc()
            self.population.append(rand_arc)

    def recode_statistics(self):
        all_fitness = np.array([c.get_fitness() for c in self.population])
        self.fitness_hist.append(np.mean(all_fitness))
        self.best_hist.append(self.population[all_fitness.argmax()])
        print(arc_utils.signature(self.population[all_fitness.argmax()]))

    def survival_selection(self, children):
        tmp_list = self.population + children
        tmp_list.sort(key=lambda x: x.get_fitness())
        tmp_list = tmp_list[len(children):]

        random.shuffle(tmp_list)
        self.population = tmp_list

    def show_statistics(self):
        print("total time of running algorithm: " + str(self.duration))
        plt.clf()
        plt.plot(range(self.iteration), self.fitness_hist, color='b')
        plt.plot(range(self.iteration), [c.get_fitness() for c in self.best_hist], color='r')
        plt.savefig(self.res_dir + '/' + 'total.png')

        print('best arc:')
        print(arc_utils.signature(self.best_hist[self.iteration - 1]))

    def show_arc_list(self, arc_list):
        res = ''
        for arc in arc_list:
            res += utils.arc_utils.signature(arc) + ', '

        print(res)

