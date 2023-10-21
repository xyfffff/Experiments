import os
import numpy as np
from concorde.tsp import TSPSolver
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import multiprocessing

class GenerateTSP(object):
    def __init__(self, num_samples, num_cities, folder_name):
        self.folder_name = folder_name
        os.makedirs(self.folder_name, exist_ok=True)
        self.data = []
        self.labels = []
        self.generate_tsp_samples(num_samples, num_cities)

    def generate_tsp_samples(self, num_samples, num_cities):
        current_dir = os.getcwd()
        os.chdir(self.folder_name)

        devnull = open(os.devnull, 'w')
        old_stdout = os.dup(1)
        os.dup2(devnull.fileno(), 1)

        for _ in tqdm(range(num_samples), desc="Generating TSP samples"):
            cities = np.random.rand(num_cities, 2)
            solver = TSPSolver.from_data(cities[:, 0], cities[:, 1], norm="EUC_2D")
            solution = solver.solve()
            self.data.append(cities)
            self.labels.append(np.array(solution.tour)[:, None])

        os.dup2(old_stdout, 1)
        devnull.close()

        torch.save((self.data, self.labels), 'data.pth')

        os.chdir(current_dir)


def create_tsp_data(num_samples, num_cities, folder_name):
    _ = GenerateTSP(num_samples, num_cities, folder_name)

if __name__ == "__main__":
    total_samples = 1280000
    num_cities = 50
    num_cores = 112
    samples_per_core = total_samples // num_cores

    processes = []
    for i in range(num_cores):
        folder_name = f'core_{i}'
        p = multiprocessing.Process(target=create_tsp_data, args=(samples_per_core, num_cities, folder_name))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
