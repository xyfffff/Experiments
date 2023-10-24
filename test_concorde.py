import numpy as np
from concorde.tsp import TSPSolver
import time

def generate_points(n, mean=0, std_dev=1, scaler=10000):
    points = np.random.randn(n, 2) * std_dev + mean
    return points * scaler

def solve_tsp(points):
    solver = TSPSolver.from_data(points[:, 0], points[:, 1], norm="EUC_2D")
    solution = solver.solve()
    return solution.optimal_value

def main(N, n, mean=0, std_dev=1, scaler=10000):
    """
        N: Number of samples
        n: Number of points
    """

    total_length = 0
    start_time = time.time()
    for _ in range(N):
        points = generate_points(n, mean, std_dev, scaler)
        length = solve_tsp(points)
        total_length += length
        print(f"Sample {_+1}, Path Length: {length / scaler}")

    average_length = total_length / N
    print(f"\nAverage Path Length: {average_length / scaler}")
    print(f"Total time taken: {time.time() - start_time} seconds")

if __name__ == "__main__":
    N = 1000
    n = 20
    mean = 0
    std_dev = 1000

    main(N, n, mean, std_dev)


# =============================
# | mean = 0 | std_dev = 1000 |
# 14870
