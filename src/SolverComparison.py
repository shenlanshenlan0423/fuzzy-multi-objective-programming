# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/9 21:31
@Auth ： Hongwei
@File ：SolverComparison.py
@IDE ：PyCharm
"""
from definitions import *
import time


def run_varying_solver(solver_names):
    time_lists = []
    for _ in range(10):
        time_list = []
        for solver_name in solver_names:
            time_start = time.time()
            cmd = 'python -um src.main --solver {} --flag {}'.format(solver_name, 'Solve')
            os.system(cmd)
            time_end = time.time()
            time_list.append((time_end - time_start) * 1000)
        time_lists.append(time_list)
    time_cost = []
    for _ in range(len(solver_names)):
        time_values = np.array(time_lists)[:, _]
        time_cost.append('{:.2f}±{:.2f}'.format(np.mean(time_values), np.std(time_values)))
    return time_cost


if __name__ == '__main__':
    solver_names = ['CBC', 'GLPK', 'GUROBI']
    time_cost = run_varying_solver(solver_names)
    time_table = pd.DataFrame(np.array([['Time cost/ms']+time_cost]), columns=['*']+solver_names)
    time_table.to_excel(TABLE_DIR + '/time_cost_table.xlsx', index=False)
    print(time_table.to_markdown())
