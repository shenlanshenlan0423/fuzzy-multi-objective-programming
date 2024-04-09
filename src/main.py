# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/7 11:27
@Auth ： Hongwei
@File ：main.py
@IDE ：PyCharm
"""
from definitions import *
from src.model.MOP import MOP


class FMOP(MOP):
    def __init__(self, args, policy, weights, objective_matrix):
        super().__init__(args, policy, weights)
        self.objective_matrix = objective_matrix
        self.objective_best = np.max(objective_matrix, axis=0)
        self.objective_worst = np.min(objective_matrix, axis=0)
        self.delta = self.objective_best - self.objective_worst

    def solve(self):
        k = 1
        while 1:
            # Terminating condition
            if np.nansum(self.demand) < self.epsilon or k > self.k:
                break
            k_tour_delivery_table = self.get_k_tour_delivery_table()
            demand = np.nansum(self.demand, axis=0)
            problem, S, U, C, V, Q_21, Q_31, Q_22, Q_32, Q_33, Q_24, Q_25, Q_35, Q_16, Q_37 = \
                self.initialize_prob(k_tour_delivery_table, demand, k)

            # additional decision variable
            lamb = LpVariable("lamb")
            # define objective function
            problem += lamb, "Maximize new objectives"

            if self.delta[0] != 0:
                problem += U - self.delta[0] * lamb >= self.objective_worst[0]
            if self.delta[1] != 0:
                problem += C + self.delta[1] * lamb <= self.objective_best[1]
            if self.delta[2] != 0:
                problem += V + self.delta[2] * lamb <= self.objective_best[2]

            # Solve problem
            status = problem.solve()
            self.save_res(S, U, C, V, Q_21, Q_31, Q_22, Q_32, Q_33, Q_24, Q_25, Q_35, Q_16, Q_37, k_tour_delivery_table, demand, k)
            k += 1

    # def print_res(self):
    #     Delivery = self.res_table1.iloc[:, 1:].sum(axis=1)
    #     self.res_table1['Demand'] = self.initial_demand.tolist() + ['*' for _ in range(3)]
    #     self.res_table1['Delivery'] = Delivery
    #     print(self.res_table1.to_markdown(index=False))
    #     self.res_table2['Total'] = self.res_table2.iloc[:, 1:].sum(axis=1)
    #     print(self.res_table2.to_markdown(index=False))


def get_configs(beta_ms, beta_v1):
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Settings")
    parser.add_argument('--f_max', type=int, default=800, help='Traffic flow boundary')
    parser.add_argument('--c', type=int, default=5, help='Carrying capacity per vehicle')
    parser.add_argument('--c_p', type=int, default=1000, help='Procurement cost of per ton of relief materials')
    parser.add_argument('--c_tr', type=list, default=[100, 200, 200], help='Transportation cost per ton of each routes')
    parser.add_argument('--c_m', type=int, default=100, help='Medical service cost of per injured people')
    parser.add_argument('--c_v', type=int, default=100, help='Order maintaining cost of per unsatisfied recipient')
    parser.add_argument('--b_region22', type=float, default=2, help='Road damage coefficient, region 2 to 2')
    parser.add_argument('--b_region21', type=float, default=2.3, help='Road damage coefficient, region 2 to 1')
    parser.add_argument('--b_region11', type=float, default=2.5, help='Road damage coefficient, region 1 to 1')
    parser.add_argument('--mu', type=float, default=0.9, help='Over-saturated adjustment factor')
    parser.add_argument('--q', type=int, default=1, help='Each victim’s demand for total relief supplies is 1 ton')
    parser.add_argument('--e_m', type=int, default=2, help='Each injured victim is equipped with 1 doctor and 1 nurse')
    parser.add_argument('--n_g', type=float, default=0.1, help='Every 10 unsatisfied victims is equipped with 1 guard')
    parser.add_argument('--beta_ms', type=float, default=beta_ms, help='Proportion of serious injured recipients')
    parser.add_argument('--beta_v1', type=float, default=beta_v1, help='Proportion of victims who feel unsatisfied')
    parser.add_argument('--beta_v2', type=float, default=0.2, help='Proportion of victims who feel satisfied')
    parser.add_argument('--x_max', type=int, default=2000, help='Maximum available truck')

    parser.add_argument('--epsilon', type=float, default=0.01, help='To determine the termination condition')
    parser.add_argument('--k', type=int, default=6, help='Max iteration')
    parser.add_argument('--flag', type=str, default='Solve', help='Solve, SensitivityAnalysis_beta_ms or SensitivityAnalysis_beta_v1')
    parser.add_argument('--solver', type=str, default='CBC', help='Solver configuration of pulp library')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_configs(beta_ms=0.2, beta_v1=0.5)
    if args.flag == 'Solve':
        for policy in ['Spontaneous', 'GovernmentRegulation']:
            objective_matrix = []
            for weights in [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]:
                mop = MOP(args, policy, weights)
                mop.solve()
                res_table = mop.get_res()
                objective_matrix.append(res_table.iloc[2:, -1].tolist())
            pd.DataFrame(np.array(objective_matrix),
                         columns=['U', 'C', 'V']).to_excel(TABLE_DIR+'policy-{}-objective_matrix.xlsx'.format(policy))
            fmop = FMOP(args, policy, weights=None, objective_matrix=np.array(objective_matrix))
            fmop.solve()
            fmop.print_res()

    policy = 'GovernmentRegulation'
    objective_matrix = []
    if args.flag == 'SensitivityAnalysis_beta_ms':
        for weights in [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]:
            mop = MOP(args, policy, weights)
            mop.solve()
            res_table = mop.get_res()
            objective_matrix.append(res_table.iloc[2:, -1].tolist())
        fmop = FMOP(args, policy, weights=None, objective_matrix=np.array(objective_matrix))
        fmop.solve()
        res_list = fmop.get_metrics().tolist()
        beta_ms_arr = load_pickle(RESULT_DIR + '/beta_ms_arr.pickle')
        # If the delivery is completed in less than 5 tours under some parameter Settings,
        # it is completed to 5 tours for comparison
        if len(res_list) == 7:
            res_list.insert(3, 0)
            res_list.insert(4, 0)
            res_list[-3] += 7  # For the policy that have been delivered in k-1 tour, the satisfaction is 1*7 in k tour
        elif len(res_list) == 8:
            res_list.insert(4, 0)
            res_list[-3] += 7
        beta_ms_arr.append(res_list)
        save_pickle(beta_ms_arr, RESULT_DIR + '/beta_ms_arr.pickle')
        pass

    if args.flag == 'SensitivityAnalysis_beta_v1':
        for weights in [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]:
            mop = MOP(args, policy, weights)
            mop.solve()
            res_table = mop.get_res()
            objective_matrix.append(res_table.iloc[2:, -1].tolist())
        fmop = FMOP(args, policy, weights=None, objective_matrix=np.array(objective_matrix))
        fmop.solve()
        res_list = fmop.res_table2.iloc[4, :].tolist()[1:]
        beta_v1_arr = load_pickle(RESULT_DIR + '/beta_v1_arr.pickle')
        beta_v1_arr.append(res_list)
        save_pickle(beta_v1_arr, RESULT_DIR + '/beta_v1_arr.pickle')
