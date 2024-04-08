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
            omega = self.get_invulnerability(k_tour_delivery_table)
            demand = np.nansum(self.demand, axis=0)

            # Instantiate the Max optimization problem
            problem = LpProblem("MyProblem", LpMaximize)
            # define decision variables
            Q_21 = LpVariable("Q_21", lowBound=0)
            Q_31 = LpVariable("Q_31", lowBound=0)
            Q_22 = LpVariable("Q_22", lowBound=0)
            Q_32 = LpVariable("Q_32", lowBound=0)
            Q_33 = LpVariable("Q_33", lowBound=0)
            Q_24 = LpVariable("Q_24", lowBound=0)
            Q_25 = LpVariable("Q_25", lowBound=0)
            Q_35 = LpVariable("Q_35", lowBound=0)
            Q_16 = LpVariable("Q_16", lowBound=0)
            Q_37 = LpVariable("Q_37", lowBound=0)
            # additional decision variable
            lamb = LpVariable("lamb")

            # define objective function
            problem += lamb, "Maximize new objectives"

            # Capacity Constraint equation (3)
            problem += Q_16 == k_tour_delivery_table[4, 0]
            problem += Q_21 + Q_22 + Q_24 + Q_25 == k_tour_delivery_table[4, 1]
            problem += Q_31 + Q_32 + Q_33 + Q_35 + Q_37 == k_tour_delivery_table[4, 2]
            # equation (4)
            problem += Q_21 <= demand[0]
            problem += Q_31 <= demand[0]
            problem += Q_22 <= demand[1]
            problem += Q_32 <= demand[1]
            problem += Q_33 <= demand[2]
            problem += Q_24 <= demand[3]
            problem += Q_25 <= demand[4]
            problem += Q_35 <= demand[4]
            problem += Q_16 <= demand[5]
            problem += Q_37 <= demand[6]
            # equation (5)
            problem += Q_21 + Q_31 <= demand[0]
            problem += Q_22 + Q_32 <= demand[1]
            problem += Q_33 <= demand[2]
            problem += Q_24 <= demand[3]
            problem += Q_25 + Q_35 <= demand[4]
            problem += Q_16 <= demand[5]
            problem += Q_37 <= demand[6]

            # "Maximize cured serious injured recipients"
            S = self.get_serious_injured_recipients(actual_delivery=k_tour_delivery_table[4, -1], k=k)
            # A location with demand of 0 has U of 1
            U_m_list = [(Q_21 + Q_31) * (1 / demand[0]),
                        (Q_22 + Q_32) * (1 / demand[1]),
                        Q_33 * (1 / demand[2]),
                        Q_24 * (1 / demand[3]),
                        (Q_25 + Q_35) * (1 / demand[4]),
                        Q_16 * (1 / demand[5]),
                        Q_37 * (1 / demand[6])]
            non_zero_idx = np.where(np.array(demand) != 0)[0]
            # "Maximize Satisfaction degree"
            U = sum([U_m_list[i] for i in non_zero_idx], 1 * (7 - len(non_zero_idx)))
            # "Minimize Operation costs"
            C = self.c_p * (Q_21 + Q_31 + Q_22 + Q_32 + Q_33 + Q_24 + Q_25 + Q_35 + Q_16 + Q_37) \
                + sum(self.c_tr * np.array([Q_16, Q_21 + Q_22 + Q_24 + Q_25, Q_31 + Q_32 + Q_33 + Q_35 + Q_37]) / self.c) \
                + self.c_m * self.e_m * S
            # "Minimize Order maintaining costs"
            V = self.get_order_maintaining_cost(n=demand[0], Q=Q_21 + Q_31, last_U=self.last_U_list[0]) + \
                self.get_order_maintaining_cost(n=demand[1], Q=Q_22 + Q_32, last_U=self.last_U_list[1]) + \
                self.get_order_maintaining_cost(n=demand[2], Q=Q_33, last_U=self.last_U_list[2]) + \
                self.get_order_maintaining_cost(n=demand[3], Q=Q_24, last_U=self.last_U_list[3]) + \
                self.get_order_maintaining_cost(n=demand[4], Q=Q_25 + Q_35, last_U=self.last_U_list[4]) + \
                self.get_order_maintaining_cost(n=demand[5], Q=Q_16, last_U=self.last_U_list[5]) + \
                self.get_order_maintaining_cost(n=demand[6], Q=Q_37, last_U=self.last_U_list[6])

            if self.delta[0] != 0:
                problem += U - self.delta[0] * lamb >= self.objective_worst[0]
            if self.delta[1] != 0:
                problem += C + self.delta[1] * lamb <= self.objective_best[1]
            if self.delta[2] != 0:
                problem += V + self.delta[2] * lamb <= self.objective_best[2]

            if S >= sum(demand * self.beta_ms):
                # Try to meet the needs of all seriously injured people in the first tour
                problem += Q_21 + Q_31 >= demand[0]*self.beta_ms
                problem += Q_22 + Q_32 >= demand[1]*self.beta_ms
                problem += Q_33 >= demand[2]*self.beta_ms
                problem += Q_24 >= demand[3]*self.beta_ms
                problem += Q_25 + Q_35 >= demand[4]*self.beta_ms
                problem += Q_16 >= demand[5]*self.beta_ms
                problem += Q_37 >= demand[6]*self.beta_ms

            # Solve problem
            status = problem.solve()
            # Table for saving result
            res_table = np.zeros((10, 3))
            # route 1
            res_table[5, 0] = value(Q_16)
            # route 2
            res_table[0, 1] = value(Q_21)
            res_table[1, 1] = value(Q_22)
            res_table[3, 1] = value(Q_24)
            res_table[4, 1] = value(Q_25)
            # route 3
            res_table[0, 2] = value(Q_31)
            res_table[1, 2] = value(Q_32)
            res_table[2, 2] = value(Q_33)
            res_table[4, 2] = value(Q_35)
            res_table[6, 2] = value(Q_37)

            # update the demand table (Table.3)
            self.last_U_list = [(value(Q_21) + value(Q_31)) * (1 / demand[0]),
                                + (value(Q_22) + value(Q_32)) * (1 / demand[1]),
                                + value(Q_33) * (1 / demand[2]),
                                + value(Q_24) * (1 / demand[3]),
                                + (value(Q_25) + value(Q_35)) * (1 / demand[4]),
                                + value(Q_16) * (1 / demand[5]),
                                + value(Q_37) * (1 / demand[6])]
            self.last_U_underline = np.median(self.last_U_list)  # the median value of U for calculate V
            self.last_n = demand
            self.last_Q = res_table[:7, :].sum(axis=1)

            unmet_demand = self.last_n - self.last_Q - self.died_demand
            new_demand = np.full((3, 7), np.nan)
            each_demand = unmet_demand / np.sum(~np.isnan(self.demand), axis=0)
            for col_idx in range(new_demand.shape[1]):
                initial_col = self.demand[:, col_idx]
                initial_col[~np.isnan(initial_col)] = each_demand[col_idx]
                new_demand[:, col_idx] = initial_col
            # There can be no demand less than 0
            new_demand[new_demand < 0] = 0
            self.demand = new_demand

            res_table[7, :] = k_tour_delivery_table[4, :-1]
            res_table[8:, :] = k_tour_delivery_table[2:4, :-1]
            df_res = pd.DataFrame(res_table, columns=['r1', 'r2', 'r3'])
            self.res_table1 = pd.concat(([self.res_table1, df_res.round(2)]), axis=1)
            self.res_table2 = pd.concat(([self.res_table2, pd.DataFrame([omega, S, min(value(U), 7),
                                                                         max(0, value(C)/1000), max(0, value(V)/1000)],
                                                                        columns=['K={}'.format(k)]).round(2)]), axis=1)
            k += 1


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
    parser.add_argument('--beta_ms', type=float, default=beta_ms, help='Proportion of serious injured recipients/victims')
    parser.add_argument('--beta_v1', type=float, default=beta_v1, help='Proportion of victims who feel unsatisfied')
    parser.add_argument('--beta_v2', type=float, default=0.2, help='Proportion of victims who feel satisfied')
    parser.add_argument('--x_max', type=int, default=2000, help='Maximum available truck')

    parser.add_argument('--epsilon', type=float, default=0.01, help='To determine the termination condition')
    parser.add_argument('--k', type=int, default=5, help='Max iteration')
    parser.add_argument('--flag', type=str, default='Solve', help='Solve or SensitivityAnalysis')
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

    if args.flag == 'SensitivityAnalysis_beta_ms':
        policy = 'GovernmentRegulation'
        objective_matrix = []
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
        if len(res_list) < 9:
            res_list.insert(4, 0)
            res_list[-3] += 7  # For the policy that have been delivered in k-1 tour, the satisfaction is 1*7 in k tour
        beta_ms_arr.append(res_list)
        save_pickle(beta_ms_arr, RESULT_DIR + '/beta_ms_arr.pickle')

    if args.flag == 'SensitivityAnalysis_beta_v1':
        policy = 'GovernmentRegulation'
        objective_matrix = []
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
