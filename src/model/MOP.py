# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/29 17:55
@Auth ： Hongwei
@File ：MOP.py
@IDE ：PyCharm
"""
from definitions import *


class MOP:
    def __init__(self, args, policy, weights):
        # The initial demand table, update in iteration
        self.demand = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, 5000, np.nan],
                                [2500, 2500, np.nan, 5000, 3000, np.nan, np.nan],
                                [2500, 2500, 5000, np.nan, 3000, np.nan, 5000]])
        self.initial_demand = np.nansum(self.demand, axis=0)
        # For calculating died people in each demand point
        self.demand_ratio = self.initial_demand / self.initial_demand.sum()
        # Current serious injured recipients
        self.serious_injured_recipients = self.initial_demand * args.beta_ms
        self.dead_flag = False
        self.dead_crowd = 0
        # Params
        self.mu = args.mu
        self.f_max = args.f_max
        self.x_max = args.x_max
        self.beta_v1 = args.beta_v1
        self.beta_v2 = args.beta_v2
        self.q = args.q
        self.beta_ms = args.beta_ms
        self.c_p = args.c_p
        self.c_tr = args.c_tr
        self.c = args.c
        self.c_m = args.c_m
        self.e_m = args.e_m
        self.c_v = args.c_v
        self.n_g = args.n_g
        self.b_region11 = args.b_region11
        self.b_region21 = args.b_region21
        self.b_region22 = args.b_region22
        self.epsilon = args.epsilon

        self.K = args.K
        self.weights = weights
        # NPO’s spontaneous relief scheme or Traffic control scheme with government regulation policy
        self.policy = policy
        # For calculating equity objective V, update in iteration
        self.last_U_list = [0 for _ in range(7)]
        self.last_U_underline = 0
        self.last_n = [0 for _ in range(7)]
        self.last_Q = [0 for _ in range(7)]
        # Tables for saving results
        self.res_table1 = pd.DataFrame()
        self.res_table1['***'] = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7',
                                  'Total delivery', 'Truck delivery', 'Actual truck delivery']
        self.res_table2 = pd.DataFrame()
        self.res_table2['***'] = ['Omega', 'S', 'U', 'C', 'V']
        # solver setting
        self.solver_name = args.solver

    def get_solver(self, solver_name):
        if solver_name == 'CBC':
            # Please replace the path of cbc.exe in your device
            return pulp.COIN_CMD(path=r'D:\anaconda3\lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe')
            # return pulp.COIN_CMD(path=r'D:\Anaconda\lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe')
        if solver_name == 'GLPK':
            # Please replace the path of glpsol.exe in your device
            return pulp.GLPK(path=r'D:\anaconda3\lib\site-packages\pulp\solverdir\glpk\w64\glpsol.exe')
        if solver_name == 'GUROBI':
            # Please install the gurobipy library in Python
            return pulp.GUROBI()

    def get_serious_injured_recipients(self, actual_delivery, k):
        # Whether the total distribution volume of the first tour can meet
        # the demand of the total seriously injured victims
        # equation (2)
        S = min(actual_delivery, sum(self.serious_injured_recipients))
        if k == 1 and S <= sum(self.serious_injured_recipients):
            # People with severe injuries whose needs were not met in the first tour will die in the second tour
            self.dead_flag = True
        else:
            self.dead_flag = False
        return S

    def get_order_maintaining_cost(self, n, Q, last_U):
        # equation (9)
        unmet_demand = (n - Q * (1 / self.q))
        if last_U <= self.last_U_underline:
            return self.c_v * self.beta_v1 * unmet_demand * self.n_g
        else:
            return self.c_v * self.beta_v2 * unmet_demand * self.n_g

    def get_invulnerability(self, delivery_table):
        # equation (10)
        demand_and_delivery = delivery_table[[0, 4], :-1]
        # Reference from Wenchuan map data, corresponds to Figure 1(b) and equation (13)
        zeta = [
            # Southern line
            1 / (self.b_region21 * 80) / (1 / 80),
            # Eastern line
            (1 / (self.b_region22 * 568) + 1 / (self.b_region21 * 44)
             + 1 / (self.b_region11 * 4.5) + 1 / (self.b_region11 * 0.5) + 1 / (self.b_region11 * 10))
            / (1 / 568 + 1 / 44 + 1 / 4.5 + 1 / 0.5 + 1 / 10),
            # Western line
            (1 / (self.b_region22 * 290) + 1 / (self.b_region22 * 353)
             + 1 / (self.b_region21 * 68) + 1 / (self.b_region21 * 59.2)
             + 1 / (self.b_region11 * 0.5) + 1 / (self.b_region11 * 0.5) + 1 / (self.b_region11 * 10))
            / (1 / 290 + 1 / 353 + 1 / 68 + 1 / 59.2 + 1 / 0.5 + 1 / 0.5 + 1 / 10)
        ]
        return sum(zeta * demand_and_delivery.min(axis=0))

    def get_actual_traffic(self, x):
        # equation (14)
        if self.policy == 'Spontaneous':
            if 0 <= x <= self.f_max:
                return x
            elif self.f_max < x <= self.x_max:
                return self.mu * self.f_max * (self.x_max - x) / (self.x_max - self.f_max)
            elif x > self.x_max:
                return 0
        # equation (15)
        elif self.policy == 'GovernmentRegulation':
            return min(x, self.f_max)

    def get_k_tour_delivery_table(self):
        demand_with_route = np.nansum(self.demand, axis=1)
        demand_ratio = demand_with_route / demand_with_route.sum()
        # Dynamically adjust the distribution volume according to the remaining demand
        # whether the total demand exceeds the maximum number of available trucks
        if demand_with_route.sum() > self.x_max * self.c:
            TruckDelivery = self.x_max * demand_ratio
        else:
            TruckDelivery = demand_with_route.sum() / self.c * demand_ratio
        ActualTruckDelivery = np.array([self.get_actual_traffic(i) for i in TruckDelivery])
        TotalDelivery = ActualTruckDelivery * self.c
        k_tour_delivery_table = np.vstack((demand_with_route, demand_ratio, TruckDelivery,
                                           ActualTruckDelivery, TotalDelivery))
        return np.hstack((k_tour_delivery_table, k_tour_delivery_table.sum(axis=1).reshape(-1, 1)))

    def initialize_prob(self, k_tour_delivery_table, demand, k):
        # Instantiate the Maximize optimization problem
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

        # Define objective function
        # Maximize cured serious injured recipients
        S = self.get_serious_injured_recipients(actual_delivery=k_tour_delivery_table[4, -1], k=k)
        # A location with demand of 0 has U of 1
        U_m_list = [(Q_21 + Q_31) * (1 / demand[0]),
                    (Q_22 + Q_32) * (1 / demand[1]),
                    Q_33 * (1 / demand[2]),
                    Q_24 * (1 / demand[3]),
                    (Q_25 + Q_35) * (1 / demand[4]),
                    Q_16 * (1 / demand[5]),
                    Q_37 * (1 / demand[6])]
        non_zero_idx = np.where(np.array(demand) > self.epsilon)[0]  # considering the float error of Python
        # Maximize Satisfaction degree
        U = sum([U_m_list[i] for i in non_zero_idx], 1 * (7 - len(non_zero_idx)))
        # Minimize Operation costs
        C = self.c_p * k_tour_delivery_table[4, 3] \
            + sum(self.c_tr * k_tour_delivery_table[3, :3]) \
            + self.c_m * self.e_m * S
        # Minimize Order maintaining costs
        V = self.get_order_maintaining_cost(n=demand[0], Q=Q_21 + Q_31, last_U=self.last_U_list[0]) + \
            self.get_order_maintaining_cost(n=demand[1], Q=Q_22 + Q_32, last_U=self.last_U_list[1]) + \
            self.get_order_maintaining_cost(n=demand[2], Q=Q_33, last_U=self.last_U_list[2]) + \
            self.get_order_maintaining_cost(n=demand[3], Q=Q_24, last_U=self.last_U_list[3]) + \
            self.get_order_maintaining_cost(n=demand[4], Q=Q_25 + Q_35, last_U=self.last_U_list[4]) + \
            self.get_order_maintaining_cost(n=demand[5], Q=Q_16, last_U=self.last_U_list[5]) + \
            self.get_order_maintaining_cost(n=demand[6], Q=Q_37, last_U=self.last_U_list[6])

        # Capacity Constraint, equation (3)
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

        if S >= sum(demand * self.beta_ms):
            # Try to meet the needs of all seriously injured people in the first tour
            problem += Q_21 + Q_31 >= demand[0] * self.beta_ms
            problem += Q_22 + Q_32 >= demand[1] * self.beta_ms
            problem += Q_33 >= demand[2] * self.beta_ms
            problem += Q_24 >= demand[3] * self.beta_ms
            problem += Q_25 + Q_35 >= demand[4] * self.beta_ms
            problem += Q_16 >= demand[5] * self.beta_ms
            problem += Q_37 >= demand[6] * self.beta_ms

        problem.setSolver(self.get_solver(self.solver_name))
        return problem, S, U, C, V, Q_21, Q_31, Q_22, Q_32, Q_33, Q_24, Q_25, Q_35, Q_16, Q_37

    def solve(self):
        k = 1
        while 1:
            # Terminating condition, equation (16)
            if np.nansum(self.demand) < self.epsilon or k > self.K:
                break
            k_tour_delivery_table = self.get_k_tour_delivery_table()
            demand = np.nansum(self.demand, axis=0)
            problem, S, U, C, V, Q_21, Q_31, Q_22, Q_32, Q_33, Q_24, Q_25, Q_35, Q_16, Q_37 = \
                self.initialize_prob(k_tour_delivery_table, demand, k)
            problem += sum(np.array(self.weights) * [S, U, -C, -V]), "Maximize weighted objectives"

            # Solve problem
            status = problem.solve()
            self.save_res(S, U, C, V, Q_21, Q_31, Q_22, Q_32, Q_33, Q_24, Q_25, Q_35, Q_16, Q_37, k_tour_delivery_table,
                          demand, k)
            k += 1

    def save_res(self, S, U, C, V, Q_21, Q_31, Q_22, Q_32, Q_33, Q_24, Q_25, Q_35, Q_16, Q_37,
                 k_tour_delivery_table, demand, k):
        omega = self.get_invulnerability(k_tour_delivery_table)
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

        # For calculating equity objective V, update in iteration
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

        # update the demand table (Table.3)
        if self.dead_flag:
            dead_crowd = self.serious_injured_recipients - self.last_Q
            dead_crowd[dead_crowd < 0] = 0
            self.dead_crowd = dead_crowd
            # Update current serious injured recipients
            self.serious_injured_recipients = np.array([0 for _ in range(7)])
        unmet_demand = self.last_n - self.last_Q - self.dead_crowd
        self.dead_crowd = 0

        new_demand = np.full((3, 7), np.nan)
        each_demand = unmet_demand / np.sum(~np.isnan(self.demand), axis=0)
        for col_idx in range(new_demand.shape[1]):
            initial_col = self.demand[:, col_idx]
            initial_col[~np.isnan(initial_col)] = each_demand[col_idx]
            new_demand[:, col_idx] = initial_col
        self.demand = new_demand

        res_table[7, :] = k_tour_delivery_table[4, :-1]
        res_table[8:, :] = k_tour_delivery_table[2:4, :-1]
        df_res = pd.DataFrame(res_table, columns=['r1', 'r2', 'r3'])
        self.res_table1 = pd.concat(([self.res_table1, df_res.round(2)]), axis=1)
        self.res_table2 = pd.concat(([self.res_table2, pd.DataFrame([omega, S, min(value(U), 7),
                                                                     max(0, value(C) / 1000),
                                                                     max(0, value(V) / 1000)],
                                                                    columns=['K={}'.format(k)]).round(2)]), axis=1)

    def get_res(self):
        self.res_table2['Total'] = self.res_table2.iloc[:, 1:].sum(axis=1)
        return self.res_table2

    def print_res(self):
        Delivery = self.res_table1.iloc[:, 1:].sum(axis=1)
        self.res_table1['Demand'] = self.initial_demand.tolist() + ['*' for _ in range(3)]
        self.res_table1['Delivery'] = Delivery
        self.res_table2['Total'] = self.res_table2.iloc[:, 1:].sum(axis=1)
        res_table2_arr = self.res_table2.values
        res_table2 = np.insert(res_table2_arr, [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                               np.full((res_table2_arr.shape[0], 1), '*', dtype=str),
                               axis=1)
        concat_res_table = pd.DataFrame(np.vstack((self.res_table1.values, res_table2)),
                                        columns=self.res_table1.columns.tolist())
        concat_res_table.to_excel(TABLE_DIR + '/policy-{}-res_table.xlsx'.format(self.policy), index=False)
        print(concat_res_table.to_markdown(index=False))

    def get_metrics(self):
        total_Delivery = np.array(self.res_table1.iloc[7].tolist()[1:]).reshape(-1, 3).sum(axis=1)
        self.res_table2['Total'] = self.res_table2.iloc[:, 1:].sum(axis=1)
        return np.round(np.hstack((total_Delivery, np.array(self.res_table2['Total'].tolist()[1:]))), 2)
