# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/2 10:21
@Auth ： Hongwei
@File ：SensitivityAnalysis.py
@IDE ：PyCharm
"""
from definitions import *


def run_varying_beta_ms(beta_mses):
    for beta_ms in beta_mses:
        cmd = 'python -um src.main --beta_ms {} --flag {}'.format(beta_ms, 'SensitivityAnalysis_beta_ms')
        os.system(cmd)


def run_varying_beta_v1(beta_v1s):
    for beta_v1 in beta_v1s:
        cmd = 'python -um src.main --beta_v1 {} --flag {}'.format(beta_v1, 'SensitivityAnalysis_beta_v1')
        os.system(cmd)


if __name__ == '__main__':
    save_pickle([], RESULT_DIR + '/beta_ms_arr.pickle')
    run_varying_beta_ms(beta_mses=[0.1, 0.2, 0.6])
    beta_ms_arr = load_pickle(RESULT_DIR + '/beta_ms_arr.pickle')
    beta_ms_table = np.hstack((np.array([0.1, 0.2, 0.6]).reshape(-1, 1), np.array(beta_ms_arr)))
    beta_ms_table = pd.DataFrame(beta_ms_table, columns=['Beta_ms', 'k1', 'k2', 'k3', 'k4', 'k5', 'S', 'U', 'C', 'V'])
    beta_ms_table.to_excel(TABLE_DIR + '/beta_ms_table.xlsx', index=False)
    print(beta_ms_table.to_markdown())

    save_pickle([], RESULT_DIR + '/beta_v1_arr.pickle')
    run_varying_beta_v1(beta_v1s=[0.3, 0.5, 0.8])
    beta_v1_arr = load_pickle(RESULT_DIR + '/beta_v1_arr.pickle')
    beta_v1_table = np.hstack((np.array([0.3, 0.5, 0.8]).reshape(-1, 1), np.array(beta_v1_arr)))
    beta_v1_table = pd.DataFrame(beta_v1_table, columns=['Beta_v1', 'k1-V', 'k2-V', 'k3-V', 'k4-V', 'k5-V'])
    beta_v1_table.to_excel(TABLE_DIR + '/beta_v1_table.xlsx', index=False)
    print(beta_v1_table.to_markdown())
