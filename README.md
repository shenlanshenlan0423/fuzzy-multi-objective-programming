# A fuzzy multi-objective programming model for the delivery and distribution of humanitarian relief materials

## Dependencies
For a straight-forward use of this code, you can install the required libraries from *requirements.txt*: `pip install -r requirements.txt` 

## Experiment Result
NPO’s spontaneous relief scheme

|          ***          |   r1    |   r2    |   r3   |   r1    |   r2    |   r3    |   r1   |   r2    |   r3    |   r1   |   r2    |   r3    |   r1   |   r2    |   r3    | Demand | Delivery |
|:---------------------:|:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:------:|:-------:|:-------:|:------:|:-------:|:-------:|:------:|:--------:|
|          m1           |   0.0   |  1000   |  0.0   |   0.0   |    0    |   0.0   |  0.0   |    0    |   0.0   |  0.0   | 3623.06 |   0.0   |  0.0   | 376.94  |   0.0   | 5000.0 |   5000   |
|          m2           |   0.0   |    0    | 1000.0 |   0.0   |    0    |   0.0   |  0.0   |    0    |   0.0   |  0.0   |    0    | 210.36  |  0.0   | 882.85  | 2906.79 | 5000.0 |   5000   |
|          m3           |   0.0   |    0    | 1000.0 |   0.0   |    0    | 2914.29 |  0.0   |    0    |   0.0   |  0.0   |    0    | 1085.71 |  0.0   |    0    |   0.0   | 5000.0 |   5000   |
|          m4           |   0.0   | 1411.11 |  0.0   |   0.0   | 3567.46 |   0.0   |  0.0   |  21.43  |   0.0   |  0.0   |    0    |   0.0   |  0.0   |    0    |   0.0   | 5000.0 |   5000   |
|          m5           |   0.0   |  1200   |  0.0   |   0.0   |    0    |   0.0   |  0.0   | 3153.01 |   0.0   |  0.0   |    0    |   0.0   |  0.0   | 1646.99 |   0.0   | 6000.0 |   6000   |
|          m6           | 1388.89 |    0    |  0.0   | 1289.68 |    0    |   0.0   | 1147.6 |    0    |   0.0   | 881.7  |    0    |   0.0   | 292.13 |    0    |   0.0   | 5000.0 |   5000   |
|          m7           |   0.0   |    0    | 1000.0 |   0.0   |    0    |   0.0   |  0.0   |    0    | 2593.22 |  0.0   |    0    | 1406.78 |  0.0   |    0    |   0.0   | 5000.0 |   5000   |
|    Total delivery     | 1388.89 | 3611.11 | 3000.0 | 1289.68 | 3567.46 | 2914.29 | 1147.6 | 3174.44 | 2593.22 | 881.7  | 3623.06 | 2702.85 | 292.13 | 2906.79 | 2906.79 |   *    |  36000   |
|    Truck delivery     | 277.78  | 722.22  | 1000.0 | 257.94  | 713.49  | 1028.57 | 229.52 | 634.89  | 1135.59 | 176.34 | 724.61  | 1099.05 | 58.43  | 581.36  | 581.36  |   *    | 9221.15  |
| Actual truck delivery | 277.78  | 722.22  | 600.0  | 257.94  | 713.49  | 582.86  | 229.52 | 634.89  | 518.64  | 176.34 | 724.61  | 540.57  | 58.43  | 581.36  | 581.36  |   *    | 7200.01  |
|         Omega         |    *    | 3251.05 |   *    |    *    | 3156.12 |    *    |   *    | 2808.41 |    *    |   *    | 2916.33 |    *    |   *    | 2454.85 |    *    |   *    | 14586.8  |
|           S           |    *    |  7200   |   *    |    *    |    0    |    *    |   *    |    0    |    *    |   *    |    0    |    *    |   *    |    0    |    *    |   *    |   7200   |
|           U           |    *    |  1.56   |   *    |    *    |  2.08   |    *    |   *    |   2.8   |    *    |   *    |  4.71   |    *    |   *    |    7    |    *    |   *    |  18.15   |
|           C           |    *    | 9732.22 |   *    |    *    | 8056.49 |    *    |   *    | 7168.91 |    *    |   *    | 7478.28 |    *    |   *    | 6344.09 |    *    |   *    |  38780   |
|           V           |    *    |  1400   |   *    |    *    | 941.14  |    *    |   *    | 597.88  |    *    |   *    | 255.88  |    *    |   *    |    0    |    *    |   *    |  3194.9  |

Traffic control scheme with government regulation

|          ***          |   r1    |   r2    |   r3   |   r1    |   r2    |   r3   |   r1    |   r2    |   r3    |   r1    |   r2    |   r3   | r1  |   r2   |  r3   | Demand | Delivery |
|:---------------------:|:-------:|:-------:|:------:|:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:------:|:---:|:------:|:-----:|:------:|:--------:|
|          m1           |   0.0   |    0    | 1000.0 |   0.0   |    0    |  0.0   |   0.0   |    0    | 1000.0  |   0.0   |  3000   |  0.0   | 0.0 |   0    |  0.0  | 5000.0 |   5000   |
|          m2           |   0.0   |  1000   |  0.0   |   0.0   |  110.7  |  0.0   |   0.0   | 3532.07 |   0.0   |   0.0   | 357.23  |  0.0   | 0.0 |   0    |  0.0  | 5000.0 |   5000   |
|          m3           |   0.0   |    0    | 2000.0 |   0.0   |    0    | 3000.0 |   0.0   |    0    |   0.0   |   0.0   |    0    |  0.0   | 0.0 |   0    |  0.0  | 5000.0 |   5000   |
|          m4           |   0.0   | 1411.11 |  0.0   |   0.0   | 3588.89 |  0.0   |   0.0   |    0    |   0.0   |   0.0   |    0    |  0.0   | 0.0 |   0    |  0.0  | 5000.0 |   5000   |
|          m5           |   0.0   |  1200   |  0.0   |   0.0   |    0    |  0.0   |   0.0   |    0    |   0.0   |   0.0   | 642.77  | 4000.0 | 0.0 | 78.61  | 78.61 | 6000.0 | 5999.99  |
|          m6           | 1388.89 |    0    |  0.0   | 1337.45 |    0    |  0.0   | 1265.75 |    0    |   0.0   | 1007.91 |    0    |  0.0   | 0.0 |   0    |  0.0  | 5000.0 |   5000   |
|          m7           |   0.0   |    0    | 1000.0 |   0.0   |    0    | 1000.0 |   0.0   |    0    | 3000.0  |   0.0   |    0    |  0.0   | 0.0 |   0    |  0.0  | 5000.0 |   5000   |
|    Total delivery     | 1388.89 | 3611.11 | 4000.0 | 1337.45 | 3699.59 | 4000.0 | 1265.75 | 3532.07 | 4000.0  | 1007.91 |  4000   | 4000.0 | 0.0 | 78.61  | 78.61 |   *    |  36000   |
|    Truck delivery     | 277.78  | 722.22  | 1000.0 | 267.49  | 739.92  | 992.59 | 253.15  | 706.41  | 1040.44 | 201.58  | 815.72  | 815.72 | 0.0 | 15.72  | 15.72 |   *    | 7864.46  |
| Actual truck delivery | 277.78  | 722.22  | 800.0  | 267.49  | 739.92  | 800.0  | 253.15  | 706.41  |  800.0  | 201.58  |   800   | 800.0  | 0.0 | 15.72  | 15.72 |   *    | 7199.99  |
|         Omega         |    *    | 3651.47 |   *    |    *    | 3664.53 |   *    |    *    | 3566.28 |    *    |    *    | 3641.54 |   *    |  *  | 62.96  |   *   |   *    | 14586.8  |
|           S           |    *    |  7200   |   *    |    *    |    0    |   *    |    *    |    0    |    *    |    *    |    0    |   *    |  *  |   0    |   *   |   *    |   7200   |
|           U           |    *    |  1.76   |   *    |    *    |  2.65   |   *    |    *    |  4.71   |    *    |    *    |  6.97   |   *    |  *  |   7    |   *   |   *    |  23.09   |
|           C           |    *    | 10772.2 |   *    |    *    | 9371.77 |   *    |    *    | 9124.42 |    *    |    *    | 9348.07 |   *    |  *  | 163.52 |   *   |   *    |  38780   |
|           V           |    *    |  1350   |   *    |    *    | 829.94  |   *    |    *    | 428.02  |    *    |    *    |  3.14   |   *    |  *  |   0    |   *   |   *    |  2611.1  |



## Example Usage

For solve the problem, you can utilize the following statement to execute the program:
```
python -um src.main
```
All parameters involved in the model are listed below:
```
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
parser.add_argument('--k', type=int, default=6, help='Max iteration')
parser.add_argument('--flag', type=str, default='Solve', help='Solve, SensitivityAnalysis_beta_ms or SensitivityAnalysis_beta_v1')
parser.add_argument('--solver', type=str, default='CBC', help='Solver configuration of pulp library')
```

To perform sensitivity analysis for the problem, you can utilize the following statement to execute the program:
```
python -um src.SensitivityAnalysis
```

For convenience, you can directly run the `src/main.py` or `src/SensitivityAnalysis.py`program to conduct the experiment.