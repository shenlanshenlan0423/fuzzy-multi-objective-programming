# A fuzzy multi-objective programming model for the delivery and distribution of humanitarian relief materials

## Dependencies
For a straight-forward use of this code, you can install the required libraries from *requirements.txt*: `pip install -r requirements.txt` 

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
parser.add_argument('--k', type=int, default=5, help='Max iteration')
parser.add_argument('--flag', type=str, default='Solve', help='Solve or SensitivityAnalysis')
```

To perform sensitivity analysis for the problem, you can utilize the following statement to execute the program:
```
python -um src.SensitivityAnalysis
```

For convenience, you can directly run the `src/main.py` or `src/SensitivityAnalysis.py`program to conduct the experiment.