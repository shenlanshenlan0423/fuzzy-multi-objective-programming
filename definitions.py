"""
This file contains basic variables and definitions that we wish to make easily accessible for any script that requires
it.

from definitions import *
"""
from pathlib import Path
from src.omni.functions import load_pickle, save_pickle
import os
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pulp import LpVariable, LpProblem, LpMaximize, LpStatus, value
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='Palatino Linotype')

ROOT_DIR = str(Path(__file__).resolve().parents[0])
SOURCE_DIR = ROOT_DIR + '/src/'
RESULT_DIR = ROOT_DIR + '/results/'
FIG_DIR = ROOT_DIR + '/FigSupp/'
TABLE_DIR = ROOT_DIR + '/TableSupp/'


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SOURCE_DIR+'/model/', exist_ok=True)
