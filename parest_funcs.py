# implements the parameter estimation routines
import numpy as np
import pandas as pd
from lmfit import minimize, Parameters
from model_funcs import sim_single_exp

def residuals_single_exp(p, c, y0, datasets):
      """ calculate residuals for a single experiment
      INPUT:
      p ... structure with parameter values to be estimated, cf. lmfit.Parameters
      c ... list of control values, to be passed to model function
            c[0] ... time point when feed was switched on [h]
            c[1] ... feed rate [L/h]
            c[2] ... substrate concentration in feed [g/L]
      y0 ... initial state vector:
            y[0] ... substrate mass (mS) in [g]
            y[1] ... bio dry mass (mX) in [g]
            y[2] ... volume of fermentation broth [L]
      datasets ... list of data frame with measurement data

      OUTPUT:
      res ... long vector with all residuals for this experiment
      """

      res = np.array([]) # empty array, will contain residuals

      weighting_factor = {'cX': 1.0, 'cS': 1.0, 'cE' : 1.0, 'base_rate': 1.0*c[6], 'CO2' : 1.0*c[7]} # individual weighting factor c[5]len(off/on) c[6]len(off/CO2)

      for dat in datasets: # loop over datasets
            t_grid = dat.index.values  # index of 'dat' = time grid of measurements = time grid for simulation   #warum eigentlich values hier?!!
            sim_exp = sim_single_exp(t_grid, y0, p, c) # simulate experiment with this time grid

            for var in dat: # loop over all measured variables
                  res_var = weighting_factor[var]*(sim_exp[var] - dat[var]).values # weighted residuals for this measured variable
                  res = np.append(res, res_var) # append to long residual vector

      return res

def residuals_all_exp(p, y0_dict, c_dict, datasets_dict):
      """ calculate residuals for all experiment
      INPUT:
      p ... structure with parameter values to be estimated, cf. lmfit.Parameters
      y0_dict ... dict: keys: experiment names, values: initial state vector y0
      c_dict ... dict: keys: experiment names, values: control variables c
      datasets(_dict?) ... dictionary: keys: experiment names, values: list of data frame with measurement data   #datasets_dict oder

      OUTPUT:
      res ... super long vector with all residuals for all experiment
      """

      exp_names = y0_dict.keys() # experiment names

      res_all_exp = [] # empty (list which will be an array), will contain residuals

      for exp in exp_names: # loop over experiments
            y0 = y0_dict[exp]
            c = c_dict[exp]     #y0 dict vorher
            datasets = datasets_dict[exp]

            res_this_exp = residuals_single_exp(p, c, y0, datasets)
            res_all_exp = np.append(res_all_exp, res_this_exp)
      
      return res_all_exp


