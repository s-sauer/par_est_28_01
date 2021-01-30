# implements the parameter estimation routines
import numpy as np
import pandas as pd
from lmfit import minimize, Parameters
from model_funcs import sim_single_exp

import sympy as sp
from sympy import Eq
from sympy import symbols




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

      #Code for chemical balancing 

      #Order: C,H,O,N
      gluc = np.array([6.0,12.0,6.0,0.0])
      O2 = np.array([0.0, 0.0, 2.0, 0.0])
      NH3 = np.array([0.0,3.0,0.0,1.0])
      biomass = np.array([1.0,p['HX'].value, p['OX'].value, p['NX'].value])
      CO2 = np.array([1.0,0.0,2.0,0.0])
      H2O = np.array([0.0,2.0,1.0,0.0])
      etoh = np.array([2.0,6.0,1.0,0.0])

      NX1 = p['NX'].value

      MW_element_dict = {"C": 12.011, "H": 1.0079, "O": 15.999, "N": 14.007}
      molecule = {"gluc": gluc, "O2": O2, "NH3" : NH3, "biomass": biomass, "CO2" : CO2, "H2O":  H2O, "etoh": etoh}

      MW = {}

      for key, mol in molecule.items():
            molecule_MW_array = ([])
            for vectorvalue, weight in zip (mol, MW_element_dict.values()):
                  vw = vectorvalue*weight
                  molecule_MW_array= np.append(molecule_MW_array, vw)
            MW[key] = sum(molecule_MW_array)


      #Oxidative Equation: gluc+ a*O2 + b*NX*NH3 = b*biomass + c*CO2 + d*H2O 
      a,b,c,d, NX = symbols("a b c d NX")
      YxsOx = p['YxsOx'].value
      b1 = YxsOx* MW["gluc"]/MW["biomass"]

      eqOx_list = []
      for num in range(3):
            eqOx = Eq(gluc[num]+ a*O2[num]+ b*NX*NH3[num], b*biomass[num] + c*CO2[num] + d*H2O[num])
            eqOx = eqOx.subs({b: b1, NX: NX1})
            eqOx_list.append(eqOx)
      
      solution_Ox = sp.solve(eqOx_list, (a, c, d), dict= True)
      a1, c1, d1 = np.float(solution_Ox[0][a]), np.float(solution_Ox[0][c]), np.float(solution_Ox[0][d])
      
      Yco2xOx = c1/b1 * MW["CO2"]/MW["biomass"]
      p.add('Yco2xOx', value=Yco2xOx, vary=False)


      #Reductive Equation:  gluc+ g*NX*NH3 = g*biomass + h*CO2 + i*H2O + j*etOh
      g,h,i,j, NX = symbols("g h i j NX")
      YxsRed = p['YxsRed'].value
      g1 = YxsRed* MW["gluc"]/MW["biomass"]

      eqRed_list = []
      for num in range(3): # range 3 because of C,H,O,  N is redundant for this LGS
            eqRed = Eq(gluc[num]+ g*NX*NH3[num],  g*biomass[num] + h*CO2[num]+ i*H2O[num]+ j*etoh[num])
            eqRed = eqRed.subs({g: g1, NX: NX1})
            eqRed_list.append(eqRed)  

      solution_Red = sp.solve(eqRed_list, (h, i, j), dict= True)
      h1,i1,j1 = np.float(solution_Red[0][h]), np.float(solution_Red[0][i]), np.float(solution_Red[0][j])
      
      YesRed = j1/1 * MW["etoh"]/MW["gluc"]
      Yco2xRed = h1/g1 * MW["CO2"]/MW["biomass"]
      p.add('YesRed', value=YesRed, vary=False)
      p.add('Yco2xRed', value=Yco2xRed, vary=False)


      #ethanol consumption: etoh + k*O2 + l*NX = l*biomass + m*CO2 + n*H2O
      k,l,m,n, NX = symbols("k l m n NX")
      Yxe = p['Yxe'].value
      l1 = Yxe* MW["etoh"]/MW["biomass"]

      eqEt_list = []
      for num in range(3):
            eqEt = Eq(etoh[num]+ k*O2[num]+ l*NX*NH3[num], l*biomass[num] + m*CO2[num] + n*H2O[num])
            eqEt = eqEt.subs({l: l1, NX: NX1})
            eqEt_list.append(eqEt)
            
      solution_Et = sp.solve(eqEt_list, (k, m, n), dict= True)
      k1, m1, n1 = np.float(solution_Et[0][k]), np.float(solution_Et[0][m]), np.float(solution_Et[0][n])

      Yco2xEt = m1/l1 * MW["CO2"]/MW["biomass"]
      p.add('Yco2xEt', value=Yco2xEt, vary=False)
       

      
      exp_names = y0_dict.keys() # experiment names

      res_all_exp = [] # empty (list which will be an array), will contain residuals

      for exp in exp_names: # loop over experiments
            y0 = y0_dict[exp]
            c = c_dict[exp]     #y0 dict vorher
            datasets = datasets_dict[exp]

            res_this_exp = residuals_single_exp(p, c, y0, datasets)
            res_all_exp = np.append(res_all_exp, res_this_exp)

      return res_all_exp


