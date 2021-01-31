# implements the kinetic model
# @Paul: Hier muss du in deiner Rolle als "Modellierer" den Code für dein kinetisches Prozessmodell hinterlegen

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


#      global inner_model
def inner_model(t,y, p, c):     
      """ r.h.s. (right hand side) function of the ODE model 

      INPUT:
      t ... current time [h]
      y ... state vector:
            y[0] ... substrate mass (mS) in [g]
            y[1] ... bio dry mass (mX) in [g]
            y[2] ... volume of fermentation broth [L]
      p ... structure with parameter values to be estimated, cf. lmfit.Parameters
      c ... list of control values
            c[0] ... time point when feed was switched on [h]
            c[1] ... feed rate [L/h]
            c[2] ... substrate concentration in feed [g/L]
            
      OUTPUT:
      dy_dt ... time derivative of state vector
      """
      #fix parameters 
      dV_gas_dt = c[4]          # c[4] air flow in each experiment
      R = 0.08314               #bar*l/mol*K
      T = c[5] + 273.15         #Kelvin   c[5] = °C
      pressure = c[8]                  #bar     c[8] = mean p for each experiment
      M_CO2 = 44.01             #g/mol

      # (potential) fit parameters

      qsmax = p['qsmax'].value
      Ks    = p['Ks'].value
      mumaxE = p['mumaxE'].value
      Ke = p['Ke'].value
      Ki = p['Ki'].value
      

      YxsOx = p['YxsOx'].value
      YxsRed = p['YxsRed'].value
      YesRed = p['YesRed'].value
      Yxe = p['Yxe'].value


      YexRed = p['YexRed'].value
      Yco2xRed = p['Yco2xRed'].value
      Yco2xOx = p['Yco2xOx'].value
      Yco2xEt = p['Yco2xEt'].value

      cSCrab = p['cSCrab'].value


      # controls
      feed_on = c[0] # time point when feed was switched on [h]
      feed_rate = c[1] # feed rate [L/h]
      Fin = feed_rate * (t > feed_on) # becomes 0 if t < feed_on
      csf = c[2] # substrate concentration in feed [g/L]

      # masses and concentrations
      mS, mX, mE, V = y
      cS, cX, cE = [mS, mX, mE] / V    

      cSCrab = p['cSCrab'].value

      #kinetics
      qs = qsmax * cS / (cS + Ks)

      #qSOx_max = qsmax  * cSCrab / (cSCrab + Ks)

      if cS > cSCrab: # qS > qSOx_max
            qsOx = qsmax  * cSCrab / (cSCrab + Ks) # qsOx = qsOxmax
            qsRed = qs - qsOx 
            muE = 0.0
            
      else:
            qsOx = qs  
            qsRed = 0
            muE = mumaxE* cE/(cE + Ke) * Ki / (cS+ Ki)

      muOx = qsOx * YxsOx
      muRed = qsRed * YxsRed
      qE = 1/Yxe * muE

      #r.h.s. of ODE
      # c_dict hat list values mit der Rheienfolge : 0: feed_on, 1 : feed_rate ,2: csf, 3: M_base, 4: gas_flow, 5: Glycerin pro ET factor, 6: timepoint end lagEt

      dmS_dt = cX *V*(- qsOx- qsRed)+ csf *Fin
      dmX_dt = (muOx+muRed+muE) * cX * V

                  
      dmE_dt = (qsRed * YesRed - qE)*cX*V

      dV_dt  = + Fin

      dmCO2_dt = (muRed* Yco2xRed + muOx*Yco2xOx + muE * Yco2xEt) *cX*V
      dnCO2_dt = dmCO2_dt/M_CO2
      dvCO2_dt = (dnCO2_dt*R*T)/pressure

      CO2_percent = 100 * dvCO2_dt / dV_gas_dt

      return dmS_dt, dmX_dt, dmE_dt,  dV_dt, CO2_percent

def model_rhs(t, y, p, c): 
      dmS_dt, dmX_dt, dmE_dt,  dV_dt, CO2_percent = inner_model(t,y, p, c)
      return dmS_dt, dmX_dt, dmE_dt,  dV_dt



def sim_single_exp(t_grid, y0, p, c):  
    """ simulates single experiment and calculates measured quantities
    
    INPUT:
    t_grid ... time grid on which to generate the solution [h]
    y0 ... initial state vector:
          y[0] ... substrate mass (mS) in [g]
          y[1] ... bio dry mass (mX) in [g]
          y[2] ... volume of fermentation broth [L]
    p ... structure with parameter values to be estimated, cf. lmfit.Parameters
    c ... list of control values, to be passed to model function
          c[0] ... time point when feed was switched on [h]
          c[1] ... feed rate [L/h]
          c[2] ... substrate concentration in feed [g/L]
          
    OUTPUT:
    sim_exp ... data frame with simulated cS, cX and base consumption rate over time (for single experiment)
    """
    
    # run ODE solver to get solution y(t)
    y_t = solve_ivp(model_rhs, [np.min(t_grid), np.max(t_grid)], y0, t_eval=t_grid, args = (p, c), method= "Radau", first_step = 0.0000001).y.T

    # unpack solution into vectors
    mS, mX, mE, V = [y_t[:,i] for i in range(4)]
    cS, cX, cE = [mS, mX, mE] / V

   


    # for base consumption rate: get value of dmX_dt at all times t
    dmX_dt = np.array([model_rhs(t_grid[i], y_t[i,:], p, c) for i in range(len(t_grid))])[:,1]
    base_rate = p['base_coef'].value /c[3] * dmX_dt
    CO2_percent = np.array([inner_model(t_grid[i], y_t[i,:], p, c) for i in range(len(t_grid))])[:,4]

    # pack everything neatly together to a pandas df
    sim_exp = pd.DataFrame(
          {'t': t_grid,
          'cS': cS, 'cX': cX,
          'V': V, 'base_rate': base_rate,
          'CO2' : CO2_percent, 'cE' : cE }
          ).set_index('t') # make time column the index column

    return sim_exp

#CO2_volpercent =  koeff / (Volume_flow (L/h)) * dmX_dt
#/base conc