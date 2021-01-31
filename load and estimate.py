import numpy as np

from scipy.integrate import odeint, solve_ivp
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import glob
import os.path
from model_funcs import sim_single_exp
import numpy as np
import pandas as pd
from lmfit import Parameters, report_fit, minimize, fit_report
from copy import deepcopy

massdat_path = '/Users/sauers/Nextcloud/Austauschordner_Masterarbeit_P_Senck/Messdaten2' # for Simeon's Macbook

metadata = pd.read_excel(os.path.join(massdat_path,"metadata.xlsx"), index_col = "fermentation")         #read metadata


exp_numbers =[4,5,6,7,8]                                           # list with number of each experiment, hilft bei der Benennung von Experimenten

start_dict = {}
for  timestamp, i  in zip(metadata["start"].T, exp_numbers):       # create dictionary with start points out of metadata
    start_dict["start{0}".format(i)]= timestamp



end_dict = {}
for  timestamp, i  in zip(metadata["end1"].T, exp_numbers):        # create dictionary with end points from the first day, zweiter Tag kommt nicht in die Schätzung 
    end_dict["end{0}".format(i)]= timestamp


path_On = os.path.join(massdat_path, 'online')

#online_files = glob.glob(os.path.join(path_On, "*.CSV"))           #create list of online files 
online_files = [os.path.join(path_On,"online{0}.CSV".format(i)) for i in exp_numbers]

online_dict = {}

for f, i in zip(online_files, exp_numbers):                        #create dict with online[number] as key and dataframe as value
    online_dict["online{0}".format(i)] = pd.read_csv(f,sep=";",encoding= 'unicode_escape',decimal=",", skiprows=[1,2] , skipfooter=1, engine ="python", usecols = ["PDatTime","BASET"])

for df, start, end, i in zip( online_dict.values(), start_dict.values(), end_dict.values(), exp_numbers ):   #loop to process online data, timeframe, indexset, base_rate creation
    df["PDatTime"] = pd.to_datetime( df["PDatTime"], format = "%d.%m.%Y  %H:%M:%S" )
    df = df[(df["PDatTime"] >= start ) &  (df["PDatTime"] <= end)]  #Zeit filtern
    df["t"] = (df["PDatTime"] - start) / pd.Timedelta(1,'h')        #for give time as decimal number
    df.set_index("t", inplace = True, drop = True)       #index setzen
    df["BASET"] = pd.to_numeric(df["BASET"], downcast="float", errors='coerce') # Bei BASET gab es vereinzelnte Messwerte die vorher kein fLoat waren
    df["base_rate"] = df["BASET"].diff()    #differential bilden der BASET werte 
    online_dict["online{0}".format(i)] = df

special_start_for_online8 = pd.to_datetime("14.12.2020  12:20:16")  #special start weil der schlauch ein loch hatte und am anfang zu viel base gepumpt wurde
online_dict["online8"] = online_dict["online8"][(df["PDatTime"] >= special_start_for_online8 )]  # vom 8ten Lauf die erste Zeit abschneiden für verlässliche base_rate values


path_Off = os.path.join(massdat_path, 'offline')             #read offline values in a dict
#offline_files = glob.glob(os.path.join(path_Off, "*.csv"))  # Simeon: hat mit '*.csv' nicht funktioniert...         
offline_files = [os.path.join(path_Off,"offline{0}.CSV".format(i)) for i in exp_numbers]

offline_dict = {}

for f, i in zip(offline_files, exp_numbers):             # create offline dict          
    offline_dict["offline{0}".format(i)] = pd.read_csv(f,sep=";", encoding= 'unicode_escape', header = 0, usecols = [0,2,3,4,5,6], names =["ts","cX","cS","cE","cGly","cP"] )

for df, start, end, i in zip( offline_dict.values(), start_dict.values(), end_dict.values(), exp_numbers ):   #processing offline dict
    df["ts"] = pd.to_datetime( df["ts"], format = "%d.%m.%Y %H:%M")  
    df = df[(df["ts"] >= start ) &  (df["ts"] <= end)]
    df["t"] = (df["ts"] - start) / pd.Timedelta(1,'h')
    df.set_index("t", inplace = True, drop = True)
    offline_dict["offline{0}".format(i)] = df


start_lag_list = [5.24, 4.0, 4.0, 2.4, 2.5 ]        
end_lag_list = [10.0, 10.0, 6.0, 5.76, 6.2 ]        # end and start points for reliable values for calculating Gly/EtOH formation

mean_cGly_cE_list = []  #calculate how much glycerol arise per ethanol and store it as list entry for every experiment 
for df, end_lag, start_lag in zip(offline_dict.values(), end_lag_list, start_lag_list):
    df = df[(df.index >= start_lag ) &  (df.index <= end_lag)]
    mean_cGly_cE_list.append(np.mean(df["cGly"]/df["cE"]))
    
Gly_per_Et = np.mean(mean_cGly_cE_list)         #mean over all experiments


path_CO2 = os.path.join(massdat_path, 'CO2')            
#CO2_files = glob.glob(os.path.join(path_CO2, "*.dat"))  
CO2_files = [os.path.join(path_CO2,"co2_{0}.dat".format(i)) for i in exp_numbers]

CO2_dict = {}

for f, i in zip(CO2_files, exp_numbers):                 #read CO2 values in a dict        
    CO2_dict["CO2_{0}".format(i)] = pd.read_csv(f, sep=";", encoding= 'unicode_escape', header = 0, skiprows=[0], usecols=[0,2,4], names =["ts","CO2", "p"])


for df, start, end, i in zip( CO2_dict.values(), start_dict.values(), end_dict.values(), exp_numbers ):   #loop to process CO2 data for correct timeframe
    
    try:
        df["ts"] = pd.to_datetime( df["ts"], format = "%d.%m.%Y %H:%M:%S", exact= False) #sometimes you have to take this format, sometimes not, depending on single specific rows
    except:
        df["ts"] = pd.to_datetime( df["ts"] )
    #df["ts"] = pd.to_datetime( df["ts"], format = "%d.%m.%Y %H:%M", exact= False, errors= "coerce") # this could maybe be an alternative solution
    df = df[(df["ts"] >= start ) &  (df["ts"] <= end)]
    df["t"] = (df["ts"] - start) / pd.Timedelta(1,'h')
    df.set_index("t", inplace = True, drop = True)
    CO2_dict["CO2_{0}".format(i)] = df


# get mean pressure values from CO2 measurements and safe them in a dict: CO2_mean_p_dict, In case we want to calculate with p_mean for each experiment later on.
CO2_mean_p_dict = {}
for key, df in CO2_dict.items():
    CO2_mean_p_dict[key] = df["p"].mean()
CO2_mean_p_dict

try:        # try/except nur weil es vorhr in einzelnen zellen war und ich die zelle öfter laufen lassen wollte, wird im Hautpcode dann wahrscheinlich entfernt.
    for df_On, df_Off, df_CO2 in zip(online_dict.values(), offline_dict.values(), CO2_dict.values() ): # Drop Columns which are not needed for estimation anymore
            df_On.drop(["BASET", "PDatTime"], inplace=True, axis=1)
            #df_Off.drop(["ts","cE","cGly","cP"], inplace=True, axis=1)
            df_Off.drop(["ts","cGly","cP"], inplace=True, axis=1)
            df_CO2.drop(["ts","p"], inplace=True, axis=1)
except Exception:
    pass

len_off_div_on = []                 # making lists for weighting factors with len(dfOff)/len(df2On or dfCO2) as values 
len_off_div_CO2 = []
for online, offline, CO2 in zip (online_dict.values(), offline_dict.values(), CO2_dict.values()):
    off_div_on = len(offline)/np.max([len(online),1]) # Änderung Simeon
    off_div_CO2 = len(offline)/np.max([len(CO2),1])   # Änderung Simeon
    len_off_div_on.append(off_div_on)
    len_off_div_CO2.append(off_div_CO2)


y0_dict = {}            # dicts which are used for the estimation the keys, are "ferm4", "ferm5" and so on = rownames from metadata
c_dict = {}
datasets_dict = {}

for column, row, end_lag, p_mean, off_div_on, off_div_CO2 in zip(metadata, metadata.T, end_lag_list, CO2_mean_p_dict.values(), len_off_div_on, len_off_div_CO2):  
    c_dict["{}".format(row)] = list(metadata.loc[row,["feed_on", "feed_rate","csf", "M_base", "gas_flow", "T"]].values)
    c_dict["{}".format(row)].append(off_div_on)     #append len off / on as control variable for respetive experiment
    c_dict["{}".format(row)].append(off_div_CO2)    #append len off / CO2 as control variable for respetive experiment
    c_dict["{}".format(row)].append(p_mean)      # falls wir den mean von Druck pro Versuch als c value haben wollen
    
    #c_dict["{}".format(row)].append(end_lag)  # falls wir den lag von ethanol einbauen wollen, ist ja aber wie besprochen: bad practise
    
    y0_dict["{}".format(row)] = list(metadata.loc[row,["mS0", "mX0", "mE0","V0"]].values)


for rowname ,i,e,z in zip(metadata.T, online_dict.values(), offline_dict.values(), CO2_dict.values()):
    datasets_dict["{}".format(rowname)] = list([i,e,z])



    
datasets_dict_original = deepcopy(datasets_dict)        #for plotting later on but not for parameterestimation, hatte ich anfänglich  gebraucht, weil ja ein paar online dicts mit der schlechten Baseauflösung rausgeworfen wurden, oder weil ich vorher noch keine CO2 values drin hatte. Nur dazu da damit trotzdem alles geplottet wird. Also ich habe den dict zum plotten genommen und datasets_dict zum schätzen. 


# delete specific dataframes which are not used in estimation
try:
    if datasets_dict["ferm4"][0].columns in ["base_rate"]:                    # try except und if wieder nur für Zellen rerunability für einzelne Jupyter Zellen, wird noch entfernt
        datasets_dict["ferm4"].pop(0); datasets_dict["ferm5"].pop(0); datasets_dict["ferm6"].pop(0) # unbrauchbare base values rausschmeißen
except Exception:
    pass



p0 = Parameters()
p0.add('qsmax', value=0.5, min=0.0001, max=5.0 , vary = True)
p0.add('mumaxE', value=0.17, vary=False)
p0.add('base_coef', value=1, min=0.0001, vary = True) 

p0.add('cSCrab', value=0.1, min=0.008, max=0.15, vary = True)
  
p0.add('Ks', value=0.1, vary=False)
p0.add('Ke', value=3.0, vary=True)   # jetzt wichiger Parameter!
p0.add('Ki', value=0.1, vary=False)  # obsolete

p0.add('YxsOx', value=0.49, vary=False)
p0.add('YxsRed', value=0.05, vary=False)
p0.add('Yxe', value=0.72, vary=False)
p0.add('YexRed', value=9.58516, vary=False)
p0.add('YesRed', value=0.4792, vary=False)

p0.add('Yco2xRed', value=9.244, vary=False)
p0.add('Yco2xOx', value=1.233, vary=False)
p0.add('Yco2xEt', value=0.8957471802080247, vary=False)   




from parest_funcs import residuals_all_exp
result = minimize(residuals_all_exp, p0, args=(y0_dict, c_dict, datasets_dict), method='leastsq', nan_policy= "omit")


report_fit(result)


p_new = result.params
fit_dict = {}
t_sim = np.linspace(0,9, 1001)

for [exp_name, y0] , c in zip(y0_dict.items(), c_dict.values()): 
    sim_exp = sim_single_exp(t_sim, y0, p_new, c)  # für jedes c, y0, mit p_new nochmal model simulieren, also für jeden versuch 
    fit_dict[exp_name] = sim_exp  





line_markers = "lines+markers"
line = "lines"



for [key_dat, df_list], [key_fit, df_fit] in zip(datasets_dict_original.items(), fit_dict.items()):
    print("this is exp_data: ", key_dat)
    fig = make_subplots( specs=[[{"secondary_y": True}]]) 
    
    #exp data add
    for df_dat in df_list:
        
        for column in df_dat:
            secondary_y_flag = column  in ["base_rate", "CO2"]
            
            fig.add_trace( 
    go.Scatter(x= df_dat.index, y= df_dat[column], name= column, mode = line_markers), 
    secondary_y=secondary_y_flag, )
        


    #fit data add
    fig.add_trace( 
        go.Scatter(x= t_sim, y= df_fit["cS"], name= "cS_fitted", mode = line, marker = dict(color = "limegreen")), 
        secondary_y=False,)
    fig.add_trace( 
        go.Scatter(x= t_sim, y= df_fit["cX"], name= "cX_fitted", mode = line, marker = dict(color = "firebrick")), 
        secondary_y=False,)

    fig.add_trace( 
        go.Scatter(x= t_sim, y= df_fit["cE"], name= "cE_fitted", mode = line, marker = dict(color = "violet")), 
        secondary_y=False,)
    
    fig.add_trace( 
        go.Scatter(x= t_sim, y= df_fit["base_rate"], name= "base_rate_fitted", mode = line, marker = dict(color = "darkblue")), 
        secondary_y=True,)

    fig.add_trace( 
        go.Scatter(x= t_sim, y= df_fit["CO2"], name= "CO2_fitted", mode = line, marker = dict(color = "orange")), 
        secondary_y=True,)
    print("together with simulated model with p, y0, c in : ", key_fit)        
    fig.show()