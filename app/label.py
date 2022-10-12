import pandas as pd 
import numpy as np 
from src.utils.PathManager import Paths as Path  

def generate_labels(well_id, 
                    feat, 
                    speed_feat,
                    polynomial_days='7d', 
                    polynomial_degree=4, 
                    off_threshold=10, 
                    flush_diff_threshold=4.9, 
                    flush_std_threshold=4, 
                    spike_diff_threshold=0.021,
                    diff_compare_window=60,
                    ramp_integral_threshold=0.7):
    print(well_id)
    tag_df=pd.read_pickle(Path.data(f"{well_id}_2016-01-01_2023-01-01_raw.pkl"))
    tag_df=tag_df.asfreq('T', method='ffill')#.resample('1min').mean()#
    tag_df=tag_df.interpolate(method='linear', axis=0)

    ########### Create well off and flush traces######################
    
    
    #remove well off periods. This also removes an area four hours after the well was off. 
    #This is because there is often a start up where the torque may spike when the well is turned back on and we do not wish to label this.
    off_trace=pd.DataFrame(index=tag_df.index, columns=['off'])
    off_times=tag_df[tag_df[feat]<off_threshold]
    for point in off_times.index:
        off_trace['off'].loc[point-pd.Timedelta("30min"):point+pd.Timedelta("4H")]=1
    print('off trace done')

    #The flush trace is for excluding spikes in the torque that are due to flushes being performed.
    #Additionally this excludes the times when the speed of the rod is changed. 
    flush_trace=pd.DataFrame(index=tag_df.index, columns=['flush'])
    flush_diff=tag_df[tag_df[speed_feat].diff(1).abs().rolling("2H").sum()>flush_diff_threshold]
    flush_std=tag_df[tag_df[speed_feat].rolling("2H").std()>flush_std_threshold]
    exlude_ind=flush_diff.index.intersection(flush_std.index)
    for point in exlude_ind:
        flush_trace['flush'].loc[point-pd.Timedelta("30min"):point+pd.Timedelta("1H")]=1
    print('flush trace done')

     #######################################################   
        

    ######### Create torque ramp alert trace ###############    
     
    ## The method used here is to every 12 hours (or frequency set in "time_increments") 
    ## take a time window leading upto the current time of "polynomial_days" in length.
    ## The off and flush times are excluded and the extremes of the data are limited 
    ## (to assist with the polynomial doing strange things). A polynomial of "polynomial_degree" is then fitted.
    ## The average value of the differential of the polynomial in this time window is taken. This makes the "differential" trace. 
    ## The sum of the differential trace is taken using a rolling window (length polynomial days) to create the integral trace.
    ## Finally if the integral trace is greater than zero and the product of the integral and differential trace is above 
    ## "ramp_integral_threshold" the time is flagged as ramping.
        
    time_increments=tag_df.index.to_series().resample('12h').max()    
        
    differential=pd.DataFrame(index=time_increments,columns=['polynomial_diff'])
    ramping_alert_df=pd.DataFrame(index=differential.index, columns= ['ramp_label'])
    for time in time_increments:
        if time -pd.Timedelta(polynomial_days) > time_increments[0]:
            x=tag_df[feat].loc[time-pd.Timedelta(polynomial_days):time]
            x=x[flush_trace['flush'].loc[time-pd.Timedelta(polynomial_days):time].isna()]
            x=x[off_trace['off'].loc[time-pd.Timedelta(polynomial_days):time].isna()]
            if len(x)>9360:
                u_limit=x.mean()+(2.5*x.std())
                l_limit=x.mean()-(2.5*x.std())
                x[x > u_limit] = u_limit
                x[x < l_limit] = l_limit
                poly=np.polyfit(list(range(len(x))),x,deg=polynomial_degree)
                differential['polynomial_diff'].loc[time]=pd.DataFrame(np.polyval(poly, list(range(len(x))))).diff().mean()[0]
    print('poly fit done')
             

    integral=differential['polynomial_diff'].rolling(polynomial_days).sum()
            
   
    for time in ramping_alert_df.index:
        if time -pd.Timedelta(polynomial_days) > ramping_alert_df.index[0]:
            if integral.loc[time]>0:
                if integral.loc[time]*differential['polynomial_diff'].loc[time]*100000>ramp_integral_threshold:
                    ramping_alert_df.loc[time]=1
    ramping_alert_df=ramping_alert_df.fillna(0)            
    print('ramping alert done')

     #######################################################   
    

    ######### Create torque spike alert trace ###############    
    
    ## The method used here is to take a rolling window of two hours and take the standard deviation. 
    ## The difference between the current time std and std "diff_compare_window" ago 
    ## is taken and this is then scaled to the value of the torque.
    ## A threshold is applied to this of "spike_diff_threshold" and the spike_label_trace is created by assigning a value of 1 
    ## to a time window from an hour before up until the detected spike.
    
    tag_df=pd.read_pickle(Path.data(f"{well_id}_2016-01-01_2023-01-01_raw.pkl"))
    tag_df=tag_df.asfreq('T', method='nearest')

    tag_df.loc[off_trace['off']==1,[feat]]=np.nan
    tag_df.loc[flush_trace['flush']==1,[feat]]=np.nan
    tag_df=tag_df[~tag_df[feat].isna()]
    peaks=tag_df[feat].rolling(120).std().diff(diff_compare_window)/tag_df[feat].rolling(120).mean()
    peak_df=peaks.to_frame()
    spike_label_df=pd.DataFrame(index=tag_df.index,columns=['spike_label'])
    for peak in peaks[peaks>spike_diff_threshold].to_frame().index:
        spike_label_df.loc[peak-pd.Timedelta("1H"):peak,['spike_label']]=1
    spike_label_df[spike_label_df.isna()]=0
     #######################################################   
    
    spike_label_df.to_pickle(Path.data(f"{well_id}_2016-01-01_2023-01-01_spike_label.pkl"))
    ramping_alert_df.to_pickle(Path.data(f"{well_id}_2016-01-01_2023-01-01_ramp_label.pkl"))
    return spike_label_df, ramping_alert_df