import pandas as pd 
import numpy as np 

class High_Torque_Model:
    def __init__(self):
        pass 
    
    def get_ramping_alerts(well_data_df: pd.DataFrame,
                           feature_name: str,
                           prediction_name: str,
                           config_dict: dict) -> pd.DataFrame:
        """
        get_ramping_alerts finds ramps for a given well.

        :param well_data_df: dataframe containing all the data for a single well over which inference will be executed.
        :param feature_name: the column name corresponding to features that is to be used in model inference.
        :param prediction_name: the column name corresponding to prediction in which model inference will be recorded.
        :param config_dict: a dictionary of model specific configuration (e.g. thresholds, etc.)
        :return: ...
        """
        #polynomial_days, flush_config - operation_method, operations_config - operation_method, polynomial_degree, ramp_integral threshold    
        time_increments=well_data_df.index.to_series().resample('12h').max()    
        torque_tag=well_data_df[feature_name]
        torque_tag=torque_tag.interpolate(method='linear', axis=0)
        differential=pd.DataFrame(index=time_increments,columns=['polynomial_diff'])
        well_data_df[prediction_name]=0
        for time in time_increments:
            if time - pd.Timedelta(config_dict['polynomial_days']) > time_increments[0]:
                x=torque_tag.loc[time-pd.Timedelta(config_dict['polynomial_days']):time]
                if not well_data_df[config_dict['flush_config']['operation_method']].loc[time-pd.Timedelta(config_dict['polynomial_days']):time].any():
                    if not well_data_df[config_dict['operations_config']['operation_method']].loc[time-pd.Timedelta(config_dict['polynomial_days']):time].any():

                        u_limit=x.mean()+(2.5*x.std())
                        l_limit=x.mean()-(2.5*x.std())
                        x[x > u_limit] = u_limit
                        x[x < l_limit] = l_limit
                        poly=np.polyfit(list(range(len(x))),x,deg=config_dict['polynomial_degree'])
                        differential['polynomial_diff'].loc[time]=pd.DataFrame(np.polyval(poly, list(range(len(x))))).diff().mean()[0]
        integral=differential['polynomial_diff'].rolling(config_dict['polynomial_days']).sum()

        for time in differential.index:
            if time - pd.Timedelta(config_dict['polynomial_days']) > well_data_df.index[0]:
                if integral.loc[time]>0:
                    if integral.loc[time]*differential['polynomial_diff'].loc[time]*100000>config_dict['ramp_integral_threshold']:
                        well_data_df.loc[time-pd.Timedelta('12hr'):time,[prediction_name]]=1
        return well_data_df
