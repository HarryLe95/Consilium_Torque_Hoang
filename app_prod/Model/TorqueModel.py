import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from functools import cached_property

#MODDED
def get_ramping_alerts(data: pd.DataFrame,
                       feature_name: str,
                       prediction_name: str,
                       polynomial_days:float,
                       polynomial_degree:float,
                       ramp_integral_threshold:float) -> pd.DataFrame:
    time_increments=data.index.to_series().resample('12h').max()    
    torque_tag=data[feature_name]
    torque_tag=torque_tag.interpolate(method='linear', axis=0)
    differential=pd.DataFrame(index=time_increments,columns=['polynomial_diff'])
    data[prediction_name]=0
    
    for time in time_increments:
        time_diff = time - pd.Timedelta(polynomial_days)
        if  time_diff > time_increments[0]:
            x = torque_tag.loc[time_diff:time]
            if not data['flush_status'].loc[time_diff:time].any():
                if not data['operation_status'].loc[time_diff:time].any():
                    u_limit=x.mean()+(2.5*x.std())
                    l_limit=x.mean()-(2.5*x.std())
                    x[x > u_limit] = u_limit
                    x[x < l_limit] = l_limit
                    poly=np.polyfit(list(range(len(x))),x,deg=polynomial_degree)
                    differential['polynomial_diff'].loc[time]=pd.DataFrame(np.polyval(poly, list(range(len(x))))).diff().mean()[0]
    integral=differential['polynomial_diff'].rolling(polynomial_days).sum()

    for time in differential.index:
        if time - pd.Timedelta(polynomial_days) > data.index[0]:
            if integral.loc[time]>0:
                if integral.loc[time]*differential['polynomial_diff'].loc[time]*100000>ramp_integral_threshold:
                    data.loc[time-pd.Timedelta('12hr'):time,[prediction_name]]=1
    return data

#MODDED
def get_torque_spike_decay_alert(data: pd.DataFrame,
                                 feature_name: str,
                                 prediction_name: str,
                                 window_size:float,
                                 window_overlap:float, 
                                 binary_threshold:float,
                                 model_path:str,
                                 merged_event_overlap: float,
                                 miminum_event_length: int) -> pd.DataFrame:
    model = keras.models.load_model(model_path, compile=False)

    feature_data = data[feature_name].interpolate()
    feature_data_idx = data[feature_name].index.values
    window_size = window_size
    window_overlap = window_overlap
    binary_threshold = binary_threshold

    num_windows = int(np.floor(feature_data.shape[0] / (window_size / 2)))
    window_starts = [int(i * window_size * (1 - window_overlap)) for i in range(num_windows)]
    window_ends = [int((i * window_size * (1 - window_overlap)) + window_size) for i in range(num_windows)]
    len_inference_timeframe = len(feature_data_idx)

    edge_effect_width = int(window_size / (2 ** 4))

    df = pd.DataFrame(index=feature_data_idx, columns=[prediction_name, f"{prediction_name}_tmp"])

    for window_start, window_end in tqdm(zip(window_starts, window_ends)):
        # get features (pad with zeros if not of length window_size)
        fdata = np.expand_dims(feature_data[window_start:window_end], 0)
        if fdata.shape[1] < window_size:
            fdata = np.pad(fdata,
                           ((0, 0),
                            (0, window_size - fdata.shape[1])),
                           'edge'
                           )

        # get predictions of positive class (i.e. a present BDTA event)
        prediction = model.predict(fdata)[0, :, 1]

        # account for edge effects, except at start and end of inference timeframe
        # start of inference timeframe
        if window_start == 0:
            s_idx = window_start
            e_idx = window_end - 1 - edge_effect_width

            s_pred_idx = 0
            e_pred_idx = len(prediction) - edge_effect_width
            # end of inference timeframe
        elif (window_end > len_inference_timeframe):
            s_idx = window_start + edge_effect_width
            e_idx = len_inference_timeframe - 1

            s_pred_idx = edge_effect_width
            e_pred_idx = window_size - (window_end - len_inference_timeframe)

        # within inference timeframe
        else:
            s_idx = window_start + edge_effect_width
            e_idx = window_end - 1 - edge_effect_width

            s_pred_idx = edge_effect_width
            e_pred_idx = len(prediction) - edge_effect_width

        # get datetime indices
        dt_s_idx = df.index[s_idx]
        dt_e_idx = df.index[e_idx]

        # collate predictions in dataframe
        df.loc[dt_s_idx:dt_e_idx, f"{prediction_name}_tmp"] = prediction[s_pred_idx:e_pred_idx]
        df.loc[dt_s_idx:dt_e_idx, prediction_name] = df.loc[dt_s_idx:dt_e_idx].max(axis=1)

    data[prediction_name] = (df[prediction_name].values > binary_threshold) * 1
    
    # filter datetimes
    event_datetimes = get_event_datetimes(data[prediction_name],
                                          merge_event_overlap=merged_event_overlap
                                         )
    
    if event_datetimes:
        print(f"Number of events detected: {len(event_datetimes)}")
        if (len(event_datetimes) > 0):
            mask = [((pd.to_datetime(event_datetime[1]) - pd.to_datetime(event_datetime[0])) > pd.Timedelta(minutes=miminum_event_length)) for event_datetime in event_datetimes]
            filtered_event_datetimes = [x for x, y in zip(event_datetimes, mask) if y == True]

            data[f"{prediction_name}_filtered"] = 0
            for (start_datetime, end_datetime) in filtered_event_datetimes:
                data[f"{prediction_name}_filtered"].loc[start_datetime:end_datetime] = 1

            data[prediction_name] = data[f"{prediction_name}_filtered"]
            data = data.drop(columns=[f"{prediction_name}_filtered"])

    return data

def which_feature(data, feature_cols):
    """
    Decide which feature to use.
    """
    vals=[]
    for t in feature_cols:
        vals.append(data[t].count())
    return feature_cols[np.argmax(vals)]

#MODDED
def get_operation_status(data: pd.DataFrame,
                         torque_feature: str,
                         off_threshold: float):

    off_trace = pd.Series(index=data.index, name='operation_method',data=0)
    torque_data = data[torque_feature].interpolate(method='linear', axis=0)
    off_times = torque_data[torque_data < off_threshold].index
    prev_point = data.index[0] - pd.Timedelta('6H')
    for point in off_times:
        if point > (prev_point + pd.Timedelta('5H')):
            off_trace.loc[point - pd.Timedelta('62min'):point + pd.Timedelta('5H')] = 1
            prev_point = point
    data['operation_status'] = off_trace
    return data

#MODDED
def get_flush_status(data: pd.DataFrame,
                     speed_feature: str,
                     flush_diff_threshold:float,
                     flush_std_threshold:float) -> pd.DataFrame:
    flush_trace = pd.Series(index=data.index, name = "flush_status", data=0)
    speed_data = data[speed_feature].interpolate(method='linear', axis=0)
    flush_diff = speed_data[speed_data.diff(1).abs().rolling('2H').sum() > flush_diff_threshold]
    flush_std  = speed_data[speed_data.rolling('2H').std() > flush_std_threshold]
    exlude_ind = flush_diff.index.intersection(flush_std.index)
    prev_point = data.index[0] - pd.Timedelta('6H')
    for point in exlude_ind:
        if point > (prev_point + pd.Timedelta('5H')):
            flush_trace.loc[point - pd.Timedelta('180min'):point + pd.Timedelta('12H')] = 1
            prev_point = point
    data["flush_status"] = flush_trace
    return data

def get_combined_alert(data: pd.DataFrame, alert_names: list, method: str = 'all') -> pd.DataFrame:
    if method == 'all':
        data['combined_alert'] = (data[alert_names] == 1).all(1).astype(int)
    elif method == 'any':
        data['combined_alert'] = (data[alert_names] == 1).any(1).astype(int)
    else:
        data['combined_alert'] = 0

    return data

def get_event_datetimes(event_data: pd.Series, merge_event_overlap: int = None):
    """
    param data:
    param merge_event_overlap:
    return event_datetimes:
    """
    inference_window_start_datetime = event_data.index[0]
    inference_window_end_datetime = event_data.index[-1]

    event_starts = event_data[event_data.diff() == 1].index.tolist()  # transition into an event
    event_ends = event_data[event_data.diff() == -1].index.tolist()  # transition out of an event

    if (len(event_starts) == 0) | (len(event_ends) == 0):
        return None

    # starts within event
    if len(event_starts) < len(event_ends):
        event_starts.insert(0, inference_window_start_datetime)

    # ends within event
    if len(event_starts) > len(event_ends):
        event_ends.append(inference_window_end_datetime)

    # starts and ends within event
    if len(event_starts) == len(event_ends):
        if event_starts[0] > event_ends[0]:
            event_starts.insert(0, inference_window_start_datetime)
            event_ends.append(inference_window_end_datetime)

    event_datetimes = [(s, e) for s, e in zip(event_starts, event_ends)]

    return event_datetimes if len(event_datetimes) > 0 else None

class Model:
    def __init__(self, 
                 model_path:str,
                 use_operation_status:bool=True,
                 use_flush_status:bool=True,
                 use_ramping_alert:bool=True,
                 use_torque_spike_alert:bool=True,
                 time_window:str='1d',
                 off_threshold:float=10, 
                 flush_diff_threshold:float=4.9,
                 flush_std_threshold:float=4,
                 polynomial_days:str='7d',
                 polynomial_degree:float=4,
                 ramp_integral_threshold:float=0.7,
                 window_size:float=2**14,
                 window_overlap:float=0.5, 
                 binary_threshold:float=0.5,
                 merged_event_overlap: float=10,
                 miminum_event_length: int=0):
        
        self.use_operation_status = use_operation_status
        self.use_flush_status = use_flush_status
        self.use_ramping_alert = use_ramping_alert
        self.use_torque_spike_alert = use_torque_spike_alert
        
        self.time_window = time_window
        self.off_threshold = off_threshold
        self.flush_diff_threshold = flush_diff_threshold
        self.flush_std_threshold = flush_std_threshold
        
        self.polynomial_days = polynomial_days
        self.polynomial_degree = polynomial_degree
        self.ramp_integral_threshold = ramp_integral_threshold
        
        self.window_size = window_size 
        self.window_overlap = window_overlap
        self.binary_threshold = binary_threshold
        self.model_path = model_path
        self.merged_evant_overlap = merged_event_overlap
        self.minimum_event_length = miminum_event_length
    
    def get_rolling_average_torque(self, torque_column, inference_data):
        inference_data[f"{torque_column}_roll_avg"] = inference_data[torque_column].interpolate().rolling(120).mean()
        return inference_data
        
    
    def get_operation_status(self, torque_column,inference_data):
        #OPERATION STATUS
        if self.use_operation_status:
            return get_operation_status(inference_data, torque_column, self.off_threshold)
        return inference_data
    
    def get_flush_status(self, speed_column, inference_data):
        #FLUSH STATUS
        if self.use_flush_status:
            return get_flush_status(inference_data, speed_column, self.flush_diff_threshold, self.flush_std_threshold)
        return inference_data 
    
    def get_ramping_alert(self, torque_column, inference_data):
        #RAMPING ALERT
        if self.use_ramping_alert:
            return get_ramping_alerts(inference_data, torque_column, "ramping_alert", self.polynomial_days, self.polynomial_degree, self.ramp_integral_threshold)
        return inference_data
    
    def get_spike_alert(self, torque_column, inference_data):
        if self.use_torque_spike_alert:
            return get_torque_spike_decay_alert(inference_data, torque_column, "torque_spike_decay_alert", self.window_size, self.window_overlap, self.binary_threshold, self.model_path, self.merged_evant_overlap, self.minimum_event_length)
        return inference_data
    
    def get_combined_alert(self, inference_data:pd.DataFrame):
        inference_data['operation_status'] = 1 - inference_data['operation_status']
        inference_data['flush_status'] = 1 - inference_data['flush_status'] 
        inference_data = get_combined_alert(inference_data, alert_names=['torque_spike_decay_alert', 'flush_status','operation_status'])        
        return inference_data
    
    def get_alert_count(self, inference_data:pd.DataFrame, alert_type:str, period:str)->pd.Series:
        return inference_data[alert_type].resample(period).sum()
    
    def alert_monitor_generation_all_wells(self,
                                           well_cd: str, 
                                           inference_data:pd.DataFrame,
                                           completion_turndown_df: pd.DataFrame,
                                           label_data_df: pd.DataFrame) -> pd.DataFrame:

        start_date  = inference_data.index.min()
        end_date    = inference_data.index.max()
        window_start= pd.to_datetime(end_date)-pd.Timedelta(self.time_window)
        window_end  = end_date
        date = end_date 
        data = inference_data
        day_index   = pd.date_range(start_date,end_date, freq='1h')
        
        todays_results_df=pd.DataFrame(index=[well_cd],
                                       columns=[f'Spiking Present (past {self.time_window}ay/s)',f'Ramping Present (past {self.time_window}ay/s)',
                                                'Spiking Present (past month)',
                                                'Ramping Present (past month)',                                             
                                                f'Spike Alert Percent Active Time in Past {self.time_window}ay/s',
                                                'Spike Alert Percent Active Time in Past Week',
                                                'Spike Alert Percent Active Time in Past 30 Days',
                                                'Spike Alert Percent Active Time in Past 60 Days',
                                                f'Ramp Alert Percent Active Time in Past {self.time_window}ay/s',
                                                'Ramp Alert Percent Active Time in Past Week',
                                                'Ramp Alert Percent Active Time in Past 30 Days',
                                                'Ramp Alert Percent Active Time in Past 60 Days',
                                                'Average gas flow (60 Days)',
                                                'TD status',
                                                'Design',
                                                'Pump age',
                                                'Days since last flush',
                                                'Days since last failure',
                                                f'Total Alert on minutes in Past {self.time_window}ay/s',
                                                'Total Alert on minutes in Past Week',
                                                'Total Alert on minutes in Past 30 Days',
                                                'Total Alert on minutes in Past 60 Days',
                                                f'Spike Alert on minutes in Past {self.time_window}ay/s',
                                                'Spike Alert on minutes in Past Week',
                                                'Spike Alert on minutes in Past 30 Days',
                                                'Spike Alert on minutes in Past 60 Days',
                                                f'Ramp Alert on minutes in Past {self.time_window}ay/s',
                                                'Ramp Alert on minutes in Past Week',
                                                'Ramp Alert on minutes in Past 30 Days',
                                                'Ramp Alert on minutes in Past 60 Days',])
        
        torque_names = ["TORQUE_MOTOR", "TORQUE_ROD"]
        torque_col = which_feature(inference_data, torque_names)
        speed_col = "SPEED_ROD"
        
        
        #ROLLING AVG Torque
        data = self.get_rolling_average_torque(torque_col,data)
        #OPERATION STATUS
        data = self.get_operation_status(torque_col, data)
        #FLUSH STATUS
        data = self.get_flush_status(data, speed_col, data)
        #RAMPING ALERT
        data = self.get_ramping_alert(torque_col, data)
        #SPIKE ALERT
        data = self.get_spike_alert(torque_col, data)
        #COMBINED ALERT
        data = self.get_combined_alert(data)
        
        peak_label_minute = self.get_alert_count(data, "combined_alert","1h", )
        ramp_label_minute = self.get_alert_count(data, "ramping_alert", "1h")
        total_label_minute = peak_label_minute + ramp_label_minute
        
        peak_label_hour = peak_label_minute.reindex(day_index, method = 'nearest')
        ramp_label_hour = ramp_label_minute.reindex(day_index, method = 'nearest')
        total_label_hour = total_label_minute.reindex(day_index, method = 'nearest')
        


        if all_well_total_label_minute[well_cd].loc[:pd.to_datetime(date)].sum()>0:
            if  peak_label_minute.loc[pd.to_datetime(date)-pd.Timedelta(self.time_window):pd.to_datetime(date)].sum().values[0]>0:
                todays_results_df.loc[well_cd,[f'Spiking Present (past {self.time_window}ay/s)']]='True'
            else:
                todays_results_df.loc[well_cd,[f'Spiking Present (past {self.time_window}ay/s)']]='False'
            if  ramp_label_minute.loc[pd.to_datetime(date)-pd.Timedelta(self.time_window):pd.to_datetime(date)].sum().values[0]>0:
                todays_results_df.loc[well_cd,[f'Ramping Present (past {self.time_window}ay/s)']]='True'
            else:
                todays_results_df.loc[well_cd,[f'Ramping Present (past {self.time_window}ay/s)']]='False' 
                
                
            if  peak_label_minute.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]>0:
                todays_results_df.loc[well_cd,['Spiking Present (past month)']]='True'
            else:
                todays_results_df.loc[well_cd,['Spiking Present (past month)']]='False'
            if  ramp_label_minute.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]>0:
                todays_results_df.loc[well_cd,['Ramping Present (past month)']]='True'
            else:
                todays_results_df.loc[well_cd,['Ramping Present (past month)']]='False' 
                    

        todays_results_df.loc[well_cd,[f'Total Alert on minutes in Past {self.time_window}ay/s']] = total_label_minute.loc[pd.to_datetime(date)-pd.Timedelta(self.time_window):pd.to_datetime(date)].sum().values[0]#+ramp_label_minute.loc[pd.to_datetime(date)-pd.Timedelta(self.time_window):pd.to_datetime(date)].sum().values[0]
        todays_results_df.loc[well_cd,['Total Alert on minutes in Past Week']] = total_label_minute.loc[pd.to_datetime(date)-pd.Timedelta('7d'):pd.to_datetime(date)].sum().values[0]#+ramp_label_minute.loc[pd.to_datetime(date)-pd.Timedelta('7d'):pd.to_datetime(date)].sum().values[0]
        todays_results_df.loc[well_cd,['Total Alert on minutes in Past 30 Days']] = total_label_minute.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]#+ramp_label_minute.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]
        todays_results_df.loc[well_cd,['Total Alert on minutes in Past 60 Days']] = total_label_minute.loc[start_date:pd.to_datetime(date)].sum().values[0]#+ramp_label_minute.loc[start_date:pd.to_datetime(date)].sum().values[0]

        todays_results_df.loc[well_cd,[f'Spike Alert on minutes in Past {self.time_window}ay/s']] = peak_label_minute.loc[pd.to_datetime(date)-pd.Timedelta(self.time_window):pd.to_datetime(date)].sum().values[0]
        todays_results_df.loc[well_cd,['Spike Alert on minutes in Past Week']] = peak_label_minute.loc[pd.to_datetime(date)-pd.Timedelta('7d'):pd.to_datetime(date)].sum().values[0]
        todays_results_df.loc[well_cd,['Spike Alert on minutes in Past 30 Days']] = peak_label_minute.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]
        todays_results_df.loc[well_cd,['Spike Alert on minutes in Past 60 Days']] = peak_label_minute.loc[start_date:pd.to_datetime(date)].sum().values[0]

        todays_results_df.loc[well_cd,[f'Ramp Alert on minutes in Past {self.time_window}ay/s']] = ramp_label_minute.loc[pd.to_datetime(date)-pd.Timedelta(self.time_window):pd.to_datetime(date)].sum().values[0]
        todays_results_df.loc[well_cd,['Ramp Alert on minutes in Past Week']] = ramp_label_minute.loc[pd.to_datetime(date)-pd.Timedelta('7d'):pd.to_datetime(date)].sum().values[0]
        todays_results_df.loc[well_cd,['Ramp Alert on minutes in Past 30 Days']] = ramp_label_minute.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]
        todays_results_df.loc[well_cd,['Ramp Alert on minutes in Past 60 Days']] = ramp_label_minute.loc[start_date:pd.to_datetime(date)].sum().values[0]

        todays_results_df.loc[well_cd,['Average gas flow (60 Days)']] = data.loc[start_date:date,["FLOW_GAS"]].interpolate("ffill").mean().values[0]
        todays_results_df.loc[well_cd,['TD status']] = completion_turndown_df[completion_turndown_df['WellCD']==well_cd]['Turndown Cat.'].values[0]
        todays_results_df.loc[well_cd,['Design']] = completion_turndown_df[completion_turndown_df['WellCD']==well_cd]['Completion Design'].values[0]
        todays_results_df.loc[well_cd,['Pump age']] = pd.to_datetime(date)-label_data_df[(label_data_df['WellCD']==well_cd) & (label_data_df.index < date)&(label_data_df['Event']=='Pump Change')].index.max()
        todays_results_df.loc[well_cd,['Days since last flush']] = pd.to_datetime(date)-label_data_df[(label_data_df['WellCD']==well_cd) & (label_data_df.index < date)&((label_data_df['Event']=='Scheduled')|(label_data_df['Event']=='Reactive')|(label_data_df['Event']=='Flushby'))].index.max()
        todays_results_df.loc[well_cd,['Days since last failure']] = pd.to_datetime(date)-label_data_df[(label_data_df['WellCD']==well_cd) & (label_data_df.index < date) & (label_data_df['Failure Mode'].notnull())].index.max()       

        todays_results_df.loc[well_cd,[f'Spike Alert Percent Active Time in Past {self.time_window}ay/s']]=(todays_results_df.loc[well_cd,[f'Spike Alert on minutes in Past {self.time_window}ay/s']].values[0]/(pd.Timedelta(self.time_window).total_seconds() / 60))*100
        todays_results_df.loc[well_cd,['Spike Alert Percent Active Time in Past Week']]=(todays_results_df.loc[well_cd,['Spike Alert on minutes in Past Week']].values[0] / 10080)*100
        todays_results_df.loc[well_cd,['Spike Alert Percent Active Time in Past 30 Days']]=(todays_results_df.loc[well_cd,['Spike Alert on minutes in Past 30 Days']].values[0]/43200)*100
        todays_results_df.loc[well_cd,['Spike Alert Percent Active Time in Past 60 Days']]=(todays_results_df.loc[well_cd,['Spike Alert on minutes in Past 60 Days']].values[0]/86400)*100
                            
        todays_results_df.loc[well_cd,[f'Ramp Alert Percent Active Time in Past {self.time_window}ay/s']]=(todays_results_df.loc[well_cd,[f'Ramp Alert on minutes in Past {self.time_window}ay/s']].values[0]/(pd.Timedelta(self.time_window).total_seconds() / 60))*100
        todays_results_df.loc[well_cd,['Ramp Alert Percent Active Time in Past Week']]=(todays_results_df.loc[well_cd,['Ramp Alert on minutes in Past Week']].values[0] / 10080)*100
        todays_results_df.loc[well_cd,['Ramp Alert Percent Active Time in Past 30 Days']]=(todays_results_df.loc[well_cd,['Ramp Alert on minutes in Past 30 Days']].values[0]/43200)*100
        todays_results_df.loc[well_cd,['Ramp Alert Percent Active Time in Past 60 Days']]=(todays_results_df.loc[well_cd,['Ramp Alert on minutes in Past 60 Days']].values[0]/86400)*100
                    

        todays_results_df=todays_results_df.loc[pd.DataFrame(all_well_total_label_minute.loc[pd.to_datetime(date)-pd.Timedelta(self.time_window):pd.to_datetime(date)].sum())[all_well_total_label_minute.loc[pd.to_datetime(date)-pd.Timedelta(self.time_window):pd.to_datetime(date)].sum()>0].index]
        return todays_results_df.sort_values(by=f'Total Alert on minutes in Past {self.time_window}ay/s',ascending=False)

