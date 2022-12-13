import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

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

def get_torque_spike_decay_alert(well_data_df: pd.DataFrame,
                                 feature_name: str,
                                 prediction_name: str,
                                 model_config: dict) -> pd.DataFrame:
    """
    get_spike_decay finds the torque spike decay regions for a given well.

    :param well_data_df: dataframe containing all the data for a single well over which inference will be executed.
    :param feature_name: the column name corresponding to feature that is to be used in model inference.
    :param prediction_name: the column name corresponding to prediction in which model inference will be recorded.
    :param model_config: a dictionary of model specific configuration (e.g. thresholds, etc.)
    :return: ...
    """
    model = keras.models.load_model(model_config['model_path'], compile=False)

    feature_data = well_data_df[feature_name].interpolate()
    feature_data_idx = well_data_df[feature_name].index.values
    window_size = model_config['window_size']
    window_overlap = model_config['window_overlap']
    binary_threshold = model_config['binary_threshold']

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

    well_data_df[prediction_name] = (df[prediction_name].values > binary_threshold) * 1
    
    # filter datetimes
    event_datetimes = get_event_datetimes(well_data_df[prediction_name],
                                          merge_event_overlap=model_config['merged_event_overlap']
                                         )
    
    if event_datetimes:
        print(f"Number of events detected: {len(event_datetimes)}")
        if (len(event_datetimes) > 0):
            mask = [((pd.to_datetime(event_datetime[1]) - pd.to_datetime(event_datetime[0])) > pd.Timedelta(minutes=model_config['miminum_event_length'])) for event_datetime in event_datetimes]
            filtered_event_datetimes = [x for x, y in zip(event_datetimes, mask) if y == True]

            well_data_df[f"{prediction_name}_filtered"] = 0
            for (start_datetime, end_datetime) in filtered_event_datetimes:
                well_data_df[f"{prediction_name}_filtered"].loc[start_datetime:end_datetime] = 1

            well_data_df[prediction_name] = well_data_df[f"{prediction_name}_filtered"]
            well_data_df = well_data_df.drop(columns=[f"{prediction_name}_filtered"])

    return well_data_df

#DEPRECATED
def load_data_from_disk(data_definition_dict: dict) -> pd.DataFrame:
    """
    Load raw data from CSV file
    
    
    """
    well_data_df = pd.read_csv(data_definition_dict['well_raw_data_path'])
    
    if 'Unnamed: 0' in well_data_df.columns:
        well_data_df.drop(columns=['Unnamed: 0'],inplace=True)

    well_data_df['TS'] = pd.to_datetime(well_data_df['TS'], format="%Y-%m-%d %H:%M:%S")
    well_data_df.set_index(['TS'], inplace=True, drop=True)
    
    well_data_df = well_data_df.loc[(well_data_df.index > data_definition_dict['start_date']) & (well_data_df.index < data_definition_dict['end_date'])]
    return well_data_df

def which_feature(well_data_df, feature_cols):
    """
    Decide which feature to use.
    """
    vals=[]
    for t in feature_cols:
        vals.append(well_data_df[t].count())
    return feature_cols[np.argmax(vals)]

#DEPRECATED
def interpolate(well_data_df: pd.DataFrame) -> pd.DataFrame:
    return well_data_df.interpolate(method='linear', axis=0)

#DEPRECATED
def resample_to_1min(well_data_df: pd.DataFrame) -> pd.DataFrame:
    well_data_df=well_data_df.groupby(['TS']).mean()
    return well_data_df.asfreq('T', method='nearest').resample('1min').mean()

#DEPRECATED
def load_labels_from_disk(label_definition_dict: dict) -> pd.DataFrame:
    """
    Load labels from CSV file
    
    """
    label_df = pd.read_csv(label_definition_dict['label_data_path'])
    
    label_df = label_df[label_df['WellCD'] == label_definition_dict['well_id']]

    label_df['Event Date'] = pd.to_datetime(label_df['Event Date'], format="%d/%m/%Y")

    label_df.dropna(subset=['Event Date'], inplace=True)
    
    label_df = label_df[(label_df['Event Date'] > label_definition_dict['start_date']) & (label_df['Event Date'] < label_definition_dict['end_date'])]
    
    label_df.set_index('Event Date', inplace=True)

    print(f"Label data loaded for {label_definition_dict['well_id']} for date range {label_definition_dict['start_date']} - {label_definition_dict['end_date']}")

    return label_df

def get_operation_status(well_data_df: pd.DataFrame,
                         feature_name: str,
                         config_dict: dict):
    """
    get_off_times gets...

    :param well_data_df: dataframe containing all the data for a single well over which inference will be executed.
    :param config_dict: a dictionary of model specific configuration (e.g. thresholds, etc.)
    :return: ...
    """

    off_trace = pd.DataFrame(index=well_data_df.index, columns=[config_dict['operation_method']])
    off_trace[config_dict['operation_method']] = 0
    torque_well_data_df = well_data_df[feature_name].interpolate(method='linear', axis=0)
    off_times = torque_well_data_df[torque_well_data_df < config_dict['off_threshold']].index
    prev_point = well_data_df.index[0] - pd.Timedelta('6H')
    for point in off_times:
        if point > (prev_point + pd.Timedelta('5H')):
            off_trace[config_dict['operation_method']].loc[point - pd.Timedelta('62min'):point + pd.Timedelta('5H')] = 1
            prev_point = point
    well_data_df[config_dict['operation_method']] = off_trace

    return well_data_df

def get_flush_status(well_data_df: pd.DataFrame,
                     feature_name: str,
                     config_dict: dict) -> pd.DataFrame:
    """
    get_flush_status gets...

    :param well_data_df: dataframe containing all the data for a single well over which inference will be executed.
    :param config_dict: a dictionary of model specific configuration (e.g. thresholds, etc.)
    :return: ...
    """

    flush_trace = pd.DataFrame(index=well_data_df.index, columns=[config_dict['operation_method']])
    flush_trace[config_dict['operation_method']] = 0
    speed_well_data_df = well_data_df[feature_name].interpolate(method='linear', axis=0)
    flush_diff = speed_well_data_df[speed_well_data_df.diff(1).abs().rolling('2H').sum() > config_dict['flush_diff_threshold']]
    flush_std = speed_well_data_df[speed_well_data_df.rolling('2H').std() > config_dict['flush_std_threshold']]
    exlude_ind = flush_diff.index.intersection(flush_std.index)
    prev_point = well_data_df.index[0] - pd.Timedelta('6H')
    for point in exlude_ind:
        if point > (prev_point + pd.Timedelta('5H')):
            flush_trace[config_dict['operation_method']].loc[point - pd.Timedelta('180min'):point + pd.Timedelta('12H')] = 1
            prev_point = point
    well_data_df[config_dict['operation_method']] = flush_trace

    return well_data_df

def get_combined_alert(well_data_df: pd.DataFrame, alert_names: list, method: str = 'all') -> pd.DataFrame:
    """
    """

    if method == 'all':
        well_data_df['combined_alert'] = (well_data_df[alert_names] == 1).all(1).astype(int)
    elif method == 'any':
        well_data_df['combined_alert'] = (well_data_df[alert_names] == 1).any(1).astype(int)
    else:
        well_data_df['combined_alert'] = 0

    return well_data_df


def get_static_plot_for_single_alert(well_id: str,
                                     well_data_df: pd.DataFrame,
                                     features_to_plot: list,
                                     alert_names: str,
                                     label_data_df: pd.DataFrame = None,
                                     plot_type: str = 'scatter',
                                     ylim: list = [0, 1000],
                                     save_fig_fname: str = None
                                    ):
    """
    
    """
    
    fig = plt.figure(figsize=(50,8))
    
    for feature in features_to_plot:
        if feature in well_data_df.columns:
            if plot_type == 'scatter':
                plt.scatter(well_data_df.index, well_data_df[feature])
            if plot_type == 'line':
                tmp_plot_df = well_data_df.dropna(subset=[feature])
                plt.plot(tmp_plot_df.index, tmp_plot_df[feature])
    for alert in alert_names:
        plt.fill_between(x=well_data_df.index,
                         y1=ylim[0], #0,
                         y2=ylim[1], #int(well_data_df[features_to_plot[0]].dropna().max()*1.1),
                         where=well_data_df[alert],
#                          color='orange',
                         alpha=0.5,
                        )

    if not label_data_df.empty:
        for j, (idx, row) in enumerate(label_data_df.iterrows()):
            plt.vlines(x=idx,
                       ymin=ylim[0], #0,
                       ymax=ylim[1], #int(well_data_df[features_to_plot[0]].dropna().max()*1.1),
                       color='r'
                      )
            plt.text(x=idx,
                     y=int(ylim[1] * (0.6 + (0.01 * j))), #int(well_data_df[features_to_plot[0]].dropna().max()),
                     s=f"{row['Event']} - {row['Failure Mode']} - {row['Flush_Comment']}",
#                      rotation=90,
#                      verticalalignment='center'
                    )

    plt.rcParams.update({'font.size': 22})

    plt.legend(features_to_plot + alert_names)#[f"ALERT: {alert_names}"])
    plt.title(f"{well_id}")
    
    plt.ylim(ylim)
#     plt.show()
    
    if save_fig_fname:
        plt.savefig(save_fig_fname,
#                     *, dpi='figure', format=None, metadata=None,
#                     bbox_inches=None, pad_inches=0.1,
#                     facecolor='auto', edgecolor='auto',
#                     backend=None, **kwargs
                   )
        print(f"Figure saved to: {save_fig_fname}")

    return fig


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

def alert_monitor_generation_all_wells(date: str,
                                       time_window: str,
                                       well_list: list,
                                       plot: bool,
                                       completion_turndown_df: pd.DataFrame,
                                       label_data_df: pd.DataFrame) -> pd.DataFrame:
    """Get alerts for all wells 

    Args:
        date (str): inference day
        time_window (str): time window 
        well_list (list): list of inference wells 
        plot (bool): whether to plot the data 
        completion_turndown_df (pd.DataFrame): completion summary
        label_data_df (pd.DataFrame): label data summary 

    Returns:
        pd.DataFrame: _description_
    """
    start_date=pd.to_datetime(date)-pd.Timedelta('60d')
    end_date=pd.to_datetime(date)
    day_index=pd.date_range(start_date,end_date, freq='1h')
    all_well_spike_alert_df=pd.DataFrame(columns=well_list,index=day_index)
    all_well_ramp_alert_df=pd.DataFrame(columns=well_list,index=day_index)
    all_well_total_alert_df=pd.DataFrame(columns=well_list,index=day_index)
    todays_results_df=pd.DataFrame(index=well_list,columns=[
                                                            f'Spiking Present (past {time_window}ay/s)',f'Ramping Present (past {time_window}ay/s)',
                                                            'Spiking Present (past month)',
                                                            'Ramping Present (past month)',                                             
                                                            f'Spike Alert Percent Active Time in Past {time_window}ay/s',
                                                            'Spike Alert Percent Active Time in Past Week',
                                                            'Spike Alert Percent Active Time in Past 30 Days',
                                                            'Spike Alert Percent Active Time in Past 60 Days',
                                                            f'Ramp Alert Percent Active Time in Past {time_window}ay/s',
                                                            'Ramp Alert Percent Active Time in Past Week',
                                                            'Ramp Alert Percent Active Time in Past 30 Days',
                                                            'Ramp Alert Percent Active Time in Past 60 Days',
                                                            'Average gas flow (60 Days)',
                                                            'TD status',
                                                            'Design',
                                                            'Pump age',
                                                            'Days since last flush',
                                                            'Days since last failure',
                                                            f'Total Alert on minutes in Past {time_window}ay/s',
                                                            'Total Alert on minutes in Past Week',
                                                            'Total Alert on minutes in Past 30 Days',
                                                            'Total Alert on minutes in Past 60 Days',
                                                            f'Spike Alert on minutes in Past {time_window}ay/s',
                                                            'Spike Alert on minutes in Past Week',
                                                            'Spike Alert on minutes in Past 30 Days',
                                                            'Spike Alert on minutes in Past 60 Days',
                                                            f'Ramp Alert on minutes in Past {time_window}ay/s',
                                                            'Ramp Alert on minutes in Past Week',
                                                            'Ramp Alert on minutes in Past 30 Days',
                                                            'Ramp Alert on minutes in Past 60 Days',])
    
    torque_names = ["TORQUE_MOTOR", "TORQUE_ROD"]
    speed_names = ['SPEED_MOTOR', 'SPEED_ROD']
                    
    for well_id in well_list:
        #peak_label_df is to be replaced by performing inference for the well here
        print(well_id)
        data_definition_dict = {
            'well_id': well_id,
            'well_raw_data_path': Path(f"/home/ec2-user/SageMaker/efs/data/october_data_extract/tag_{well_id}.csv"),
            'start_date': pd.to_datetime(start_date, format="%Y-%m-%d"),
            'end_date': pd.to_datetime(end_date, format="%Y-%m-%d")
        }
        well_data_df = load_data_from_disk(data_definition_dict)
        if len(well_data_df)==0:
            print(f'EMPTY DATA FOR WELL {well_id}')
            pass
        else:
            # resample data
            well_data_df = resample_to_1min(well_data_df)

            torque_col = which_feature(well_data_df, torque_names)

            speed_col = which_feature(well_data_df, speed_names)

            ### OPERATION STATUS
            operation_methods = [
                'operation_status',
                'flush_status',
            ]

            ###---------------------------------------
            ### WELL OPERATION STATUS DETECTION - ON / OFF IDENTIFICATION
            ###---------------------------------------

            operations_config = {
                'operation_method': 'operation_status',
                'off_threshold': 10
            }


            if operations_config['operation_method'] in operation_methods:
#                 print(f"Generating {operations_config['operation_method']}")
                well_data_df = get_operation_status(well_data_df,
                                                    feature_name=torque_col,
                                                    config_dict=operations_config)


            ###---------------------------------------
            ### WELL FLUSH STATUS DETECTION
            ###---------------------------------------

            flush_config = {
                'operation_method': 'flush_status',
                'flush_diff_threshold': 4.9,
                'flush_std_threshold': 4,
            }

            if flush_config['operation_method'] in operation_methods:
#                 print(f"Generating {flush_config['operation_method']}")
                well_data_df = get_flush_status(well_data_df,
                                                feature_name=speed_col,
                                                config_dict=flush_config)


            ### Run inference for selected techniques on the loaded well data

            # this list will be populated with each technique used for the alert creation
            inference_methods = [
                'torque_spike_decay_alert',
                'ramping_alert',
            ]





            ###---------------------------------------
            ### RAMPING ALERT
            ###---------------------------------------

            model_config = {
                'inference_method': 'ramping_alert',
                'polynomial_degree': 4,
                'polynomial_days': '7d',
                'ramp_integral_threshold': 0.7,
                'operations_config': operations_config,
                'flush_config': flush_config,
            }

            if model_config['inference_method'] in inference_methods:
#                 print(f"Generating {model_config['inference_method']}")
                well_data_df = get_ramping_alerts(well_data_df,
                                                  feature_name=torque_col,
                                                  prediction_name=model_config['inference_method'],
                                                  config_dict=model_config)
#                 print("Complete")



            ###---------------------------------------
            ### TORQUE SPIKE DECAY DETECTION
            ###---------------------------------------

            model_config = {
                'inference_method': 'torque_spike_decay_alert',
                'model_path': Path("/home/ec2-user/SageMaker/efs/models/BRETT_Multi_Well_Model_v2.h5"),
                'window_size': 2 ** 14,
                'window_overlap': 0.5,
                'binary_threshold': 0.5,
                'merged_event_overlap': 10,
                'miminum_event_length': 0,
            }

            well_data_df[f"{torque_col}_roll_avg"] = well_data_df[torque_col].interpolate().rolling(120).mean()

            if model_config['inference_method'] in inference_methods:
                well_data_df = get_torque_spike_decay_alert(well_data_df,
                                                            feature_name=f"{torque_col}",
                                                            prediction_name=model_config['inference_method'],
                                                            model_config=model_config)

            well_data_df['operation_status'] = (well_data_df['operation_status'] == 0) * 1
            well_data_df['flush_status'] = (well_data_df['flush_status'] == 0) * 1
            well_data_df = get_combined_alert(well_data_df, alert_names=['torque_spike_decay_alert', 'flush_status','operation_status'])
            peak_label_df=well_data_df.loc[:,['combined_alert']].resample('1h').sum()
            ramp_label_df=well_data_df.loc[:,['ramping_alert']].resample('1h').sum()
            total_alert_df=pd.DataFrame(index=peak_label_df.index)
            total_alert_df['total_alert']=well_data_df.loc[:,['combined_alert']].resample('1h').sum().values+well_data_df.loc[:,['ramping_alert']].resample('1h').sum().values
            all_well_spike_alert_df[well_id]=peak_label_df.reindex(index=all_well_spike_alert_df.index, method = 'nearest')
            all_well_ramp_alert_df[well_id]=ramp_label_df.reindex(index=all_well_ramp_alert_df.index, method = 'nearest')
            all_well_total_alert_df[well_id]=total_alert_df.reindex(index=all_well_spike_alert_df.index, method = 'nearest')

            if all_well_total_alert_df[well_id].loc[pd.to_datetime(date)-pd.Timedelta(time_window):pd.to_datetime(date)].sum()>0:
                if  peak_label_df.loc[pd.to_datetime(date)-pd.Timedelta(time_window):pd.to_datetime(date)].sum().values[0]>0:
                    todays_results_df.loc[well_id,[f'Spiking Present (past {time_window}ay/s)']]='True'
                else:
                    todays_results_df.loc[well_id,[f'Spiking Present (past {time_window}ay/s)']]='False'
                if  ramp_label_df.loc[pd.to_datetime(date)-pd.Timedelta(time_window):pd.to_datetime(date)].sum().values[0]>0:
                    todays_results_df.loc[well_id,[f'Ramping Present (past {time_window}ay/s)']]='True'
                else:
                    todays_results_df.loc[well_id,[f'Ramping Present (past {time_window}ay/s)']]='False' 
                    
                    
                if  peak_label_df.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]>0:
                    todays_results_df.loc[well_id,['Spiking Present (past month)']]='True'
                else:
                    todays_results_df.loc[well_id,['Spiking Present (past month)']]='False'
                if  ramp_label_df.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]>0:
                    todays_results_df.loc[well_id,['Ramping Present (past month)']]='True'
                else:
                    todays_results_df.loc[well_id,['Ramping Present (past month)']]='False' 
                

                todays_results_df.loc[well_id,[f'Total Alert on minutes in Past {time_window}ay/s']] = total_alert_df.loc[pd.to_datetime(date)-pd.Timedelta(time_window):pd.to_datetime(date)].sum().values[0]#+ramp_label_df.loc[pd.to_datetime(date)-pd.Timedelta(time_window):pd.to_datetime(date)].sum().values[0]
                todays_results_df.loc[well_id,['Total Alert on minutes in Past Week']] = total_alert_df.loc[pd.to_datetime(date)-pd.Timedelta('7d'):pd.to_datetime(date)].sum().values[0]#+ramp_label_df.loc[pd.to_datetime(date)-pd.Timedelta('7d'):pd.to_datetime(date)].sum().values[0]
                todays_results_df.loc[well_id,['Total Alert on minutes in Past 30 Days']] = total_alert_df.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]#+ramp_label_df.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]
                todays_results_df.loc[well_id,['Total Alert on minutes in Past 60 Days']] = total_alert_df.loc[start_date:pd.to_datetime(date)].sum().values[0]#+ramp_label_df.loc[start_date:pd.to_datetime(date)].sum().values[0]

                todays_results_df.loc[well_id,[f'Spike Alert on minutes in Past {time_window}ay/s']] = peak_label_df.loc[pd.to_datetime(date)-pd.Timedelta(time_window):pd.to_datetime(date)].sum().values[0]
                todays_results_df.loc[well_id,['Spike Alert on minutes in Past Week']] = peak_label_df.loc[pd.to_datetime(date)-pd.Timedelta('7d'):pd.to_datetime(date)].sum().values[0]
                todays_results_df.loc[well_id,['Spike Alert on minutes in Past 30 Days']] = peak_label_df.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]
                todays_results_df.loc[well_id,['Spike Alert on minutes in Past 60 Days']] = peak_label_df.loc[start_date:pd.to_datetime(date)].sum().values[0]

                todays_results_df.loc[well_id,[f'Ramp Alert on minutes in Past {time_window}ay/s']] = ramp_label_df.loc[pd.to_datetime(date)-pd.Timedelta(time_window):pd.to_datetime(date)].sum().values[0]
                todays_results_df.loc[well_id,['Ramp Alert on minutes in Past Week']] = ramp_label_df.loc[pd.to_datetime(date)-pd.Timedelta('7d'):pd.to_datetime(date)].sum().values[0]
                todays_results_df.loc[well_id,['Ramp Alert on minutes in Past 30 Days']] = ramp_label_df.loc[pd.to_datetime(date)-pd.Timedelta('30d'):pd.to_datetime(date)].sum().values[0]
                todays_results_df.loc[well_id,['Ramp Alert on minutes in Past 60 Days']] = ramp_label_df.loc[start_date:pd.to_datetime(date)].sum().values[0]

                todays_results_df.loc[well_id,['Average gas flow (60 Days)']] = well_data_df.loc[start_date:date,["FLOW_GAS"]].interpolate("ffill").mean().values[0]
                todays_results_df.loc[well_id,['TD status']] = completion_turndown_df[completion_turndown_df['WellCD']==well_id]['Turndown Cat.'].values[0]
                todays_results_df.loc[well_id,['Design']] = completion_turndown_df[completion_turndown_df['WellCD']==well_id]['Completion Design'].values[0]
                todays_results_df.loc[well_id,['Pump age']] = pd.to_datetime(date)-label_data_df[(label_data_df['WellCD']==well_id) & (label_data_df.index < date)&(label_data_df['Event']=='Pump Change')].index.max()
                todays_results_df.loc[well_id,['Days since last flush']] = pd.to_datetime(date)-label_data_df[(label_data_df['WellCD']==well_id) & (label_data_df.index < date)&((label_data_df['Event']=='Scheduled')|(label_data_df['Event']=='Reactive')|(label_data_df['Event']=='Flushby'))].index.max()
                todays_results_df.loc[well_id,['Days since last failure']] = pd.to_datetime(date)-label_data_df[(label_data_df['WellCD']==well_id) & (label_data_df.index < date) & (label_data_df['Failure Mode'].notnull())].index.max()       

                todays_results_df.loc[well_id,[f'Spike Alert Percent Active Time in Past {time_window}ay/s']]=(todays_results_df.loc[well_id,[f'Spike Alert on minutes in Past {time_window}ay/s']].values[0]/(pd.Timedelta(time_window).total_seconds() / 60))*100
                todays_results_df.loc[well_id,['Spike Alert Percent Active Time in Past Week']]=(todays_results_df.loc[well_id,['Spike Alert on minutes in Past Week']].values[0] / 10080)*100
                todays_results_df.loc[well_id,['Spike Alert Percent Active Time in Past 30 Days']]=(todays_results_df.loc[well_id,['Spike Alert on minutes in Past 30 Days']].values[0]/43200)*100
                todays_results_df.loc[well_id,['Spike Alert Percent Active Time in Past 60 Days']]=(todays_results_df.loc[well_id,['Spike Alert on minutes in Past 60 Days']].values[0]/86400)*100
                                      
                todays_results_df.loc[well_id,[f'Ramp Alert Percent Active Time in Past {time_window}ay/s']]=(todays_results_df.loc[well_id,[f'Ramp Alert on minutes in Past {time_window}ay/s']].values[0]/(pd.Timedelta(time_window).total_seconds() / 60))*100
                todays_results_df.loc[well_id,['Ramp Alert Percent Active Time in Past Week']]=(todays_results_df.loc[well_id,['Ramp Alert on minutes in Past Week']].values[0] / 10080)*100
                todays_results_df.loc[well_id,['Ramp Alert Percent Active Time in Past 30 Days']]=(todays_results_df.loc[well_id,['Ramp Alert on minutes in Past 30 Days']].values[0]/43200)*100
                todays_results_df.loc[well_id,['Ramp Alert Percent Active Time in Past 60 Days']]=(todays_results_df.loc[well_id,['Ramp Alert on minutes in Past 60 Days']].values[0]/86400)*100
                
                
                if plot:
                    _ = get_static_plot_for_single_alert(well_id,
                                                     well_data_df.loc[start_date:date],
                                                     features_to_plot=[torque_col,
                                                                       'SPEED_ROD',
                                                                       'FLOW_GAS'
                                                                      ],
                                                     alert_names=['combined_alert','ramping_alert'],
                                                     label_data_df=label_data_df[label_data_df['WellCD']==well_id].loc[(label_data_df[label_data_df['WellCD']==well_id].index > start_date) & (label_data_df[label_data_df['WellCD']==well_id].index < end_date)],
                                                    )

    todays_results_df=todays_results_df.loc[pd.DataFrame(all_well_total_alert_df.loc[pd.to_datetime(date)-pd.Timedelta(time_window):pd.to_datetime(date)].sum())[all_well_total_alert_df.loc[pd.to_datetime(date)-pd.Timedelta(time_window):pd.to_datetime(date)].sum()>0].index]
    return todays_results_df.sort_values(by=f'Total Alert on minutes in Past {time_window}ay/s',ascending=False)

