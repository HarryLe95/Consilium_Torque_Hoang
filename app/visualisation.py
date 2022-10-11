import matplotlib.pyplot as plt 
import pandas as pd 
from datetime import timedelta 

colour_dict = {
               'Flushby':'red', 
               'Reactive':'darkred',
               'Pump Change':'wheat', 
               'Scheduled':'yellowgreen',
               'Rig Intervention':'mediumvioletred',
               'Ramp Label 0': 'gold',
               'Ramp Label 1': 'magenta',
               'Spike Label 0': 'aqua',
               'Spike Label 1': 'darkorchid'}

feature_set={"MOTOR":['TORQUE_MOTOR','SPEED_MOTOR'],
             'ROD':['TORQUE_ROD','SPEED_ROD']}

def plot_label(ax: plt.axes, 
               label:pd.DataFrame, 
               start:str=None, 
               end:str=None):
    #Slice dataframe based on start and end 
    if start is None and end is None:
        label_df = label
    elif start is None and end is not None:
        label_df = label.loc[:end]
    elif start is not None and end is None:
        label_df = label.loc[start:]
    else:
        label_df = label.loc[start:end]
    #Overlaying label columns
    for i in range(len(label_df)):
        x_start = label_df.index[i]
        x_end = x_start + timedelta(days=1)
        event = label_df.iloc[i,:]['Event']
        colour = colour_dict[event]
        ax.axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
        
def plot_spike(ax: plt.axes, 
               label:pd.DataFrame, 
               start:str=None, 
               end:str=None):
    #Slice dataframe based on start and end 
    if start is None and end is None:
        label_df = label
    elif start is None and end is not None:
        label_df = label.loc[:end]
    elif start is not None and end is None:
        label_df = label.loc[start:]
    else:
        label_df = label.loc[start:end]
    label_df = label_df[label_df.spike_label==1]
    #Overlaying label columns
    for i in range(len(label_df)):
        x_start = label_df.index[i]
        x_end = x_start + timedelta(days=1)
        event = label_df.loc[x_start].values[0]
        colour = colour_dict['Spike Label '+str(event)]
        ax.axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.05)
        
def plot_ramp(ax: plt.axes, 
               label:pd.DataFrame, 
               start:str=None, 
               end:str=None):
    #Slice dataframe based on start and end 
    if start is None and end is None:
        label_df = label
    elif start is None and end is not None:
        label_df = label.loc[:end]
    elif start is not None and end is None:
        label_df = label.loc[start:]
    else:
        label_df = label.loc[start:end]
    label_df = label_df[label_df.ramp_label==1]
    #Overlaying label columns
    for i in range(len(label_df)):
        x_start = label_df.index[i]
        x_end = x_start + timedelta(days=1)
        event = label_df.loc[x_start].values[0]
        colour = colour_dict['Ramp Label '+str(event)]
        ax.axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.05)


def plot_data(ax: plt.Axes, 
              df:pd.Series, 
              start:str=None, 
              end:str=None, 
              color:str='g',
              size:int=20,
              marker:str='.'):
    #Slice dataframe based on start and end 
    if start is None and end is None:
        plot_df = df
    elif start is None and end is not None:
        plot_df = df.loc[:end]
    elif start is not None and end is None:
        plot_df = df.loc[start:]
    else:
        plot_df = df.loc[start:end]
        
    #Plot overlaying features 
    ax.scatter(plot_df.index, plot_df.values, c=color, s=size, marker=marker, label=plot_df.name)
    ax.grid()
    
def plot_TORQUE(raw_df, 
                well_name, 
                label_df=None,
                ramp_label=None,
                spike_label=None,
                start=None, 
                end=None, 
                well_type:str="ROD",
                ylim:dict = {"TORQUE_ROD":[100, 200]},
                save_name='Image'):     

    if label_df is not None:
        label = label_df[label_df.WellCD==well_name].sort_index()
   
    raw_features = feature_set[well_type]
    fig, ax = plt.subplots(len(raw_features), figsize=(30,15), sharex=True)

    for idx, feature in enumerate(raw_features):
        ax[idx].yaxis.set_tick_params(labelsize=20)
        plot_data(ax[idx], raw_df[feature], start, end)
        if feature in ylim:
            ax[idx].set_ylim(ylim[feature])
        ax[idx].set_ylabel(feature, size=20)
        ax[idx].legend(loc='upper left', prop={'size': 30})
        if label_df is not None:
            plot_label(ax[idx],label,start,end)
        if ramp_label is not None:
            plot_ramp(ax[idx], ramp_label, start, end)
        if spike_label is not None: 
            plot_spike(ax[idx], spike_label, start, end)
            
        ax[idx].legend(loc='upper left', prop={'size': 30})
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='',ms=30) for color in colour_dict.values()]
    plt.legend(markers, colour_dict.keys(), numpoints=1,prop={'size': 15})
    fig.suptitle(f'{save_name[:-4]}',fontsize=96)
    plt.xticks(fontsize=15)
    plt.xlabel("TS")
    plt.savefig(f"Viz/{save_name}.png")
    return fig 