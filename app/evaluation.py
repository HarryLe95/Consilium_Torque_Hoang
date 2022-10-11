import pandas as pd 
import numpy as np 
from pathlib import Path
from datetime import timedelta 
from src.utils.PathManager import Paths as Path

EFS_PATH = Path('/home/ec2-user/SageMaker/efs/')
data_from_folder='brett_extract_from_s3'
output_save_folder='att2_new_extracts'

def get_data(well_id, label_df=None, buffer_days=30):
    tag_df=pd.read_pickle(f'/home/ec2-user/SageMaker/efs/data/{data_from_folder}/limited_tag_{well_id}.pkl')
    tag_df=tag_df.asfreq('T', method='ffill')#.resample('1min').mean()#
    tag_df=tag_df.interpolate(method='linear', axis=0)

    ramp_df = pd.read_pickle(f'/home/ec2-user/SageMaker/efs/labels/{output_save_folder}/ramping_alert_df{well_id}.pkl')
    ramp_df = ramp_df[ramp_df.ramp_label==1]
    spike_df = pd.read_pickle(f'/home/ec2-user/SageMaker/efs/labels/{output_save_folder}/1hrtorque_spike_label_{well_id}.pkl')
    spike_df = spike_df[spike_df.spike_label==1]
    label = label_df[label_df.WellCD==well_id].sort_index()

    return tag_df,ramp_df, spike_df, label

def get_cm_statistic(pred_df, tag_df, label, feature='TORQUE_ROD', off_threshold=10, pseudo_threshold=0.1):
    #Get positive ground truths
    min_date = tag_df.index.min().date
    max_date = label.index.max().date
    import pdb;pdb.set_trace()
    all_negatives = np.unique(pd.date_range(min_date, max_date, freq='D').date)
    positive_preds = np.unique(pred_df.index.date)
    negative_preds = np.setdiff1d(all_negatives, positive_preds)
    positive_gt_ = label[label.Event!="Scheduled"].index.unique()
    positive_gt = np.unique([pd.date_range(x-timedelta(days=30),x).date for x in positive_gt_])

    #For a buffer period of 30 days after an event occurs, 
    #if the torque is 0 for 10% of the time and if any label is issued in that 30 day period -> the label is not a false positive 
    #Exception to the rule: if both the torque and speed are non-zero for 7 days after the last 0 and label is issued -> label is a false positive
    for positive_date in positive_gt_:
        buffer_period = pd.date_range(positive_date,positive_date+timedelta(days=30)).date
        data = tag_df.loc[buffer_period.min():buffer_period.max(), feature].values
        thresholded_data = data[data<=off_threshold]
        last_day = thresholded_data.index.max()
        ratio = len(thresholded_data)/len(data)
        import pdb;pdb.set_trace()

        if ratio >= pseudo_threshold:
            positive_gt = np.union1d(positive_gt, pd.date_range(positive_date,last_day).date)

    negative_gt = np.setdiff1d(all_negatives, positive_gt_)
    check_positive = np.vectorize(lambda x: x in positive_gt)
    check_negative = np.vectorize(lambda x: x in negative_gt)
    #TP is number of positives being picked up 
    TP = positive_preds[check_positive(positive_preds)]
    FP = np.setdiff1d(positive_preds,TP)
    TN = negative_preds[check_negative(negative_preds)]
    FN = np.setdiff1d(negative_preds,TN)

    TPn = len(TP)
    FPn = len(FP)
    TNn = len(TN)
    FNn = len(FN)
    TPR = TPn/(TPn+FNn) 
    FPR = FPn/(FPn+TNn)
    TNR = TNn/(TNn+FPn)
    FNR = FNn/(TPn+FNn)
    return TPn, FPn, TNn, FNn, TPR, FPR, TNR, FNR, TP, FP, TN, FN

def evaluate_model(well_id, label_df,buffer_days=30,feature='TORQUE_ROD',off_threshold=10,pseudo_threshold=0.1):
    tag_df,ramp_df, spike_df, label= get_data(well_id, label_df=label_df, buffer_days=buffer_days)
    if len(tag_df)==0:
        print(f"Empty tag df for well: {well_id}")
        return None
    if len(label)==0:
        print(f"Empty label for well: {well_id}")
        return None
    metrics = {mode:{'TP':0,'FP':0,'TN':0,'FN':0,'TPR':0,'FPR':0,'TNR':0,'FNR':0} for mode in ['ramp','spike','both']}
    #Evaluate Ramp only
    for mode in ['ramp','spike','both']:
        if mode == 'ramp':
            pred_df = ramp_df
        if mode == 'spike':
            pred_df = spike_df
        else:
            pred_df = pd.concat([ramp_df,spike_df],axis=0)
        TP,FP,TN,FN,TPR,FPR,TNR,FNR,_,_,_,_= get_cm_statistic(pred_df,tag_df,label,feature=feature,off_threshold=off_threshold,pseudo_threshold=pseudo_threshold)
        metrics[mode]['TP'] = TP
        metrics[mode]['TPR'] = TPR
        metrics[mode]['TN'] = TN
        metrics[mode]['TNR'] = TNR
        metrics[mode]['FP'] = FP
        metrics[mode]['FPR'] = FPR
        metrics[mode]['FN'] = FN
        metrics[mode]['FNR'] = FNR
    return metrics