from datetime import datetime, timedelta
from typing import Sequence
import logging 

import pandas as pd 
import numpy as np 

from Dataset.DataOperator import TorqueOperator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def concat_no_exception(dfs:Sequence[pd.DataFrame|None],axis=0)->pd.DataFrame:
    all_df = [df for df in dfs if df is not None]
    return pd.concat(all_df,axis=axis)

class DataManager: 
    def __init__(self,
                 wells:Sequence[str], 
                 run_mode:str,
                 backfill_start:str,
                 data_operator:TorqueOperator,
                 inference_window:int=60,
                 datetime_index_column:str="TS",
                 **kwargs,
                 ):
        """Metaclass that manages dataset functionalities. Primary methods include: 
    
        Methods:
            get_metadata: get metadata information for all inference wells 
            update_metadata: update metadata based on inference response 
            get_training_dataset: get training dataset 
            get_inference_dataset: get a dictionary of {well_name: well_inf_df}

        Args:
            wells (Sequence[str]): inference wells
            run_mode (str): inference run mode. One of ["live", "backfill"]
            backfill_start (str): backfill start date if run_mode is "backfill"
            data_operator (Dataset): dataset operator that handles read/write
            inference_window (int, optional): number of previous days of supporting data for inference. Defaults to 30.
            datetime_index_column (str, optional): column name to be interpreted as datetime index. Defaults to "TS".
        """
        self.wells = wells  
        self.run_mode = run_mode
        self.data_operator = data_operator
        self.backfill_start = backfill_start 
        self.datetime_index_column = datetime_index_column
        self.inference_window = inference_window
        self.metadata = self.get_metadata()
             
    def get_metadata(self) -> dict[str,pd.DataFrame]:
        """Get medata data based on run mode.
        
        live mode: get latest metadata 
        backfill mode: create an artificial metadata with chosen backfill_start date 
        
        Returns:
            dict[str,pd.DataFrame] - dictionary of {well_code:metadata_dataframe}
        """
        if self.run_mode == "live":
            logger.debug("Getting metadata in live mode.")
            return {well:self.data_operator.read_metadata(well) for well in self.wells}
        elif self.run_mode == "backfill": 
            try:
                logger.debug("Getting metadata in backfill mode.") 
                metadata_dict = {well: pd.DataFrame.from_dict({self.datetime_index_column:[self.backfill_start]}) for well in self.wells}
                return metadata_dict
            except Exception as e: 
                logger.error(f"Error getting backfill mode metadata. Error Message: {e}")
                raise e 
    
    def update_metadata(self, inference_output:dict, append:bool=True) -> None: 
        """Update metadata based on inference output

        Args:
            inference_output (dict): dictionary of inference output 
            append (bool): whether to append the new metadata information or to overwrite. Defaults to True
        """
        for well in inference_output: 
            status = inference_output[well]['STATUS']
            start_time = inference_output[well]['START_TS']
            end_time = inference_output[well]['END_TS']
            inf_first_TS = inference_output[well]['FIRST_TS']
            inf_last_TS = inference_output[well]['LAST_TS']
            if status == 0: 
                next_date = datetime.strptime(inf_first_TS,"%Y-%m-%d %H:%M") + timedelta(days=1)
                next_date = next_date.strftime("%Y-%m-%d %H:%M")
            else:
                next_date = inf_first_TS    
            output= pd.DataFrame({'STATUS':status, 'START_TS': start_time, 'END_TS': end_time,  
                                  'FIRST_TS':inf_first_TS, 'LAST_TS': inf_last_TS, self.datetime_index_column:next_date}, index = [0])
            if append:
                output = concat_no_exception([self.metadata[well],output],axis=0)
            self.metadata[well] = output
            self.data_operator.write_metadata(output, well)
    
    def get_event_log(self)->dict[str,pd.DataFrame]:
        try:
            logger.debug("Getting event log")
            return {well:self.data_operator.read_event_log(well) for well in self.wells}
        except Exception as e: 
            logger.error(f"Error getting event log from database. Error message: {e}")
            raise e 
    
    def update_event_log(self, inference_output:dict, append:bool) -> None:
        for well in inference_output:
            status = inference_output[well]['STATUS']
            body = inference_output[well]['BODY']
            if status == 0:
                output = pd.DataFrame(body, index=[0])
                if append: #Append to event log logic
                    historical_log = self.data_operator.read_event_log(well)
                    output = concat_no_exception([historical_log, output], axis = 0)
                self.data_operator.write_event_log(output, well)
    
    def combine_event_log(self)->pd.DataFrame:
        event_log_dict ={well:self.data_operator.read_event_log(well).iloc[[-1],:] for well in self.wells} 
        metadata_dict = {well:self.data_operator.read_metadata(well).iloc[-1,:] for well in self.wells}
        all_data = []
        for well in self.wells:
            if metadata_dict[well]["status"] == 0:
                all_data.append(event_log_dict[well])
        notification_df = concat_no_exception(all_data,axis=0)
        return self.post_process_event_log(notification_df)
    
    def post_process_event_log(self, event_log:pd.DataFrame)->pd.DataFrame:
        event_log = event_log.sort_values(by=["SPIKE_PERCENT_7DAY"], ascending = [False])
        event_log["PRIORITY"] = len(event_log) - np.arange(len(event_log))
        return event_log
    
    def get_inference_day_dataset_(self, well:str) -> tuple[datetime.date,pd.DataFrame]|None:
        """Get data for the inference day. Inference day is the last day (day in the last row) of the corresponding metadata

        Args:
            well (str): well code 

        Returns:
            tuple[datetime.date,pd.DataFrame]|None: if data is insufficient, return None. Otherwise return the data of the inference day.
        """
        inf_start = datetime.strptime(self.metadata[well][self.datetime_index_column].values[-1], "%Y-%m-%d %H:%M").date()
        inf_end = inf_start + timedelta(days=1)
        inf_data = self.data_operator.read_data(well, inf_start, inf_end)
        inf_end_actual = inf_data.index.max()
        return (inf_start,inf_end_actual,inf_data)
            
    def get_inference_dataset(self) -> dict[str, np.ndarray]:
        """Get inference dataset 
        
        Returns:
            dict[str, np.ndarray]: dictionary of inference data 
        """
        inference_dataset = {}
        metadata_dataset = {} 
        for well in self.wells:
            inf_start, inf_end, inf_day_df = self.get_inference_day_dataset_(well)
            metadata_dataset[well] = (inf_start, inf_end)
            if inf_day_df is not None:
                window_start = inf_start - timedelta(days=self.inference_window-1)
                window_end = inf_start - timedelta(days=1)
                supp_day_df = self.data_operator.read_data(well, window_start, window_end)
                inf_df = pd.concat([supp_day_df, inf_day_df],axis=0)
                inference_dataset[well] = self.data_operator.process_data(inf_df)
            else:
                inference_dataset[well] = None
        return inference_dataset, metadata_dataset

    def get_label_df(self)->pd.DataFrame:
        """TORQUE Project specific: manual annotation data

        Returns:
            pd.DataFrame: label dataframe
        """
        return self.data_operator.read_label_data()
    
    def get_completion_turndown_df(self)->pd.DataFrame:
        """TORQUE Project specific: completion turndown df

        Returns:
            pd.DataFrame: completion dataframe
        """
        return self.data_operator.read_completion_data()

    #TODO 
    def get_training_dataset(self) -> dict[str, np.ndarray]:
        pass 
