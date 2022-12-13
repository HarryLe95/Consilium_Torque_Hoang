from datetime import datetime, timedelta
from typing import Sequence
import logging 

import pandas as pd 
import numpy as np 

from Dataset.DataOperator import DataOperator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DataManager: 
    def __init__(self,
                 wells:Sequence[str], 
                 run_mode:str,
                 backfill_start:str,
                 data_operator:DataOperator,
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
    
    #TODO update this later 
    def update_metadata(self, inference_output:dict, append:bool=True) -> None: 
        """Update metadata based on inference output

        Args:
            inference_output (dict): dictionary of inference output 
            append (bool): whether to append the new metadata information or to overwrite. Defaults to True
        """
        for well in inference_output: 
            status = inference_output[well]['inference_status']
            message = inference_output[well]['message']
            start_time = inference_output[well]['start_time']
            end_time = inference_output[well]['end_time']
            tzinfo = inference_output[well]['tzinfo']
            inf_first_TS = inference_output[well]['inference_first_TS']
            inf_last_TS = inference_output[well]['inference_last_TS']
            if status == 0: 
                next_date = datetime.strptime(inf_first_TS,"%Y-%m-%d %H:%M") + timedelta(days=1)
                next_date = next_date.strftime("%Y-%m-%d %H:%M")
            else:
                next_date = inf_first_TS    
            output= pd.DataFrame({'status':status, 'message': message, 'start_time': start_time, 'end_time': end_time, 'tzinfo':tzinfo, 
                                  'inf_first_TS':inf_first_TS, 'inf_last_TS': inf_last_TS, self.datetime_index_column:next_date}, index = [0])
            if append:
                output = pd.concat([self.metadata[well],output],axis=0)
            self.metadata[well] = output
            self.data_operator.write_metadata(output, well)
            
    def update_event_log(self, inference_output:dict, append:bool) -> None:
        for well in inference_output:
            status = inference_output[well]['inference_status']
            body = inference_output[well]['body']
            if status == 0:
                output = pd.DataFrame(body, index=[0])
                self.data_operator.write_event_log(output, well, append)
    
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

    #TODO 
    def get_training_dataset(self) -> dict[str, np.ndarray]:
        pass 
