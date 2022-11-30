from datetime import datetime, timedelta
from typing import Sequence
import logging 

import pandas as pd 
import numpy as np 

from Dataset.generic import ABC_DataManager
from Dataset.Dataset import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DataManager(ABC_DataManager):    
    def __init__(self,
                 wells:Sequence[str], 
                 run_mode:str,
                 backfill_start:str,
                 dataset:Dataset,
                 backfill_date_format:str="%Y-%m-%d %H:%M",
                 perform_model_training:bool=True,
                 perform_model_inference:bool=False,
                 datetime_index_column:str="TS",
                 **kwargs,
                 ):
        self.wells = wells  
        self.perform_model_training = perform_model_training 
        self.perform_model_inference = perform_model_inference
        self.run_mode = run_mode
        self.dataset = dataset 
        self.backfill_start = backfill_start 
        self.backfill_date_format = backfill_date_format
        self.datetime_index_column = datetime_index_column
        self.metadata = self.get_metadata()
             
    def get_metadata(self) -> dict:
        if self.run_mode == "live":
            return self.dataset.get_metadata(self.wells)
        elif self.run_mode == "backfill": 
            try: 
                metadata_dict = {well: pd.DataFrame.from_dict({self.datetime_index_column:[self.backfill_start]}) for well in self.wells}
                return metadata_dict
            except Exception as e: 
                raise e 
    
    def update_metadata(self, inference_output:dict) -> None: 
        for well in inference_output: 
            status = inference_output[well]['status']
            if status == 0: 
                current_date = datetime.strptime(self.metadata[well][self.datetime_index_column][0],self.backfill_date_format)
                next_date = current_date + timedelta(days=1)
                self.metadata[well][self.datetime_index_column] = next_date.strftime(self.backfill_date_format)
                
    def write_metadata(self):
        pass 
        
    def get_inference_dataset(self) -> dict[str, np.ndarray]:
        inference_dataset = {} 
        for well in self.wells:
            start = datetime.strptime(self.metadata[well][self.datetime_index_column].values[0], self.backfill_date_format).date()
            end = start + timedelta(days=1)
            inference_dataset.update(self.dataset.get_inference_dataset([well],str(start),str(end)))
        return inference_dataset
    
    def get_training_dataset(self) -> dict[str, np.ndarray]:
        pass 

