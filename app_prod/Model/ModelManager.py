import datetime 
from typing import Sequence
import logging 

import numpy as np
import pandas as pd
from Model.generic import (ABC_Model, ABC_Feature_Extractor)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ABC_ModelManager:
    def get_all_notifications(self, inference_data:pd.DataFrame) -> dict:
        """Get all inference output tags for an inference_data point. Inference data is a one row dataframe restricted to a one-day time window. 

        Args:
            inference_data (pd.DataFrame): inference_data fetched by data manager
        """
        pass 
    
    def get_inference_output(self, inference_data_dict: dict[str, pd.DataFrame]) -> dict:
        pass 

class ModelManager(ABC_ModelManager):
    INFERENCE_STATUS_CODE = {"0": "Inference completed successfully",
                             "1": "RuntimeExceptions encountered",
                             "2": "Insufficient data"}    
        
    def __init__(self, 
                 feature_extractor:ABC_Feature_Extractor, 
                 classification_model:ABC_Model, 
                 regression_model:ABC_Model):
        self.classification_model = classification_model
        self.regression_model = regression_model
        self.feature_extractor = feature_extractor
        
    @classmethod 
    def get_model_version(cls, model_dict:dict[str, ABC_Model]) -> dict[str, str]:
        model_version = {}
        for model in model_dict: 
            model_version[model] = model_dict[model].get_model_version()
        return model_version 
    
   