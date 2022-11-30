"""
Defining the interfaces for different modelling aspects

Classification model - provides labels for different operating modes
Regression model - provides the number of days until failure 
Feature extractor model - extract features to be fed into the classification models and regression models

"""

import numpy as np 

class ABC_Model:
    def fit(self, training_data: np.ndarray): 
        pass 
    
    def predict(self, inference_data: np.ndarray):
        pass 
        
    def get_model_version(self): 
        pass 
    
class ABC_Feature_Extractor:
    @classmethod 
    def get_well_status(cls, inference_data:np.ndarray):
        pass
    
    @classmethod 
    def get_downtime(cls, inference_data:np.ndarray):
        pass
    
    @classmethod
    def get_max_volt(cls, inference_data:np.ndarray):
        pass
    
    @classmethod 
    def get_min_volt(cls, inference_data:np.ndarray):
        pass
    
    @classmethod 
    def get_charge_volt(cls,inference_data:np.ndarray):
        pass


