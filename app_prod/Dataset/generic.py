import numpy as np 
import pandas as pd 

class ABC_Dataset:
    def extract_keyword(self, kwargs:dict) -> None:
        pass
    
    def get_dataset(self,start:str,end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> np.ndarray|pd.DataFrame|dict:
        pass 
        
class ABC_DataManager:
    def get_training_dataset(self) -> dict[str, np.ndarray]:
        pass 
    
    def get_inference_dataset(self) -> dict[str, np.ndarray]:
        pass 
    
    def get_metadata(self) -> dict: 
        pass 
    
    def update_metadata(self, response:dict) -> None: 
        pass 

 