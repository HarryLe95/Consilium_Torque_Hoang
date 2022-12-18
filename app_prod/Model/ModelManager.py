import datetime 
import logging 

import pandas as pd
from Model.TorqueModel import TorqueModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ModelManager:
    INFERENCE_STATUS_CODE = {"0": "Inference completed successfully",
                             "1": "RuntimeExceptions encountered",
                             "2": "Insufficient data"}   
    
    LOCAL_TIMEZONE = str(datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo)
    
    FLOW_STATUS = {0: "Shut-In", 1: "Online"} 
    def __init__(self, **kwargs):
        self.labeler = TorqueModel(**kwargs)
        
    def run_training(self,training_data)->None:
        pass
    
    def run_inference(self, 
                      inference_data:tuple[dict[str,pd.DataFrame], dict[str,tuple]],
                      completion_turndown_df:pd.DataFrame,
                      label_data_df: pd.DataFrame)->dict:
        logger.debug("Running inference for all wells.")
        all_responses = {}
        inference_df = inference_data[0]
        inference_metadata = inference_data[1]
        for well in inference_df:
            well_inference_df = inference_df[well]
            well_inference_metadata = inference_metadata[well]
            all_responses[well] = self.run_inference_(well,well_inference_df, well_inference_metadata, completion_turndown_df,label_data_df)
        return all_responses
    
    def run_inference_(self, 
                       well_cd:str,
                       inference_df:pd.DataFrame, 
                       inference_metadata:pd.DataFrame,
                       completion_turndown_df:pd.DataFrame,
                       label_data_df:pd.DataFrame,
                       )->dict:
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        status = 0
        exception_message = "Inference run successfully"
        response = None
        try:
            response = self.labeler.run_well_inference(well_cd, inference_df, completion_turndown_df, label_data_df)
        except Exception as e:
            exception_message = e
            logger.error(f"Errors encountered when running inference. Exception encountered: {exception_message}")            
            status = 1
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return {"STATUS":status, "BODY": response, "START_TS": start_time, "END_TS": end_time, 
                "FIRST_TS": inference_metadata[0].strftime("%Y-%m-%d %H:%M"),
                "LAST_TS": inference_metadata[1].strftime("%Y-%m-%d %H:%M")} 
        
