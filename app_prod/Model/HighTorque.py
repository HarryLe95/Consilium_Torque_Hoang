<<<<<<< HEAD
from Dataset.DataManager import DataManager
from Dataset.DataOperator import TorqueOperator
from Model.ModelManager import ModelManager
from utils.advancedanalytics_util import aauconnect_
import datetime

class HighTorque:
    def __init__(self,
                 group_config:dict,
                 inference_config:dict, 
                 data_connection_config: dict,
                 torque_config:dict,
                 model_config:dict,):
        """ROC Model trainer. Manage training and inference pipeline.

        Args:
            group_config (dict): dictionary of group keywords
            inference_config (dict): dictionary of inference keywords
            data_connection_config (dict): dictionary of data connection keywords
            torque_config (dict): dictionary of roc config keywords
            model_config (dict): dictionary of model config keywords 
        """
        
        self._validate_group_config(group_config)
        self.group_config = group_config 
        self._gp_info = self._get_group_info()
        self.inference_wells = [d['WELL_CD'] for d in self._gp_info]
        
        self._validate_inference_config(inference_config)
        self.inference_config = inference_config
        self.run_mode = self.inference_config["run_mode"]
        if self.run_mode == "backfill":
            self.backfill_start = self.inference_config["backfill_start"]
            if "backfill_end" not in self.inference_config:
                self.inference_config['backfill_end'] = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
            self.backfill_end = self.inference_config["backfill_end"]
        
        self._validate_data_connection_config(data_connection_config)
        self.data_connection_config = data_connection_config
        
        self._validate_torque_config(torque_config)
        self.torque_config = torque_config 
        
        self._validate_model_config(model_config)
        self.model_config = model_config 
        
        self.data_operator = TorqueOperator(**self.data_connection_config, **self.torque_config)
        self.data_manager = DataManager(wells= self.inference_wells,  data_operator=self.data_operator, **self.inference_config)
        # self.model_manager = ModelManager(**self.model_config)
        
    def _validate_inference_config(self, config:dict)->None:
        assert("run_mode" in config), f"inference_info must contain keyword 'run_mode'"
        assert(config["run_mode"] in ["live", "backfill"]), f"inference_info's 'run_mode' must be one of ['live', 'backfill']"
        if config["run_mode"] == "back_fill":
            assert("backfill_start" in config), f"Backfill start date must be provided when running inference in backfill mode. inference_info dictionary must contain\
                'backfill_start' keyword."
            assert("backfill_date_format" in config), f"Keyword 'backfill_date_format' must be provided in inference_info dictionary"
        assert("inference_window" in config), f"inference_info must contain keyword 'inference_window'"
        assert("datetime_index_column" in config), f"inference_info must contain keyword 'datetime_index_column'"

    def _validate_general_connection_config(self, config:dict, parent_config:str)->None:
        assert("connection_type" in config), f"{parent_config} must contain keyword 'connection_type'"
        if config["connection_type"] == "s3":
            assert("region" in config), f"{parent_config} must contain keyword 'region' if 'connection_type' is 's3'."
            assert("user" in config), f"{parent_config} must contain keyword 'user' if 'connection_type' is 's3'."
            assert("bucket" in config), f"{parent_config} must contain keyword 'bucket' if 'connection_type' is 's3'." 
        assert("path" in config), f"{parent_config}_info must contain keyword 'path'"
        assert("partition_mode" in config), f"{parent_config} must contain keyword 'partition_mode'"
        if not("tzname" in config):
            config["tzname"]=None
    
    def _validate_data_connection_config(self, config:dict)->None:
        self._validate_general_connection_config(config, "data_connection_info")
        assert("file_prefix" in config), f"data_connection_info must contain keyword 'file_prefix'"
        assert("file_suffix" in config), f"data_connection_info must contain keyword 'file_suffix'"
             
    def _validate_torque_config(self,config:dict)->None:
        assert("features" in config), f"roc_info must contain keyword 'features'"
        assert("fill_method" in config), f"roc_info must contain keyword 'fill_method'"
        assert("datetime_index_column" in config), f"roc_info must contain keyword 'datetime_index_column'"

    def _validate_group_config(self,config:dict)->None:
        assert("group_connection_info" in config), f"group_info must contain keyword 'group_connection'"
        self._validate_general_connection_config(config["group_connection_info"],"group_info['group_connection_info']")
        assert("file" in config["group_connection_info"]), f"group_info['group_connection_info'] must contain keyword 'file'"
        assert("group_sql" in config), f"group_info must contain keyword 'group_sql'"
        assert("group_kwargs" in config), f"group_info must contain keyword 'group_kwargs'"
        assert("group_id" in config), f"group_info must contain keyword 'group_id'"
    
    def _validate_model_config(self, config:dict)->None:
        pass
    
    def _get_group_info(self) -> dict:
        sql = self.group_config["group_sql"]
        args = {'GROUP_ID': self.group_config["group_id"]}
        kwargs = self.group_config["group_kwargs"]
        connection = aauconnect_(self.group_config["group_connection_info"])
        return connection.read(sql=sql, args=args, edit=[], orient='records', do_raise=True, **kwargs)

    def run_model_training(self):
        pass
    
    def _get_model_training_data(self):
        pass

    def get_inference_dataset(self)->tuple[dict,dict]:
        return self.data_manager.get_inference_dataset()
    
    def run_inference_(self, append:bool=False):
        inference_dict = self.model_manager.run_inference(self.get_inference_dataset())
        self.data_manager.update_metadata(inference_dict, append)
        self.data_manager.update_event_log(inference_dict, append)       
    
    def run_model_inference(self):
        """Run inference pipeline:
        
        If run mode is live:
            Fetch inference data 
            Get inference result 
            Update metadata information 
            Update event log 
        If run mode is backfill: 
            Repeat the process from backfill start_date t0 backfill end_date
        """
        if self.run_mode == "live":
            self.run_inference_(True)
        elif self.run_mode == "backfill":
            start_time = datetime.datetime.strptime(self.backfill_start, "%Y-%m-%d %H:%M")
            end_time = datetime.datetime.strptime(self.backfill_end, "%Y-%m-%d %H:%M")
            append = False
            while start_time != end_time:
                self.run_inference_(append)
                append = True
                start_time += datetime.timedelta(days=1)

if __name__ == "__main__":
    import config.__config__ as base_config
    import pandas as pd 
    config = base_config.init()
    model = HighTorque(config['group_info'],
                config['inference_info'],
                config['data_connection_info'],
                config['torque_info'],
                config['model_info'])
    data = model.get_inference_dataset()
    completion_turndown_df = model.data_operator.read_completion_data()
    label_df = model.data_operator.read_label_data()
    from Model.TorqueModel import Model
    inf_model = Model("C:/Users/HoangLe/Desktop/Consilium_TORQUE_HOANG/app_prod/Saved_Model/BRETT_Multi_Well_Model_v2.h5")
    df = inf_model.alert_monitor_generation_all_wells("RM07-22-2", data[0]["RM07-22-2"],completion_turndown_df=completion_turndown_df, label_data_df=label_df)
    sample_df = pd.read_csv("C:/Users/HoangLe/Desktop/Consilium_TORQUE_HOANG/app_prod/TORQUE/TAG_DATA/Output_df.csv")
    
    print("END")
=======
import pandas as pd 
import numpy as np 

class High_Torque_Model:
    def __init__(self):
        pass 
    
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
        #polynomial_days, flush_config - operation_method, operations_config - operation_method, polynomial_degree, ramp_integral threshold    
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
>>>>>>> 5a97d8fc5a28ec8663078b5c253462e9e0bdc3e0
