from Dataset.DataManager import DataManager
from Dataset.DataOperator import TorqueOperator
from Model.ModelManager import ModelManager
from utils.advancedanalytics_util import aauconnect_
import datetime
from functools import cached_property

class Trainer:
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
        #Validate group keywords
        self._validate_group_config(group_config)
        self.group_config = group_config 
        self.group_info = self.get_group_info()
        self.inference_wells = [d['WELL_CD'] for d in self.group_info]
        
        #Validate inference keywords
        self._validate_inference_config(inference_config)
        self.inference_config = inference_config
        self.run_mode = self.inference_config["run_mode"]
        if self.run_mode == "backfill":
            self.backfill_start = self.inference_config["backfill_start"]
            if "backfill_end" not in self.inference_config:
                self.inference_config['backfill_end'] = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
            self.backfill_end = self.inference_config["backfill_end"]
        
        #Validate data connection keywords
        self._validate_data_connection_config(data_connection_config)
        self.data_connection_config = data_connection_config
        
        #Validate torque keywords
        self._validate_torque_config(torque_config)
        self.torque_config = torque_config 
        
        #Validate model config
        self._validate_model_config(model_config)
        self.model_config = model_config 
        
        #Setup model and data manager
        self.setup()
        
        
    def setup(self)->None:
        data_operator = TorqueOperator(**self.data_connection_config, **self.torque_config)
        self.data_manager = DataManager(wells= self.inference_wells,  data_operator=data_operator, **self.inference_config)
        self.model_manager = ModelManager(**self.model_config)
        
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
        assert("model_path" in config), f"model_info must contain keyword 'model_path'"
        assert("use_operation_status" in config), f"model_info must contain keyword 'use_operation_status'"
        assert("use_flush_status" in config), f"model_info must contain keyword 'use_flush_status'"
        assert("use_ramping_alert" in config), f"model_info must contain keyword 'use_ramping_alert'"
        assert("use_torque_spike_alert" in config), f"model_info must contain keyword 'use_torque_spike_alert'"
        assert("time_window" in config), f"model_info must contain keyword 'time_window'"
        assert("off_threshold" in config), f"model_info must contain keyword 'off_threshold'"
        assert("flush_diff_threshold" in config), f"model_info must contain keyword 'flush_diff_threshold'"
        assert("flush_std_threshold" in config), f"model_info must contain keyword 'flush_std_threshold'"
        assert("polynomial_days" in config), f"model_info must contain keyword 'polynomial_days'"
        assert("polynomial_degree" in config), f"model_info must contain keyword 'polynomial_degree'"
        assert("ramp_integral_threshold" in config), f"model_info must contain keyword 'ramp_integral_threshold'"
        assert("window_size" in config), f"model_info must contain keyword 'window_size'"
        assert("window_overlap" in config), f"model_info must contain keyword 'window_overlap'"
        assert("binary_threshold" in config), f"model_info must contain keyword 'binary_threshold'"
        assert("merged_event_overlap" in config), f"model_info must contain keyword 'merged_event_overlap'"
        assert("minimum_event_length" in config), f"model_info must contain keyword 'minimum_event_length'"
            
    def get_group_info(self) -> dict:
        sql = self.group_config["group_sql"]
        args = {'GROUP_ID': self.group_config["group_id"]}
        kwargs = self.group_config["group_kwargs"]
        connection = aauconnect_(self.group_config["group_connection_info"])
        return connection.read(sql=sql, args=args, edit=[], orient='records', do_raise=True, **kwargs)

    def run_model_training(self):
        pass
                
    def run_inference_(self, append:bool=False):      
        inference_dataset = self.data_manager.get_inference_dataset()
        completion_turndown_df = self.data_manager.get_completion_turndown_df()
        label_df = self.data_manager.get_label_df()
        inference_dict = self.model_manager.run_inference(inference_dataset,completion_turndown_df, label_df)
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

