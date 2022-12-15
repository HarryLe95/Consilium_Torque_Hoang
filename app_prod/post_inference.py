import config.__config__ as base_config
from Dataset.DataOperator import DataOperator
from Dataset.DataManager import DataManager
from utils.advancedanalytics_util import aauconnect_
from utils.logging import get_logger
logger = get_logger(__name__)

def main():
    config = base_config.init()
    logger.debug("Aggregating event logs")
    #Get all wells
    connection = aauconnect_(config["group_info"]["group_connection_info"])
    group_table = connection.read(args=[], edit=[], orient='df', do_raise=True)
    all_wells = group_table["WELL_CD"].values
    
    #DataManager 
    data_operator = DataOperator(**config["data_connection_info"], **config["torque_info"])
    data_manager = DataManager(all_wells, data_operator = data_operator, **config["inference_info"])
    
    #Notification DataFrame
    notification_df = data_manager.combine_event_log()
        
    notification_file_name = "TRQ_INFERENCE_LAST.csv"
    kwargs = {'file':notification_file_name, 'path':config['data_connection_info']['path'], 'partition_mode': None, "append":False}
    logger.debug("Writting aggregated event log to database")
    logger.debug(f"Aggregated event log info - connection: {config['data_connection_info']['connection_type']}; path: {kwargs['path']}; name: {kwargs['file']}")
    connection.write_many(args=notification_df, edit=[], orient="df",**kwargs)
    logger.debug("Aggregating event logs completed")
    
if __name__ == "__main__":
    main()