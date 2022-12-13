import unittest 

from utils.advancedanalytics_util import aauconnect_
from utils.PathManager import PathManager 
import config.__config__ as base_config
from Dataset.Dataset import Dataset 
from Dataset.DataManager import DataManager

PathManager.local_test_data = PathManager.UTILS_PATH.parents[1] / "roc/PROCESSED_DATA"

class TestDataManager(unittest.TestCase):
    def _setup_file_config(self) -> Dataset:
        config = {"connection_type":"file", "path":PathManager.local_test_data, "file_prefix":"ROC_PROCESSED_DATA", "file_suffix":".csv"}
        connection = aauconnect_(config)
        self.datetime_index_column="TS"
        
        return Dataset(connection, path=config['path'],
                      file_prefix=config['file_prefix'],
                      file_suffix=config['file_suffix'],
                      datetime_index_column=self.datetime_index_column)
    
    def _setup_S3_file_config(self) -> Dataset:
        config = base_config.init()
        connection = aauconnect_(config['cfg_s3_info'])
        self.datetime_index_column="TS"
        
        return Dataset(connection, path=config['cfg_s3_info']['path'],
                      file_prefix=config['cfg_s3_info']['file_prefix'],
                      file_suffix=config['cfg_s3_info']['file_suffix'],
                      datetime_index_column=self.datetime_index_column,
                      bucket=config['cfg_s3_info']['bucket'])
        
    def _test_read_data(self, database_type:str, start:str, end:str):
        dataset = self._setup_file_config() if database_type =="file" else self._setup_S3_file_config()
        well_name = "ACRUS1"
        inference_data = dataset.read_data(well_name, start, end,False)
        date_range = dataset.get_date_range(start, end)
        for i in range(len(date_range)-1):
            file_start = date_range[i].strftime("%Y%m%d")
            file_end = date_range[i+1].strftime("%Y%m%d")
            self.assertTrue(f"{well_name}_ROC_PROCESSED_DATA_{file_start}_{file_end}.csv" in inference_data, f"file not fetched: {well_name}_ROC_PROCESSED_DATA_{file_start}_{file_end}.csv")
    
    def test_read_data_file(self):
        self._test_read_data("file","2017-01-01","2018-01-01")
        self._test_read_data("file","2016-01-01","2017-01-01")
        
    def test_s3_data_file(self):
        self._test_read_data("S3","2017-01-01","2018-01-01")
        self._test_read_data("S3","2016-01-01","2017-01-01")
    
if __name__ == "__main__":
    unittest.main()