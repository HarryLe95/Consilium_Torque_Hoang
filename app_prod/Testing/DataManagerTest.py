import unittest 
import datetime

from utils.advancedanalytics_util import aauconnect_
import config.__config__ as base_config
from Dataset.Dataset import Dataset 
from Dataset.DataManager import DataManager

class TestDataManager(unittest.TestCase):
    @classmethod 
    def setUp(self) -> None:
        config = base_config.init()
        self.wells = ['ACRUS1']
        connection = aauconnect_(config['cfg_file_info'])
        self.backfill_date_format="%Y-%m-%d %H:%M"
        self.perform_model_training=True
        self.perform_model_inference=False
        self.datetime_index_column="TS"
        
        self.dataset = Dataset(connection, path="C:/Users/HoangLe/Desktop/Consilium_ROC_HOANG/app_prod/roc/PROCESSED_DATA",
                      file_prefix="ROC_PROCESSED_DATA",file_suffix=".csv",bucket=config['cfg_s3_info']['bucket'], 
                      datetime_index_column=self.datetime_index_column)
    
    def _test_backfill_get_inference_data(self, backfill_start:str)->DataManager:
        run_mode = "backfill"
        manager = DataManager(self.wells, run_mode, backfill_start, self.dataset, self.backfill_date_format, 
                              self.perform_model_training, self.perform_model_inference, self.datetime_index_column)
        inference_data = manager.get_inference_dataset()
        self.assertEqual(set(inference_data.keys()), set(self.wells), f"Test fails for backfill_start input: {backfill_start}. Inference data fetch does not contain the right set of wells")
        for well in self.wells:
            if inference_data[well] is not None:
                end_time = datetime.datetime.strptime(backfill_start, self.backfill_date_format) + datetime.timedelta(days=1)
                self.assertLess(inference_data[well].index.max(),end_time, f"Test fails for backfill_start input: {backfill_start}. Incorrect period fetched")
        return manager 
                
    def test_live_get_inference_data(self) -> None:
        run_mode = "live"
        manager = DataManager(self.wells, run_mode, "2016-01-01 00:00", self.dataset, self.backfill_date_format, 
                              self.perform_model_training, self.perform_model_inference, self.datetime_index_column)
        inference_data = manager.get_inference_dataset()
        self.assertEqual(set(inference_data.keys()), set(self.wells), f"Test fails for live mode. Inference data fetch does not contain the right set of wells")
        for well in self.wells:
            if inference_data[well] is not None:
                start_time = manager.metadata[well][self.datetime_index_column][0]
                end_time = datetime.datetime.strptime(start_time, self.backfill_date_format) + datetime.timedelta(days=1)
                self.assertLess(inference_data[well].index.max(),end_time, f"Test fails for live mode. Incorrect period fetched")
                
    def test_backfill_get_inference_data(self) -> None:
        """Test backfill mode data fetching - 
            Check all wells data fetched - inference data should has data for all wells provide 
            Check for correct date: inference data should be a one-day window from backfill_start input 
        """
        self._test_backfill_get_inference_data("2023-04-04 00:00")
        self._test_backfill_get_inference_data("2017-01-01 00:00")
        self._test_backfill_get_inference_data("2017-01-01 10:00")
        
    def test_update_metadata_success(self) -> None:
        """Test metadata update give successful inference
        """
        manager = self._test_backfill_get_inference_data("2017-01-01 00:00")
        #Mock output: 
        success_output = {well: {"status":0} for well in self.wells}
        manager.update_metadata(success_output)
        for well in self.wells:
            self.assertEqual(manager.metadata[well][self.datetime_index_column][0], "2017-01-02 00:00")
            
    def test_update_metadata_exception(self) -> None:
        """Test metadata update given an exception has occured at inference 
        """
        manager = self._test_backfill_get_inference_data("2017-01-01 00:00")
        #Mock output: 
        success_output = {well: {"status":1} for well in self.wells}
        manager.update_metadata(success_output)
        for well in self.wells:
            self.assertEqual(manager.metadata[well][self.datetime_index_column][0], "2017-01-01 00:00")
    

if __name__ == "__main__":
    unittest.main()