from typing import Sequence 
from datetime import timedelta
import datetime 
from dateutil import relativedelta
from copy import deepcopy
import logging

import pandas as pd 
import numpy as np 

from Dataset.generic import ABC_Dataset
from utils.advancedanalytics_util import AAUConnection, S3Connection, FileConnection, AAPandaSQL, aauconnect_

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PROCESSOR_MIXIN:
    #TODO: fix process data once finalised 
    @classmethod
    def process_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        return data
    
class FILENAMING_MIXIN:
    @staticmethod
    def parse_date(date:str|datetime.datetime, strp_format='%Y-%m-%d') -> datetime.datetime:
        """Parse str as datetime object

        Args:
            date (str): datestring
            strp_format (str, optional): format. Defaults to '%Y-%m-%d'.

        Returns:
            datetime.datetime: datetime object from date
        """
        try:
            return datetime.datetime.strptime(date, strp_format)
        except:
            raise ValueError(f"Incompatiable input date {date} and format: {strp_format}")

    @classmethod
    def get_filename(cls,
                well_cd:str, 
                file_prefix:str, 
                start:datetime.datetime, 
                end:datetime.datetime,
                strf_format:str='%Y%m%d',
                file_suffix:str='.csv') -> str:
        """Get filename that adheres to Santos' naming convention

        Example: 
        >>> S3Manager.get_filename("MOOMBA","SOLAR_DATA","2020-01-01","2020-02-01","%Y-%m-%d")
        >>> MOOMBA_SOLAR_DATA_2020-01-01_2020_02_01.csv
        Args:
            well_cd (str): well_cd 
            file_prefix (str): file_prefix
            start (datetime.datetime): start date
            end (datetime.datetime): end date
            strf_format (str, optional): format suffix date in file name. Defaults to '%Y%m%d'.
            file_suffix (str, optional): file_suffixension. Defaults to 'csv'.
        
        Returns:
            str: formatted filename 
        """
        return f"{well_cd}_{file_prefix}_{start.strftime(strf_format)}_{end.strftime(strf_format)}{file_suffix}"
        
    @classmethod 
    def get_metadata_name(cls,
                well_cd:str, 
                file_prefix:str, 
                file_suffix:str='.csv') -> str:
        """Get metadata filename that adheres to Santos' naming convention

        Example: 
        >>> S3Manager.get_filename("TIRRA80","ROC_PROCESSED_DATA",".csv")
        >>> TIRRA80_ROC_PROCESSED_DATA_LAST.csv
        Args:
            well_cd (str): well_cd 
            file_prefix (str): file_prefix
            file_suffix (str, optional): file_suffixension. Defaults to 'csv'.
        
        Returns:
            str: formatted filename 
        """
        return f'{well_cd}_{file_prefix}_LAST{file_suffix}'
    
    @classmethod
    def get_date_range(self, start_date:str, end_date:str, strp_format:str='%Y-%m-%d') -> pd.Series:
        """Get a date range from strings specifying the start and end date. If start and end are different days of the same month and year, calling the method
        returns the interval [month, month+1]

        Args:
            start_date (str): start date
            end_date (str): end date
            strp_format (str): how the start and end date strings should be formatted. Defaults to Y-M-D

        Returns:
            pd.Series: date range 
        """
        start_date = self.parse_date(start_date, strp_format=strp_format).replace(day=1)
        end_date = self.parse_date(end_date, strp_format=strp_format).replace(day=1)
        if end_date == start_date: 
            end_date += relativedelta.relativedelta(months=1) 

        return pd.date_range(start_date, end_date, freq="MS")
        
class Dataset(ABC_Dataset, PROCESSOR_MIXIN, FILENAMING_MIXIN):
    def __init__(self, 
                 connection: AAUConnection, 
                 path:str, 
                 file_prefix:str, 
                 file_suffix:str, 
                 datetime_index_column:str="TS",
                 **kwargs) -> None:
        self.connection = connection 
        self.partition_mode = None 
        self.path = path
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.datetime_index_column = datetime_index_column
        self.kwargs = {"path":self.path, "partition_mode":self.partition_mode}
        self.extract_keyword(kwargs)
        
    def extract_keyword(self, kwargs:dict) -> None:
        """Extract relevant keywords from kwargs 

        Args:
            kwargs (dict): dictionary containing connection specific keywords 
        """
        if isinstance(self.connection, S3Connection):
            self.extract_s3_keyword(kwargs)
        if isinstance(self.connection, FileConnection):
            self.extract_file_keyword(kwargs) 
        if isinstance(self.connection, AAPandaSQL):
            self.extract_sql_keyword(kwargs)
    
    def extract_s3_keyword(self,kwargs:dict) -> None:
        if kwargs is None: 
            raise KeyError("Additional keywords must be provided for S3 connections. kwargs must include: bucket")
        try:
            self.bucket = kwargs['bucket']
        except: 
            raise KeyError("Bucket must be provided in config")
        
    def extract_file_keyword(self,kwargs:dict)-> None:
        pass 
    
    def extract_sql_keyword(self, kwargs:dict)-> None:
        pass 
    
    def get_output_time_index_(self, start:str|datetime.date, end:str|datetime.date, strp_format="%Y-%m-%d") -> tuple[str,str]:
        """Get start and end time for running inference. For example: if today is 2nd of Nov 2022, in live mode, the model runs inference for data 
        on the 1st of Nov 2022. The start and end time indices are - '2016-01-01 00:00' and '2016-01-01 23:59'

        Args: 
            start (str | datetime.date): start date. Output start index = start 
            end (str | datetime.date): end date. Output end index = end - 1 minute
            strp_format (str, optional): _description_. Defaults to "%Y-%m-%d".

        Raises:
            TypeError: if input types are neither string nor datetime.date

        Returns:
            tuple[str,str]: start, end time indices 
        """
        minute_string_format = "%Y-%m-%d %H:%M"
        def process_date_slices(d:str|datetime.date, offset:timedelta=timedelta(minutes=0)):
            if not isinstance(d,(str,datetime.date)):
                raise TypeError(f"Expected type: (str,datetime.date), actual type: {type(d)}")
            if isinstance(d,str):
                d = datetime.datetime.strptime(d, strp_format)
            if isinstance(d,datetime.date): #Convert from datetime.date to datetime.datetime object 
                d = datetime.datetime.strptime(d.strftime(minute_string_format),minute_string_format) 
            d = d + offset
            return d.strftime(minute_string_format)
        start_ = process_date_slices(start)
        end_ = process_date_slices(end, timedelta(minutes=-1))
        return start_, end_
    
    def read_data(self, well_cd:str, start:str, end:str, concat:bool=True, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> pd.DataFrame:
        response = {}
        date_range = self.get_date_range(start, end, strp_format=strp_format)
        kwargs=deepcopy(self.kwargs)
        
   
        for d in range(len(date_range)-1):
            file_start, file_end = date_range[d], date_range[d+1]
            file_name = self.get_filename(well_cd=well_cd, file_prefix=self.file_prefix, 
                                        start=file_start, end=file_end, file_suffix=self.file_suffix,
                                        strf_format=strf_format)
            kwargs['file']=file_name
            try:
                result = self.connection.read(sql=None, args={}, edit=[], orient='df', do_raise=False, **kwargs)
                response[file_name] = result
            except Exception as e:
                raise e 

        if concat:
            try:
                all_output = [data for data in response.values() if data is not None]
                all_df =  pd.concat(all_output,axis=0,ignore_index=True)
                all_df["TS"] = pd.to_datetime(all_df['TS'])
                all_df.set_index("TS",inplace=True)
                start_, end_ = self.get_output_time_index_(start, end, strp_format)
                return all_df.loc[start_:end_,:]
            except Exception as e:
                logger.error(e)
                return None
        return response
    
    def read_metadata(self, well_cd:str) -> pd.DataFrame: 
        kwargs = deepcopy(self.kwargs)
        file_name = self.get_metadata_name(well_cd, self.file_prefix, self.file_suffix)
        kwargs['file']=file_name
        try: 
            result = self.connection.read(sql=None, args={}, edit=[], orient="df", do_raise=False, **kwargs)
            if result is not None: 
                return {well_cd: result}
        except Exception as e:
            raise e 

    def get_metadata(self, wells: Sequence[str]) -> dict[str, pd.DataFrame]:
        metadata_dict = {}
        for well in wells:
            metadata_dict.update(self.read_metadata(well))
        return metadata_dict 
    
    def get_training_dataset_(self, wells:Sequence[str], start:str, end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> pd.DataFrame:
        all_wells = []
        for well in wells:
            well_df = self.read_data(well, start, end, concat=True, strp_format=strp_format, strf_format=strf_format)
            well_df['WELL_CD'] = well
            if well_df is not None: 
                all_wells.append(well_df)
        if len(all_wells)==0:
            raise ValueError("Empty training dataframe")
        return pd.concat(all_wells, axis=0, ignore_index=True)
    
    def get_training_dataset(self, wells:Sequence[str], start:str, end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> np.ndarray:
        return self.process_data(self.get_training_dataset_(wells, start, end, strp_format, strf_format))

    def get_inference_dataset_(self, wells:Sequence[str], start:str, end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> pd.DataFrame:
        all_wells = {}
        for well in wells:
            well_df = self.read_data(well, start, end, concat=True, strp_format=strp_format, strf_format=strf_format)
            if well_df is not None: 
                well_df['WELL_CD'] = well
                all_wells[well]=well_df
            else:
                all_wells[well]=None
        return all_wells
    
    def get_inference_dataset(self, wells:Sequence[str], start:str, end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> dict[str,pd.DataFrame]:
        data_dict = self.get_inference_dataset_(wells, start, end, strp_format, strf_format)
        return {k: self.process_data(v) for k,v in data_dict.items()}
   
if __name__ == "__main__":
    import config.__config__ as base_config
    config = base_config.init()
    connection = aauconnect_(config['cfg_file_info'])
    
    dataset = Dataset(connection, path="C:/Users/HoangLe/Desktop/Consilium_ROC_HOANG/app_prod/roc/PROCESSED_DATA",
                      file_prefix="ROC_PROCESSED_DATA",file_suffix=".csv",bucket=config['cfg_s3_info']['bucket'])
    data_dict = dataset.read_data("ACRUS1","2016-01-01","2017-01-01")
    
    