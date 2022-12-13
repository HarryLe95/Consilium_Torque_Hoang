from typing import Sequence 
from datetime import timedelta
import datetime 
from dateutil import relativedelta
from copy import deepcopy
import logging

import pandas as pd 

from utils.advancedanalytics_util import aauconnect_

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PROCESSOR_MIXIN:
    """Utility object that handles data processing and transformation

    Methods:
        extract_time_period - get start and end timestamps of input data 
        fill_missing_data - fill missing timestamps in input data from start to end 
        select_features - select relavent features required for analysis
        create_mask_and_fill_nan - get mask for missing data features 
        normalise_data - apply data normalisation 
        aggregate_data - group data by date and aggregate data within a day to a list 
        process_data_ - apply all transformation methods sequentially
    """
    @classmethod 
    def resample_to_1min(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Resample subminute information. Done to remove duplicate

        Args:
            data (pd.DataFrame): input dataframe 

        Returns:
            pd.DataFrame: output dataframe 
        """
        data = data.groupby(data.index).mean()
        return data.asfreq('T', method='nearest').resample('1min').mean()
    
    @classmethod 
    def extract_time_period(cls, data:pd.DataFrame)->tuple:
        """Extract the start and end timestamps from a given dataset. 
        start timestamp is rounded down to the nearest %Y-%m-%d 00:00:00
        end timestamp is rounded up to the nearest %Y-%m-%d 23:59

        Args:
            data (pd.DataFrame): _description_

        Returns:
            tuple: start and end timestamps 
        """
        start = data.index.min().replace(hour=0,minute=0)
        end = data.index.max().replace(hour=23,minute=59)
        return start, end 
    
    @classmethod
    def fill_missing_date(cls, data: pd.DataFrame, start:str, end:str) -> pd.DataFrame:
        """Fill minutely missing timestamps

        Args:
            data (pd.DataFrame): data 
            start (str): start date - minutely datetime
            end (str): end date - minutely datetime 

        Returns:
            pd.DataFrame: dataframe with nan values for missing time gaps 
        """
        index = pd.date_range(start,end,freq="T")
        return data.reindex(index)
        
    @classmethod 
    def select_features(cls, data: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
        """Select relevant features from dataframe 

        Args:
            data (pd.DataFrame): input dataframe
            features (Sequence[str]): feature list 

        Returns:
            pd.DataFrame: dataframe with selected features
        """
        for feature in features: 
            if feature not in data.columns: 
                logger.debug(f"Missing feature {feature}")
        return data.loc[:,features]
    
    @classmethod 
    def create_mask_and_fill_nan(cls, data: pd.DataFrame, features: Sequence[str], fill_method:str="interpolate") -> pd.DataFrame:
        """Create a binary mask for each feature describing whether the data point at corresponding mask is raw or interpolated 

        Args:
            data (pd.DataFrame): input dataframe 
            features (Sequence[str]): relevant features for modelling purposes 
            fill_method (str, optional): how to fill nan values - either zero or interpolate

        Returns:
            pd.DataFrame: new data frame with mask and nan value filled 
        """
        for feature in features: 
            data[f'Mask_{feature}']=1-data[feature].isna()
            if fill_method == "zero":
                data[feature] = data[feature].fillna(value=0)
            else:
                data[feature] = data[feature].interpolate(method='linear')    
        return data         
    
    @classmethod 
    def normalise_data(cls, data:pd.DataFrame, normalise_params)->pd.DataFrame:
        return data
    
    @classmethod 
    def aggregate_data(cls, data: pd.DataFrame) -> pd.DataFrame: 
        """Aggregate data so that each row contains a list of 1440 data points

        Args:
            data (pd.DataFrame): input dataframe 

        Returns:
            pd.DataFrame: aggregated dataframe
        """
        data = data.groupby(data.index.date).agg(list)
        data.index = pd.to_datetime(data.index)
        data.index.name = "TS"
        return data
    
    @classmethod
    def process_data_(cls, data: pd.DataFrame, features: Sequence[str], fill_method:str, normalise_params:dict) -> pd.DataFrame:
        """Apply transformation on unprocessed data 

        Args:
            data (pd.DataFrame): raw dataframe with index as date-time objects 
            features (Sequence[str]): relevant features to select from data.
            fill_method (str): method to fill nan values 
            normalise_params (dict): normalisation parameters 

        Returns:
            pd.DataFrame: processed data frame 
        """
        data = cls.resample_to_1min(data)
        start, end = cls.extract_time_period(data)
        data = cls.fill_missing_date(data, start, end)
        data = cls.select_features(data, features)
        data = cls.create_mask_and_fill_nan(data, features, fill_method)
        data = cls.normalise_data(data, normalise_params)
        return cls.aggregate_data(data)
    
class FILENAMING_MIXIN:
    """Utility object that handles datetime and filenaming
    
    Methods:
        parse_date: parse input string to output date object
        get_filename: parse filename components (well_cd,dates,etc) and return the filename 
        get_metadata_name: parse metadata components and return the file metadata
        get_date_range: return pd.DatetimeIndex from start to end 
        get_output_time_slice: return date slices for data fetching
    """
    #TESTED
    @staticmethod
    def parse_date(date:str|datetime.datetime|datetime.date, strp_format='%Y-%m-%d') -> datetime.date:
        """Get datetime.date object from input date. If date is a string, parse datestring using format strp_format 

        Args:
            date (str, datetime.datetime, datetime.date): datestring
            strp_format (str, optional): format. Defaults to '%Y-%m-%d'.

        Returns:
            datetime.date: datetime.date object from date
        """
        if isinstance(date, datetime.datetime):
            return date.date()
        if isinstance(date,datetime.date):
            return date
        if isinstance(date, str):
            try:
                return datetime.datetime.strptime(date, strp_format).date()
            except:
                raise ValueError(f"Incompatiable input date {date} and format: {strp_format}")

    #TESTED
    @classmethod
    def get_filename(cls,
                well_cd:str, 
                file_prefix:str, 
                start:datetime.date, 
                end:datetime.date,
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
    
    #TESTED  
    @classmethod 
    def get_metadata_name(cls,
                well_cd:str, 
                file_prefix:str, 
                file_suffix:str='.csv') -> str:
        """Get metadata filename that adheres to Santos' naming convention

        Example: 
        >>> S3Manager.get_metadata_name("TIRRA80","ROC_PROCESSED_DATA",".csv")
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
    def get_event_log_name(cls, well_cd:str, file_prefix:str, file_suffix:str=".csv")->str:
        """Get event log filename that adheres to Santos' naming convention

        Example: 
        >>> S3Manager.get_event_log_name("TIRRA80","ROC_PROCESSED_DATA",".csv")
        >>> TIRRA80_ROC_PROCESSED_DATA_EVENTS.csv
        Args:
            well_cd (str): well_cd 
            file_prefix (str): file_prefix
            file_suffix (str, optional): file_suffixension. Defaults to 'csv'.
        
        Returns:
            str: formatted filename 
        """
        return f'{well_cd}_{file_prefix}_EVENTS{file_suffix}'
    
    #TESTED
    @classmethod
    def get_date_range(self, 
                       start_date:datetime.date, 
                       end_date:datetime.date) -> pd.DatetimeIndex:
        """Get a date range from strings specifying the start and end date. 
        
        start_date is rounded down to the first day of the month. end_date is rounded to the first day of next month (if current day is not 1).
        Santos files recorded data from the 1st of one month to the minute right before the 1st of next month. For instance: 
        ACRUS1_PROCESSED_DATA_20170101_20170201 records data in ACRUS1 from 1/1/2017 to 31/1/2017. Therefore, if start_date = 2017-01-05, end_date = 2017-02-06, the date range includes
        2017_01_01_2017_02_01 and 2017_02_01_2017_03_01
        
        Args:
            start_date (datetime.date): start date
            end_date (datetime.date): end date

        Returns:
            pd.Series: date range 
        """
        if start_date > end_date: 
            raise ValueError(f"End date {end_date} must come after start date {start_date}.")
        if end_date.day != 1:
            end_date = end_date.replace(day = 1)
            end_date += relativedelta.relativedelta(months=1) 
        start_date = start_date.replace(day = 1) 
        return pd.date_range(start_date, end_date, freq="MS")
    
    @classmethod
    def get_output_time_slice(cls, start:datetime.date, end:datetime.date) -> tuple[datetime.datetime,datetime.datetime]:
        """Get start and end time for running inference. Add time attribute (hour, minute) information to date objects start and end.
        
        start is promoted as it is (hour=0,minute=0)   
        end is promoted by subtracting 1 minute from it. - i.e. end = end - timedelta(minute=1)
          
        Example:
            If today is 3rd of Nov 2022, in live mode (start=2022-11-02, end=2022-11-03), the model runs inference for data 
            on the 2nd of Nov 2022. The output start and end time indices are - '2022-11-02 00:00' and '2022-11-02 23:59'. This means 
            the csv fetched will be ACRUS1_PROCESSED_DATA_20221101_20221201.csv, which will be further sliced from "2022-11-02 00:00" to 
            "2022-11-02 23:59:

        Args: 
            start (datetime.date): start date. Yesterday's date in live mode. Output start index = start 
            end (datetime.date): end date. Today's date in live mode. Output end index = end - 1 minute

        Returns:
            tuple[datetime.datetime,datetime.datetime]: start_, end_ time indices 
        """
        minute_string_format = "%Y-%m-%d %H:%M"
        def process_date_slices(d: datetime.date, offset:timedelta=timedelta(minutes=0)) -> datetime.datetime:
            """Convert datetime.date to datetime.datetime, with the time attribute specified by offset

            Args:
                d (datetime.date): input datetime.date object 
                offset (timedelta, optional): the amount of offset. Defaults to timedelta(minutes=0).

            Returns:
                datetime.datetime: casted datetime.datetime object
            """
            d = datetime.datetime.strptime(d.strftime(minute_string_format),minute_string_format) 
            return d + offset
        start_ = process_date_slices(start)
        end_ = process_date_slices(end, timedelta(minutes=-1))
        return start_, end_
        
class DataOperator(PROCESSOR_MIXIN, FILENAMING_MIXIN):
    def __init__(self, 
                 connection_type: str,
                 path:str, 
                 file_prefix:str, 
                 file_suffix:str,
                 features:Sequence[str], 
                 fill_method:str="zero",
                 normalise_params:dict=None,
                 datetime_index_column:str="TS",
                 tzname:str=None,
                 **kwargs) -> None:
        """ETL class concerning well tag and metadata. 

        Methods:
            setup: setup database connection based on specified keywords 
            read_data: read tag data from database 
            read_metadata: read metadata from database 
            read_event_log: read event log from database 
            write_metadata: write metadata to database 
            write_event_log: write event log to database 
            process_data: transform raw data for analytic purposes 

        Args:
            connection_type (str): database connection type. Currently supporting "file" and "s3"
            path (str): path to tag/metadata
            file_prefix (str): file prefix
            file_suffix (str): file suffix
            features (Sequence[str]): relevant tags for analysis 
            fill_method (str, optional): method to fill missing data. Either "zero" or "interpolate". Defaults to "zero".
            normalise_params (dict, optional): normalisation dictionary. Defaults to None.
            datetime_index_column (str, optional): name of tag column to be treated as datatime index. Defaults to "TS".
            tzname (str, optional): data time zone. Defaults to None.
        """
        self.connection_type = connection_type
        self.tzname = tzname 
        self.partition_mode = None 
        self.path = path
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.features = features
        self.fill_method = fill_method 
        self.normalise_params = normalise_params
        self.datetime_index_column = datetime_index_column
        self.kwargs = {"path":self.path, "partition_mode":self.partition_mode}
        self.setup(kwargs)

    def setup(self,kwargs:dict):
        """Set up aauconnection_ based on connection_type. additional keywords contained in kwargs

        Args:
            kwargs (dict): additional keywords

        Raises:
            ValueError: if no connection is setup.
        """
        if self.connection_type == "s3":
            self.bucket = kwargs['bucket']
            self.region = kwargs['region']
            self.user   = kwargs['user']
            self.connection = aauconnect_(info={"connection_type": self.connection_type, 
                                                "tzname":self.tzname, "bucket":self.bucket, 
                                                "region":self.region, "user": self.user, 
                                                "partition_mode":self.partition_mode})
        if self.connection_type == "file":
            self.connection = aauconnect_(info={"connection_type": self.connection_type, 
                                                "tzname": self.tzname, 
                                                "partition_mode": self.partition_mode})
        if not hasattr(self, "connection"):
            logger.error("Data connection uninitialised.")
            raise ValueError("Data connection uninitialised.")
    
    #TESTED
    def read_data(self, well_cd:str, start:str|datetime.date, end:str|datetime.date, concat:bool=True, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> dict|pd.DataFrame:
        """Read well data from database 
        
        Args:
            well_cd (str): well code
            start (str): start date
            end (str): end date 
            concat (bool, optional): whether to concatenate to form a concatenated dataframe. Defaults to True.
            strp_format (str, optional): input dates (start, end) formats. Defaults to '%Y-%m-%d'.
            strf_format (str, optional): output date string format (format on actual filename). Defaults to '%Y%m%d'.

        Raises:
            e: Credential Exception or File Not on Database Exception

        Returns:
            dict|pd.DataFrame: output object: dict whose keys are dates and values are the dataframes, or a concatenated dataframe
        """
        start_datetime = self.parse_date(start, strp_format)
        end_datetime = self.parse_date(end, strp_format)
        response = {}
        date_range = self.get_date_range(start_datetime, end_datetime)
        kwargs=deepcopy(self.kwargs)
        
        #Read data from database 
        for d in range(len(date_range)-1):
            #Get filename
            file_start, file_end = date_range[d], date_range[d+1]
            file_name = self.get_filename(well_cd=well_cd, file_prefix=self.file_prefix, 
                                        start=file_start, end=file_end, file_suffix=self.file_suffix,
                                        strf_format=strf_format)
            kwargs['file']=file_name
            logger.debug(f"Reading file: {file_name} from {self.connection_type} database.")
            try:
                result = self.connection.read(sql=None, args={}, edit=[], orient='df', do_raise=False, **kwargs)
                if result is not None:
                    result[self.datetime_index_column] = pd.to_datetime(result[self.datetime_index_column])
                    result.set_index(self.datetime_index_column,inplace=True)
                    
                    #Slice output datetime if start and end dates are not the first day of month
                    if start_datetime >= file_start.date(): 
                        start_,_ = self.get_output_time_slice(start_datetime, file_end)
                        result = result.loc[start_:,:]            
                    if end_datetime <= file_end.date():
                        start_, end_ = self.get_output_time_slice(end_datetime.replace(day=1), end_datetime)
                        result = result.loc[:end_,:]
                    response[file_name] = result
            except Exception as e:
                logger.debug(f"Error reading file: {file_name}, error message: {e}")
                raise e 

        #Concatenate data to form a single dataframe 
        if concat:
            logger.debug(f"Concatenating all read files.")
            try:
                all_output = [data for data in response.values() if data is not None]
                all_df =  pd.concat(all_output,axis=0,ignore_index=False)
                return all_df
            except Exception as e:
                logger.error(f"Error encountered trying to concatenate files: {e}")
                return None
        return response
    
    #TESTED
    def read_metadata(self, well_cd:str) -> pd.DataFrame: 
        """Read meta data from database. Metadata file name is well_cd_ROC_PROCESSED_DATA_LAST.csv 

        Args:
            well_cd (str): well code

        Raises:
            e: invalid credential exception or file does not exist exception

        Returns:
            pd.DataFrame: metadata dataframe
        """
        logger.debug(f"Getting metadata for well: {well_cd}")
        kwargs = deepcopy(self.kwargs)
        file_name = self.get_metadata_name(well_cd, self.file_prefix, self.file_suffix)
        kwargs['file']=file_name
        try: 
            result = self.connection.read(sql=None, args={}, edit=[], orient="df", do_raise=False, **kwargs)
            if result is not None: 
                return result
        except Exception as e:
            logger.error(f"Error getting metadata for well: {well_cd}. Error message: {e}")
            raise e 

    def write_metadata(self, metadata:pd.DataFrame, well_cd:str)->None:
        """Write metadata to external database 

        Args:
            metadata (dict): metadata dataframe for one day inference
            well_cd (str): well name

        Raises:
            e: exception caught
        """
        logger.debug(f"Updating metadata for well: {well_cd}")
        kwargs = deepcopy(self.kwargs)
        kwargs['append']=False
        file_name = self.get_metadata_name(well_cd, self.file_prefix, self.file_suffix)
        kwargs['file']=file_name
        try:
            self.connection.write_many(sql=None, args=metadata, **kwargs)
        except Exception as e: 
            logger.error(f"Error writing metadata")
            raise e 
    
    def read_event_log(self, well_cd:str)->pd.DataFrame:
        """Read event log from database 

        Args:
            well_cd (str): well to get event log from

        Raises:
            e: file not exist exception 

        Returns:
            pd.DataFrame: event log dataframe 
        """
        logger.debug(f"Reading event log data for well: {well_cd}")
        kwargs = deepcopy(self.kwargs)
        file_name = self.get_event_log_name(well_cd, self.file_prefix, self.file_suffix)
        kwargs['file']=file_name
        try: 
            result = self.connection.read(sql=None, args={}, edit=[], orient="df", do_raise=False, **kwargs)
            if result is not None: 
                return result
        except Exception as e:
            logger.error(f"Error getting event log data for well: {well_cd}. Error message: {e}")
            raise e  
        
    def write_event_log(self, event_output:pd.DataFrame, well_cd:str, append:bool)->None:
        logger.debug(f"Updating event log for well: {well_cd}")
        kwargs = deepcopy(self.kwargs)
        kwargs['append']=False
        file_name = self.get_event_log_name(well_cd, self.file_prefix, self.file_suffix)
        
        if append: #Append to event log logic
            historical_log = self.read_event_log(well_cd)
            if historical_log is not None:
                event_output = pd.concat([historical_log, event_output], axis = 0)
                
        kwargs['file']=file_name
        try:
            self.connection.write_many(sql=None, args=event_output, **kwargs)
        except Exception as e: 
            logger.error(f"Error writing event log")
            raise e 
        
    def process_data(self, data:pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to input data

        Args:
            data (pd.DataFrame): input data

        Returns:
            pd.DataFrame: transformed data
        """
        return self.process_data_(data, self.features, self.fill_method, self.normalise_params)
    
class TorqueOperator(DataOperator):
    def __init__(self, 
                 connection_type: str,
                 path:str, 
                 file_prefix:str, 
                 file_suffix:str,
                 features:Sequence[str], 
                 fill_method:str="zero",
                 normalise_params:dict=None,
                 datetime_index_column:str="TS",
                 tzname:str=None,
                 **kwargs) -> None:
        """ETL class concerning TORQUE well tag and metadata. 

        Methods:
            setup: setup database connection based on specified keywords 
            read_data: read tag data from database 
            read_metadata: read metadata from database 
            read_event_log: read event log from database 
            write_metadata: write metadata to database 
            write_event_log: write event log to database 
            process_data: transform raw data for analytic purposes
            read_completion_data: read completion data from database 
            read_label_data: read label data from database 

        Args:
            connection_type (str): database connection type. Currently supporting "file" and "s3"
            path (str): path to tag/metadata
            file_prefix (str): file prefix
            file_suffix (str): file suffix
            features (Sequence[str]): relevant tags for analysis 
            fill_method (str, optional): method to fill missing data. Either "zero" or "interpolate". Defaults to "zero".
            normalise_params (dict, optional): normalisation dictionary. Defaults to None.
            datetime_index_column (str, optional): name of tag column to be treated as datatime index. Defaults to "TS".
            tzname (str, optional): data time zone. Defaults to None.
        """
        super().__init__(connection_type, path, file_prefix, file_suffix, features, fill_method, normalise_params, datetime_index_column, tzname, **kwargs)
                    
    def read_label_data(self) -> pd.DataFrame:
        """Read TORQUE label data frame 

        Raises:
            e: file not found exception 

        Returns:
            pd.DataFrame: label dataframe 
        """
        logger.debug(f"Getting label data.")
        kwargs = deepcopy(self.kwargs)
        kwargs['file'] = "All_Roma_Flush_Fail_PCPChange.csv"
        try: 
            result = self.connection.read(sql=None, args={}, edit=[], orient="df", do_raise=False, **kwargs)
            if result is not None: 
                result['Event Date'] = pd.to_datetime(result['Event Date'], format="%d/%m/%Y")
                result.dropna(subset=['Event Date'], inplace=True)
                result.set_index("Event Date", inplace=True)
                return result
        except Exception as e:
            logger.error(f"Error getting label data. Error message: {e}")
            raise e 
        
    def read_completion_data(self) -> pd.DataFrame:
        """Read completion turndown dataframe 

        Raises:
            e: file not found exception 

        Returns:
            pd.DataFrame: completion turndown dataframe 
        """
        logger.debug(f"Getting completion turndown data.")
        kwargs = deepcopy(self.kwargs)
        kwargs['file'] = "All_Roma_Completion_Turndown_ProductionStatus.csv"
        try: 
            result = self.connection.read(sql=None, args={}, edit=[], orient="df", do_raise=False, **kwargs)
            if result is not None: 
                return result
        except Exception as e:
            logger.error(f"Error getting completion turndown data. Error message: {e}")
            raise e 
        
    @classmethod
    def process_data_(cls, data: pd.DataFrame, features: Sequence[str], fill_method:str, normalise_params:dict) -> pd.DataFrame:
        """Apply transformation on unprocessed data 

        Args:
            data (pd.DataFrame): raw dataframe with index as date-time objects 
            features (Sequence[str]): relevant features to select from data.
            fill_method (str): method to fill nan values 
            normalise_params (dict): normalisation parameters 

        Returns:
            pd.DataFrame: processed data frame 
        """
        data = cls.resample_to_1min(data)
        data = cls.select_features(data, features)
        data = cls.create_mask_and_fill_nan(data, features, fill_method)
        data = cls.normalise_data(data, normalise_params)
        return data 