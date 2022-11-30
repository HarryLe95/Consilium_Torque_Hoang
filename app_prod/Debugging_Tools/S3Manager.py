from utils.advancedanalytics_util import S3
from pathlib import Path
from utils.PathManager import PathManager
from datetime import datetime  
import pandas as pd
from botocore.errorfactory import ClientError

import logging 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class S3Manager(S3):
    """aa_utils wrapper that provides methods for looking up file hierarchy, list files from directory and downloading combined csv files

    Args:
        S3 (_type_): advancedanalytics_utils.py S3 class 
    """
    @classmethod 
    def from_config(cls,config_path:str='config.csv'):
        config_file = PathManager.read_config(config_path)
        info = config_file['cfg_s3_info']
        return cls(info)
    
    def __init__(self, info:dict):
        super().__init__(info)
        if 'bucket' in info:
            self.add_active_bucket(info['bucket'])

    def add_active_bucket(self, bucket:str):
        """Method to add active bucket 

        Args:
            bucket (str): bucket name 

        Raises:
            e: raise 404 Not Found if bucket does not exist or 403 Forbidden if user has no access 
        """
        try:
            self.client.head_bucket(Bucket=bucket)
            self.bucket = bucket 
        except Exception as e:
            raise e

    def _list_dir(self, prefix:str='', bucket:str=None, ) -> tuple[dict, list]:
        """Internal method for listing directory in a bucket given a prefix

        Example: if an S3 folder structure of MyBucket is as follows:
        ROC/
            PROCESSED_DATA/
            SOLAR_DATA/
            LABEL_DATA/
                HUMAN_LABEL/
                MACHINE_LABEL/

        Then:
        >>> S3Manager._list_dir('',MyBucket)
        >>> {'':['ROC']}, ['ROC']
        >>> S3Manager._list_dir('ROC',MyBucket)
        >>> {'ROC': ['ROC/PROCESSED_DATA', 'ROC/SOLAR_DATA', 'ROC/LABEL_DATA']}, ['ROC/PROCESSED_DATA', 'ROC/SOLAR_DATA', 'ROC/LABEL_DATA']
        >>> S3Manager._list_dir('ROC/LABEL_DATA', MyBucket )
        >>> {'ROC/LABEL_DATA': ['ROC/LABEL_DATA/HUMAN_LABEL','ROC/LABEL_DATA/MACHINE_LABEL']}, ['ROC/LABEL_DATA/HUMAN_LABEL','ROC/LABEL_DATA/MACHINE_LABEL']

        Args:
            bucket (str): s3 bucket
            prefix (str, optional): subdirectory prefix. Defaults to ''.

        Returns:
            tuple[dict, list]: dictionary and list of all sub_directories under the current prefix directory at 1 level
        """
        if bucket is None:
            try:
                bucket=self.bucket
            except: 
                raise ValueError("Either a bucket must be provided in argument list or the object must have an active bucket attribute.")

        prefix_path = prefix +'/' if prefix != '' else prefix
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix_path, Delimiter='/')
        try:
            sub_folder_prefix = [Path(item['Prefix']).as_posix() for item in response['CommonPrefixes']]
            sub_folder_rel_path = {prefix: sub_folder_prefix}
            return sub_folder_rel_path, sub_folder_prefix
        except KeyError as e:
            return {}, []

    def list_dir(self, prefix:str='', bucket:str=None, recursive:bool=False) -> dict:
        """List directory in a bucket given a prefix. If bucket parameter is not provided, use the object's active bucket.

        Example: if an S3 folder structure of MyBucket is as follows:
        ROC/
            PROCESSED_DATA/
            SOLAR_DATA/
            LABEL_DATA/
                HUMAN_LABEL/
                MACHINE_LABEL/

        Then:
        >>> S3Manager.list_dir('', MyBucket)
        >>> {'':['ROC']}
        >>> S3Manager.list_dir('ROC', MyBucket,)
        >>> {'ROC': ['ROC/PROCESSED_DATA', 'ROC/SOLAR_DATA', 'ROC/LABEL_DATA']}
        >>> S3Manager.list_dir('ROC/LABEL_DATA', MyBucket, )
        >>> {'ROC/LABEL_DATA': ['ROC/LABEL_DATA/HUMAN_LABEL','ROC/LABEL_DATA/MACHINE_LABEL']}
        >>> S3Manager.list_dir('', MyBucket,,True)
        >>> {'':['ROC'], 'ROC': ['ROC/PROCESSED_DATA', 'ROC/SOLAR_DATA', 'ROC/LABEL_DATA'], 'ROC/LABEL_DATA': ['ROC/LABEL_DATA/HUMAN_LABEL','ROC/LABEL_DATA/MACHINE_LABEL']}

        Args:
            bucket (str): s3 bucket
            prefix (str, optional): subdirectory prefix. Defaults to ''.
            recursive (bool): if False, only get directories that are direct children of the directory specified in prefix. Otherwise, get all descendent directories. Defaults to False.

        Returns:
            dict: dictionary whose values are immediate children directory of the corresponding key directory.
        """
        ls, next = self._list_dir(bucket=bucket, prefix=prefix)
        status=True
        if recursive: 
            while status:
                if len(next)==0:
                    break
                prefix = next.pop()
                current_ls, next_prefix = self._list_dir(bucket=bucket, prefix=prefix)
                next.extend(next_prefix)
                ls.update(current_ls)
        return ls 

    def list_files(self, prefix:str='ROC', bucket:str=None,  file_prefix: str=None) -> list:
        """List all files in directory specified by prefix containing file_prefix. If bucket is not provided, use the object's active bucket

        Example: folder structure in MyBucket
        LABEL_DATA/
            V1_LABEL_20180101_20180201.csv
            V1_LABEL_20180201_20180301.csv
            V2_LABEL_20180101_20180201.csv

        >>> S3Manager.list_files(MyBucket, 'LABEL_DATA')
        >>> [V1_LABEL_20180101_20180201.csv, V1_LABEL_20180201_20180301.csv, V2_LABEL_20180101_20180201.csv]
        >>> S3Manager.list_files(MyBucket, 'LABEL_DATA', 'V1')
        >>> [V1_LABEL_20180101_20180201.csv, V1_LABEL_20180201_20180301.csv]
        >>> S3Manager.list_files(MyBucket, 'LABEL_DATA', 'V2')
        >>> [V2_LABEL_20180101_20180201.csv]

        Args:
            bucket (str): bucket
            prefix (str, optional): directory to file. Defaults to 'ROC'.
            file_prefix (str, optional): file prefix. Defaults to None.

        Returns:
            list: list of all files in prefix directory containing file_prefix as prefix
        """
        if bucket is None:
            try:
                bucket=self.bucket
            except: 
                raise ValueError("Either a bucket must be provided in argument list or the object must have an active bucket attribute.")
        prefix_path = prefix +'/' if file_prefix is None else prefix+'/'+file_prefix
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix_path, Delimiter='/')
        try:
            keys = [item['Key'] for item in response['Contents']]
        except Exception as e:
            logger.error(f"Query response doesn't contain Contents. Due to either invalid filepath: {prefix} or invalid file_prefix: {file_prefix}")
            raise e
        while response['IsTruncated']:
            continuation_token = response['NextContinuationToken']
            response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix_path, Delimiter='/', ContinuationToken=continuation_token)
            next_keys = [item['Key'] for item in response['Contents']]
            keys.extend(next_keys)
        return keys
    
    @staticmethod
    def parse_date(date:str, strp_format='%Y-%m-%d') -> datetime:
        """Parse str as datetime object

        Args:
            date (str): datestring
            strp_format (str, optional): format. Defaults to '%Y-%m-%d'.

        Returns:
            datetime: datetime object from date
        """
        try:
            return datetime.strptime(date, strp_format)
        except:
            raise ValueError(f"Incompatiable input date {date} and format: {strp_format}")

    @classmethod
    def get_filename(cls,
                well_cd:str, 
                file_prefix:str, 
                start:datetime|str, 
                end:datetime|str, 
                strp_format:str='%Y%m%d',
                strf_format:str='%Y%m%d',
                file_suffix:str='.csv') -> str:
        """Get filename that adheres to Santos' naming convention

        Example: 
        >>> S3Manager.get_filename("MOOMBA","SOLAR_DATA","2020-01-01","2020-02-01","%Y-%m-%d")
        >>> MOOMBA_SOLAR_DATA_2020-01-01_2020_02_01.csv
        Args:
            well_cd (str): well_cd 
            file_prefix (str): file_prefix
            start (datetime | str): start date
            end (datetime | str): end date
            strp_format (str, optional): format to read start and end if given as string. Defaults to '%Y%m%d'.
            strf_format (str, optional): format suffix date in file name. Defaults to '%Y%m%d'.
            file_suffix (str, optional): file_suffixension. Defaults to 'csv'.
        
        Returns:
            str: formatted filename 
        """
        if isinstance(start,str):
            start = cls.parse_date(start, strp_format)
        if isinstance(end,str):
            end = cls.parse_date(end, strp_format)
        fn = '{}_{}_{}_{}{}'.format(well_cd, 
                                    file_prefix, 
                                    start.strftime(strf_format), 
                                    end.strftime(strf_format), 
                                    file_suffix)
        return fn

    @classmethod
    def get_date_range(self, start_date:str, end_date:str, freq:str='monthly_start', strp_format:str='%Y-%m-%d') -> pd.Series:
        """Get a date range from strings specifying the start and end date

        Args:
            start_date (str): start date
            end_date (str): end date
            freq (str): one of monthly_start, monthly_end, hourly, minutely. Defaults to monthly_start.
            strp_format (str): how the start and end date strings should be formatted. Defaults to Y-M-D

        Returns:
            pd.Series: date range 
        """
        start_date = self.parse_date(start_date, strp_format=strp_format)
        end_date = self.parse_date(end_date, strp_format=strp_format)
        freq_dict = {"monthly_start": "MS", "monthly_end": "M",
                     "daily": "D","hourly": "H", "minutely": "T"}
        return pd.date_range(start_date, end_date, freq=freq_dict[freq])

    def exists(self, bucket:str, filepath: str) -> bool:
        try:
            self.client.head_object(Bucket=bucket, Key=filepath)
            return True
        except ClientError as e:
            logging.warning(f"Object {filepath} doesn't exists on the provided S3 Bucket.")
            return False 

    def read_from_storage(self,
                          path:str,
                          file_prefix: str,
                          well_cd: str,
                          start: datetime|str,
                          end: datetime|str,
                          bucket:str=None,
                          file_suffix:str='.csv',
                          strp_format:str='%Y-%m-%d',
                          strf_format:str='%Y%m%d',
                          combine_output:bool=False,
                          **kwargs,  
                          ) -> pd.DataFrame:
        """Read and combined files on S3 storage from start to end.

        Example: to read weather data for Jackson from 2018 to 2019: 
        >>> S3 = S3Manager(info)
        >>> S3.read_from_storage(bucket=MyBucket, path="ROC/SOLAR_DATA", file_prefix='SOLAR_DATA', 
                                  well_cd = "Jackson", start="2018-01-01", end="2019-01-01")
        Args:
            bucket (str): bucket
            path (str): S3 path to file
            file_prefix (str): file_prefix
            well_cd (str): item code
            start (datetime | str): start date
            end (datetime | str): end date
            strp_format (str, optional): format to intepret start and end dates. Defaults to '%Y-%m-%d'.
            strf_format (str, optional): storage file's date format . Defaults to '%Y%m%d'.
            file_suffix (str, optional): file extension. Defaults to 'csv'.

        Returns:
            pd.DataFrame: combined dataframe
        """
        if bucket is None:
            try:
                bucket=self.bucket
            except: 
                raise ValueError("Either a bucket must be provided in argument list or the object must have an active bucket attribute.")
        #Concatenating all csvs at return is twice as fast as if doing it during intermediate steps
        alldf = {}
        date_range = self.get_date_range(start, end, freq='monthly_start', strp_format=strp_format) #Files on S3 are compiled on the first day of every month
        kwargs={'bucket': bucket, 'path': path, 'partition_mode':None}
        for d in range(len(date_range)-1):
            fs, fe = date_range[d], date_range[d+1]
            fn = self.get_filename(well_cd=well_cd, file_prefix=file_prefix, start=fs, end=fe, file_suffix=file_suffix,strf_format=strf_format)
            kwargs['file']=fn
            filepath = Path(path) / fn
            try:
                result = self.read(sql=None, args={}, edit=[], orient='df', do_raise=False, **kwargs)
                alldf[filepath] = result
            except Exception as e:
                logger.error(e)
                raise e 

        if combine_output:
            all_output = [data for data in alldf.values()]
            return pd.concat(all_output,axis=0,ignore_index=True)
        return alldf



