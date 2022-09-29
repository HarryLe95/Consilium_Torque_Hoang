import logging 
from   typing   import List 
from   datetime import timedelta 
from   datetime import datetime as datetime

import pandas as pd 
import numpy as np 
import yaml
from geopy import distance 
from sklearn.preprocessing import StandardScaler

from src.aau.S3Manager import S3Manager
from src.utils.PathManager import Paths as Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


weather_features = ['temperature_2m', 'cloudcover',
       'cloudcover_low', 'cloudcover_mid', 'cloudcover_high',
       'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
       'direct_normal_irradiance']

well_features = ['ROC_VOLTAGE', 'FLOW']
class S3ROCManager(S3Manager):
    """S3 aau wrapper that provides a convenient interface for working with ROC data.

    In summary, there are three types of ROC data that S3ROCManager handles
    - processed_data: raw ROC data from battery sensors containing ROC_VOLTAGE, FLOW, PRESSTURE_TH and their repective mask and corresponding timestamp.
    - weather_data: raw data from weather stations, with features including cloudcover, radiations, and corresponding timestamp.
    - labelled_data: csv data that is a combination of processed_data, weather_data, and manual labels stored in a yaml config file. 

    For convenience, file paths, prefixes, file_extension and related S3 keywords are stored in processed_data_dict and solar_data_dict. 

    Args:
        S3Manager (aau.S3Manager): inherits S3Manager from aau.S3Manager 
    """
    def __init__(self, info:dict):
        super().__init__(info)
        self.info = info 
        self.init_processed_data()
        self.init_solar_data()

        if not Path.config("nearest_station.yaml").exists():
            self.nearest_station = self.get_nearest_station()
        else:
            with open(Path.config("nearest_station.yaml"),'r') as file:
                self.nearest_station = yaml.safe_load(file)
        self.all_labelled_wells = list(self.nearest_station.keys())
        self.all_stations = list(set(self.nearest_station.values()))

    def init_label_data(self):
        with open(Path.config("well_labels.yaml"),'r') as file:
            self.label_dict = yaml.safe_load(file)

    def get_well_label_count(self):
        self.init_label_data()
        
        def get_label_count(x):
            value = list(x)
            count = {i:0 for i in range(10)}
            label, label_count = np.unique(value, return_counts=True)
            for i,l in enumerate(label):
                count[l] = label_count[i]
            return count 
        count_dict = {well:get_label_count(self.label_dict[well].values()) for well in self.label_dict}
        self.well_label_count = pd.DataFrame(count_dict).T
        return self.well_label_count

    @staticmethod
    def process_well_location(file:str='well_location.csv'):
        well_loc_df = pd.read_csv(Path.data(file))
        well_loc_df['COORD'] = well_loc_df.apply(lambda x: (x.LATITUDE, x.LONGITUDE), axis=1)
        well_loc_df.to_csv(Path.data(file),index=False)
        well_loc_df.set_index("WELL_CD", inplace=True)
        well_loc_df = well_loc_df.loc[:,['COORD']]
        return well_loc_df
    
    @staticmethod
    def process_station_location(file:str='station_location.csv'):
        station_loc_df = pd.read_csv(Path.data(file))
        if np.any(station_loc_df.Long.values < 0):
            rename_dict = {'Lat': "Long", "Long": "Lat"}
            station_loc_df.rename(columns=rename_dict, inplace=True)
        station_loc_df['COORD'] = station_loc_df.apply(lambda x: (x.Lat, x.Long), axis=1)
        station_loc_df.to_csv(Path.data(file),index=False)
        station_loc_df.set_index("Location", inplace=True)
        station_loc_df = station_loc_df.loc[:,["COORD"]]
        return station_loc_df

    def get_well_location(self, file:str="well_location.csv") -> pd.DataFrame :
        """Get well location dataframe from local storage and preprocess

        Args:
            file (str, optional): well location file name. Defaults to "well_location.csv".

        Returns:
            pd.DataFrame: dataframe whose index is WELL_CD and whose column is COORD
        """
        try:
            df = pd.read_csv(Path.data(file), index_col = "WELL_CD", usecols =["WELL_CD", "COORD"])
            df['COORD'] = df.eval(df['COORD'])
            return df
        except Exception as e:
            logger.error(f"Error encountered: {e}")
            return self.process_well_location(file)

    def get_station_location(self, file:str='station_location.csv') -> pd.DataFrame :
        """Get station location dataframe 

        Args:
            file (str, optional): station location file name. Defaults to 'station_location.csv'.

        Returns:
            pd.DataFrame: dataframe whose index is Location and whose column is COORD
        """
        try:
            df = pd.read_csv(Path.data(file), index_col='Location', usecols=['Location', 'COORD'])
            df['COORD'] = df.eval(df['COORD'])
            return df 
        except Exception as e:
            logger.error(f"Error encountered: {e}")
            return self.process_station_location(file)

    def get_distance_matrix(self, first: pd.DataFrame, second: pd.DataFrame, first_coord_column:str="COORD", second_coord_column:str="COORD") -> pd.DataFrame: 
        """Generate df of the distance between every pair of the first and second group.
        
        The final_df is of size (MxN) where M and N are the numbers of row and column of the first and second dataframe respectively. 
        The index and columns of final_df are the indices of the first and second dataframes.

        Args:
            first (pd.DataFrame): first group df, should have a coordinate column containing (lat,long) tuple, and index containing the well's name
            second (pd.DataFrame): second group df, should havea coordinate column containing (lat,long) tuple and index containing the well's name
            first_coord_column (str, optional): name of coordinate column in the first df. Defaults to "COORD".
            second_coord_column (str, optional): name of the coordinate column in the second df. Defaults to "COORD".

        Returns:
            pd.DataFrame: final_df containing the pair-wise distance between first and second 
        """
        second_grid, first_grid = np.meshgrid(second[second_coord_column].values, first[first_coord_column].values)
        f = np.vectorize(self.getDistance)
        distance_grid = f(first_grid, second_grid)
        final_df = pd.DataFrame(data=distance_grid, index=first.index, columns=second.index)
        return final_df
        
    @staticmethod
    def getDistance(x_coord: tuple, y_coord: tuple):
        return distance.geodesic(x_coord, y_coord).kilometers

    @staticmethod
    def _get_nearest_neighbor(distance_df:pd.DataFrame) -> dict:
        """Get df containing the nearest neighbors whose distances are specified in distance_df
        
        Args:
            distance_df (pd.DataFrame): distance_df[row, col] contains the distance in KM between the object described in row index and the object described in 
            col index.
        
        Returns: 
            dict: final_df containing the name of the nearest col for each row.
        """ 
        final_df = distance_df.idxmin(axis='columns')

        with open(Path.config("nearest_station.yaml"),'w') as file:
            yaml.dump(final_df.to_dict(), file)

        return final_df.to_dict()
    
    def get_nearest_station(self, 
                             well_location:str='well_location.csv', 
                             station_location:str='station_location.csv') -> dict:
        """Get nearest neighbor dictionary whose keys are wells and corresponding values the nearest stations.

        Args:
            well_location (str, optional): well_location csv containing lattitude and longitude of the wells. Defaults to 'well_location.csv'.
            station_location (str, optional): station_location csv containing lattitude and longitude of the stations. Defaults to 'station_location.csv'.

        Returns:
            dict: nearest station dictionary 
        """
        self.init_label_data()
        well_loc = self.get_well_location(well_location)
        station_loc = self.get_station_location(station_location)
        well_loc = well_loc.loc[self.all_labelled_wells]
        distance_matrix = self.get_distance_matrix(well_loc, station_loc)
        logger.info("Getting nearest well:station dict.")
        return self._get_nearest_neighbor(distance_matrix)

    def init_processed_data(self):
        self.processed_data_dict = {"path": "ROC/PROCESSED_DATA", 
                                    "file_prefix": "ROC_PROCESSED_DATA",
                                    "args_ts": "DTSMIN",
                                    "bucket": self.bucket,
                                    "file_ext": "csv"}
        if 'procdata_kwargs' in self.info:
            if 'args_ts' in self.info['procdata_kwargs']:
                self.processed_data_dict["args_ts"] = self.info['procdata_kwargs']["args_ts"]
            if 'path' in self.info['procdata_kwargs']:
                self.processed_data_dict["path"] = self.info['procdata_kwargs']["path"]
            if 'file_ext' in self.info['procdata_kwargs']:
                self.processed_data_dict["file_ext"] = self.info['procdata_kwargs']["ext"]
            if 'file_prefix' in self.info['procdata_kwargs']:
                self.processed_data_dict["file_ext"] = self.info['procdata_kwargs']["ext"]
            if 'bucket' in self.info['procdata_kwargs']:
                self.processed_data_dict["bucket"] = self.info['procdata_kwargs']["bucket"]
        if "DTSMIN" in self.processed_data_dict['args_ts'] and isinstance(self.processed_data_dict['args_ts'], list):
            self.processed_data_dict['args_ts'] = "DTSMIN"

        logger.info(f"Initialising processed data S3 path: {self.processed_data_dict['path']}")
        logger.info(f"Initialising processed data S3 file prefix: {self.processed_data_dict['file_prefix']}")
        logger.info(f"Initialising processed data S3 file time stamp: {self.processed_data_dict['args_ts']}")
        logger.info(f"Initialising processed data S3 file extension: {self.processed_data_dict['file_ext']}")

    def init_solar_data(self):
        self.solar_data_dict = {"path": "ROC/SOLAR_DATA", 
                                "file_prefix": "SOLAR_DATA",
                                "args_ts": "TS",
                                "bucket": self.bucket,
                                "file_ext": "csv"}
        if 'solardata_kwargs' in self.info:
            if 'args_ts' in self.info['solardata_kwargs']:
                self.solar_data_dict["args_ts"] = self.info['solardata_kwargs']["args_ts"]
            if 'path' in self.info['solardata_kwargs']:
                self.solar_data_dict["path"] = self.info['solardata_kwargs']["path"]
            if 'file_ext' in self.info['solardata_kwargs']:
                self.solar_data_dict["file_ext"] = self.info['solardata_kwargs']["ext"]
            if 'file_prefix' in self.info['solardata_kwargs']:
                self.solar_data_dict["file_ext"] = self.info['solardata_kwargs']["ext"]
            if 'bucket' in self.info['solardata_kwargs']:
                self.solar_data_dict["bucket"] = self.info['solardata_kwargs']["bucket"]

        logger.info(f"Initialising solar data S3 path: {self.solar_data_dict['path']}")
        logger.info(f"Initialising solar data S3 file prefix: {self.solar_data_dict['file_prefix']}")
        logger.info(f"Initialising solar data S3 file time stamp: {self.solar_data_dict['args_ts']}")
        logger.info(f"Initialising solar data S3 file extension: {self.solar_data_dict['file_ext']}")

    def read_solar(self,
                   station_code:str, 
                   start: datetime|str,
                   end: datetime|str,
                   strp_format:str='%Y-%m-%d',
                   strf_format:str='%Y%m%d',
                   to_csv:bool=False) -> pd.DataFrame:
        """Read combined solar data from S3 database 

        Args:
            station_code (str): station code
            start (datetime | str): start date
            end (datetime | str): end date
            strp_format (str, optional): interpretation format for start and end. Defaults to '%Y-%m-%d'.
            strf_format (str, optional): S3 file storage date format. Defaults to '%Y%m%d'.
            to_csv (bool, optional): whether to save solar data to dedicated local directory. Defaults to False.

        Returns:
            pd.DataFrame: combined solar data
        """
        
        logger.info(f"Read solar file from database for station: {station_code} from {start} to {end}")
        alldf =  self.read_from_storage(item_cd=station_code, start=start, end = end,
                                      strp_format=strp_format, strf_format=strf_format,
                                      **self.solar_data_dict)
            
        #Apply solar preprocessing - remove duplicates and make continuous time index
        TS = self.solar_data_dict['args_ts']
        alldf[TS]=pd.to_datetime(alldf[TS])
        alldf.set_index(TS,inplace=True)

        date_range = pd.date_range(alldf.index.min(), alldf.index.max(), freq='H')
        alldf = alldf.groupby(alldf.index).mean()
        alldf = alldf.reindex(date_range)
        alldf.index.name = "TS"
        alldf.reset_index(inplace=True)
        if to_csv:
            alldf.to_csv(Path.data(f"{station_code}_{start}_{end}_weather.csv"), index=False)
            logger.info(f"Save solar data to {station_code}_{start}_{end}_weather.csv")
        return alldf
        
    def read_processed_data(self,
                   well_code:str, 
                   start: datetime|str,
                   end: datetime|str,
                   strp_format:str='%Y-%m-%d',
                   strf_format:str='%Y%m%d',
                   nan_replace_method:str='interpolate',
                   to_csv:bool=False) -> pd.DataFrame:
        """Read combined sensor data from S3 database 

        Args:
            well_code (str): well code
            start (datetime | str): start date
            end (datetime | str): end date
            strp_format (str, optional): interpretation format for start and end. Defaults to '%Y-%m-%d'.
            strf_format (str, optional): S3 file storage date format. Defaults to '%Y%m%d'.
            nan_replace_method (str, optional): method to replace nan values. One of 'zero' or 'interpolate'. Defaults to 'interpolate'.
            to_csv (bool, optional): whether to save solar data to dedicated local directory. Defaults to False.

        Returns:
            pd.DataFrame: combined raw data
        """

        logger.info(f"Read well data from database for well: {well_code} from {start} to {end}")
        alldf =  self.read_from_storage(item_cd=well_code, start=start, end = end,
                                      strp_format=strp_format, strf_format=strf_format,
                                      **self.processed_data_dict)
        
        #Processed data preprocessing - remove sub minute duplicates 
        #Pad data to form continuous time sequence
        #Create Nan Mask, and replace nan 
        TS = self.processed_data_dict['args_ts']
        alldf[TS]=pd.to_datetime(alldf[TS])
        alldf.set_index(TS,inplace=True)
        alldf = alldf.loc[:,['ROC_VOLTAGE','FLOW','PRESSURE_TH']]

        date_range = pd.date_range(alldf.index.min(), alldf.index.max(), freq='T')
        alldf = alldf.groupby(alldf.index).mean()
        alldf = alldf.reindex(date_range)
        alldf['Mask_ROC_VOLTAGE']=1-alldf.ROC_VOLTAGE.isna()
        alldf['Mask_FLOW']=1-alldf.FLOW.isna()
        alldf['Mask_PRESSURE_TH']=1-alldf.PRESSURE_TH.isna()

        if nan_replace_method == 'zero':
            alldf.fillna(0, inplace=True)
        elif nan_replace_method == 'interpolate':
            alldf.interpolate(method='linear', inplace=True, limit_direction='both')
        else:
            logger.error("Invalid nan_replace_method. Accepts either zero or interpolate.")
        alldf.index.name = "TS"
        alldf.reset_index(inplace=True)
        if to_csv:
            alldf.to_csv(Path.data(f"{well_code}_{start}_{end}_raw.csv"),index=False)
            logger.info(f"Save well data to {well_code}_{start}_{end}_raw.csv")
        return alldf
    
    def list_weather_stations(self) -> List:
        """List all availabel weather stations from files on S3 storage

        Returns:
            List: list of all stations 
        """
        all_data = self.list_files(self.solar_data_dict['path'])
        unique_stations = {x.split(self.solar_data_dict['path'])[1].split('_'+self.solar_data_dict['file_prefix'])[0] for x in all_data}
        return unique_stations

    def list_all_wells(self) -> List:
        """List all wells whose csvs are available on the S3 storage 

        Returns:
            List: list of wells
        """
        all_data = self.list_files(self.processed_data_dict['path'])
        unique_stations = {x.split(self.processed_data_dict['path'])[1].split('_'+self.processed_data_dict['file_prefix'])[0] for x in all_data}
        return unique_stations

    def read_labelled_data(self,
                   well_code:str, 
                   start: datetime|str,
                   end: datetime|str,
                   strp_format:str='%Y-%m-%d',
                   strf_format:str='%Y%m%d',
                   nan_replace_method:str='interpolate',
                   raw_csv:str|pd.DataFrame=None,
                   weather_csv:str|pd.DataFrame=None,
                   label_config:str='well_labels.yaml',
                   window_size:int=6,
                   to_pickle:bool=False) -> pd.DataFrame:
        """Read and combine raw sensor, raw weather data, and labels.

        Raw data are combined in a window -i.e. if window_size = 6,
        for each TS, a ROC_VOLTAGE feature is a sequence of size 1440*7, containing 
        ROC_VOLTAGE values from 6 previous days and the value of the current TS. 

        Args:
            well_code (str): well_code
            start (datetime | str): start date
            end (datetime | str): end_date
            strp_format (str, optional): string format for start and end. Defaults to '%Y-%m-%d'.
            strf_format (str, optional): string format for files on S3. Defaults to '%Y%m%d'.
            nan_replace_method (str, optional): method to replace nan in raw data. One of 'interpolate','zero'. Defaults to 'interpolate'.
            raw_csv (str | pd.DataFrame, optional): raw_df. If provided, will read raw_df from local storage. If None, will use read_processed_data to read from S3. Defaults to None.
            weather_csv (str | pd.DataFrame, optional): weather df. If provided, will read weather_df from local storage. If None, will use read_solar to read from S3. Defaults to None.
            label_config (str, optional): yaml file containing file labels. Defaults to 'well_labels.yaml'.
            window_size (int, optional): window sequence length - determines the number of previous day to pool in forming feature. Defaults to 6.
            to_pickle (bool, optional): whether to save data locally. Defaults to False.

        Returns:
            pd.DataFrame: labelled_df
        """

        logger.info(f"Calling read_labelled_data method for well: {well_code} from {start} to {end}")
        #Prepare raw data df
        if raw_csv is None:
            raw_df = self.read_processed_data(well_code, start, end, strp_format, strf_format, nan_replace_method)
            raw_df.set_index("TS",inplace=True)
        else:
            if isinstance(raw_csv, str):
                raw_df = pd.read_csv(Path.data(raw_csv), index_col="TS", parse_dates=["TS"])
            if isinstance(raw_csv, pd.DataFrame):
                raw_df = raw_csv
                if "TS" in raw_df.columns():
                    raw_df.TS = pd.to_datetime(raw_df.TS)
                    raw_df.set_index("TS",inplace=True)
        daily_raw_df = raw_df.groupby(raw_df.index.date).agg(list)
        daily_raw_df.index = pd.to_datetime(daily_raw_df.index)

        #Prepare weather data df
        if weather_csv is None:
            weather_df = self.read_solar(self.nearest_station[well_code], start, end, strp_format, strf_format)
            weather_df.set_index("TS",inplace=True)
        else:
            if isinstance(weather_csv, str):
                weather_df = pd.read_csv(Path.data(weather_csv), index_col="TS", parse_dates=["TS"])
            if isinstance(weather_csv, pd.DataFrame):
                weather_df = weather_csv
                if "TS" in weather_df.columns():
                    weather_df.TS = pd.to_datetime(weather_df.TS)
                    weather_df.set_index("TS",inplace=True)
        daily_weather_df = weather_df.groupby(weather_df.index.date).agg(list)
        daily_weather_df.index = pd.to_datetime(daily_weather_df.index)

        #Prepare labels from label_config
        with open(Path.config(label_config),'r') as file:
            label = yaml.safe_load(file)[well_code]
        
        label_df = pd.DataFrame({"TS": label.keys(), "labels": label.values()})
        label_df.set_index("TS",inplace=True)

        #Closure to combine raw data and labels
        def process(raw_df, weather_df, index, label, new_df, max_date = 4):
            days = pd.date_range(index-timedelta(days=max_date),index)
            try:
                raw_window = raw_df.loc[days,:]
                weather_window = weather_df.loc[days,:]
            except Exception as e:
                return 
            new_df.loc[index,'labels']=label
            drop_index = False 

            for col in raw_df.columns:
                column_val = np.hstack(raw_window[col].values).astype(object)
                if len(column_val) < 1440 * (max_date+1): 
                    logger.warning(f"Issue processing well {well_code}, time-index: {index} incomplete feature length. Feature: {col}, size: {len(column_val)}")
                    drop_index=True
                new_df.loc[index,col]=column_val

            for col in weather_df.columns:
                column_val = np.hstack(weather_window[col].values).astype(object)
                if len(column_val) < 24 * (max_date+1): 
                    logger.warning(f"Issue processing well {well_code}, time-index: {index} incomplete feature length. Feature: {col}, size: {len(new_df.loc[index,col])}")
                    drop_index=True
                new_df.loc[index,col]=column_val
            if drop_index:
                new_df.drop(index, inplace=True)
        #Combine raw data and label by iterating through each label
        logger.info(f"Combine raw data and labels for well: {well_code} from {start} to {end}")
        cols = np.concatenate([raw_df.columns.values, label_df.columns.values, weather_df.columns.values])
        new_df = pd.DataFrame(columns = cols)
        for index in label_df.index:
            label = label_df.loc[index,'labels']
            process(daily_raw_df, daily_weather_df, index, label, new_df, window_size)
        
        if to_pickle:
            new_df.to_pickle(Path.data(f"{well_code}_{start}_{end}_labelled.pkl"))
            logger.info(f"Save labelled data to {well_code}_{start}_{end}_labelled.pkl")
        return new_df

    def classify_voltage_type(self):
        voltage = {}
        for well in self.all_labelled_wells:
            df = pd.read_csv(Path.data(f"{well}_2016-01-01_2023-01-01_raw.csv"))
            v = df.ROC_VOLTAGE[df.Mask_ROC_VOLTAGE==1].mean()

            if v >= 20:
                voltage[well]=24
            else:
                voltage[well]=12
        with open(Path.config("well_type.yaml"),'w') as file:
            yaml.dump(voltage, file, default_flow_style=False)
        return voltage

    def calculate_weather_transform_params(self):
        weather = {}
        all_df = []
        for station in self.all_stations:
            weather[station]={feature:{} for feature in weather_features}
            df = pd.read_csv(Path.data(f"{station}_2016-01-01_2023-01-01_weather.csv"), usecols=weather_features)
            all_df.append(df)
            for feature in df.columns:
                scaler = StandardScaler()
                scaler.fit(df[feature].values.reshape(-1,1))
                weather[station][feature]['mean']=float(scaler.mean_[0])
                weather[station][feature]['var']=float(scaler.var_[0])
                weather[station][feature]['scale']=float(scaler.scale_[0])
        all_df = pd.concat(all_df,axis=0)
        all_weather = {feature:{} for feature in weather_features}
        for feature in all_df.columns:
            scaler = StandardScaler()
            scaler.fit(all_df[feature].values.reshape(-1,1))
            all_weather[feature]['mean']=float(scaler.mean_[0])
            all_weather[feature]['var']=float(scaler.var_[0])
            all_weather[feature]['scale']=float(scaler.scale_[0])
        weather_param_dict = {'one': weather, 'all': all_weather}
        with open(Path.config("station_transform_params.yaml"),'w') as file:
            yaml.dump(weather_param_dict, file, default_flow_style=False)
        return weather_param_dict

    def calculate_well_transform_params(self):
        well_params = {}
        all_df = {12:[], 24: []}
        all_well_params = {12:{}, 24: {}}

        if not Path.config("well_type.yaml").exists():
            voltage_type = self.classify_voltage_type()
        else:
            with open(Path.config("well_type.yaml"),'r') as file:
                voltage_type = yaml.safe_load(file)

        for well in self.all_labelled_wells:
            well_params[well]={feature:{} for feature in well_features}
            df = pd.read_csv(Path.data(f"{well}_2016-01-01_2023-01-01_raw.csv"))
            voltage = voltage_type[well]
            if voltage == 12:
                all_df[12].append(df)
            else:
                all_df[24].append(df)
            for feature in well_features:
                mask = df["Mask_"+feature] == 1
                scaler = StandardScaler()
                scaler.fit(df[feature][mask].values.reshape(-1,1))
                well_params[well][feature]['mean']=float(scaler.mean_[0])
                well_params[well][feature]['var']=float(scaler.var_[0])
                well_params[well][feature]['scale']=float(scaler.scale_[0])

        for v in [12,24]:
            all_df_ = pd.concat(all_df[v],axis=0)
            all_well_params[v] = {feature: {} for feature in well_features}
            for feature in well_features:
                mask = all_df_["Mask_"+feature] == 1
                scaler = StandardScaler()
                scaler.fit(all_df_[feature][mask].values.reshape(-1,1))
                all_well_params[v][feature]['mean']=float(scaler.mean_[0])
                all_well_params[v][feature]['var']=float(scaler.var_[0])
                all_well_params[v][feature]['scale']=float(scaler.scale_[0])

        well_param_dict = {'one': well_params, 'all': all_well_params}
        with open(Path.config("well_transform_params.yaml"),'w') as file:
            yaml.dump(well_param_dict, file, default_flow_style=False)
        return well_param_dict, all_df