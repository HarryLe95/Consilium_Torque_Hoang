from src.utils.PathManager import Paths as Path 
from src.utils.Data import get_combined_data, get_random_split_from_image_label
from typing import Sequence 
import yaml 
import tensorflow as tf 
import numpy as np
import logging 
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
 
weather_features = ['temperature_2m', 'cloudcover',
       'cloudcover_low', 'cloudcover_mid', 'cloudcover_high',
       'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
       'direct_normal_irradiance']

well_features = ['ROC_VOLTAGE', 'FLOW', "PRESSURE_TH"]


class ROC_Generator:
    """
    ROCGenerator class - provides dataset interface for model training and validation from stored csvs

    Method:
        self.setup: load data from csv and apply downstream transformations. Only call when neeeded as this might be computationally expensive.
    Attributes:
        self.dataset(tf.data.Dataset|tuple[tf.data.Dataset, tf.data.Dataset]): one dataset or a training/validation dataset pair.
    """
    with open(Path.config("well_transform_params.yaml"),'r') as file:
        well_params_config = yaml.safe_load(file)

    with open(Path.config("well_type.yaml"),'r') as file:
        well_type_config = yaml.safe_load(file)

    with open(Path.config("station_transform_params.yaml"),'r') as file:
        station_params_config = yaml.safe_load(file)

    with open(Path.config("nearest_station.yaml"),'r') as file:
        nearest_station_config = yaml.safe_load(file)
    
    def __init__(self, 
                 wells: str|Sequence[str], 
                 features: Sequence[str] = ["ROC_VOLTAGE"],
                 last_day:str=None,
                 num_days:int=7,
                 normalise_mode:str='all',
                 label_mapping:dict={1:0,2:1,3:0,4:0,5:1},
                 drop_labels:Sequence[int]=[0,6,7,8,9],
                 split:bool=False,
                 split_ratio:float=0.8,
                 num_classes:int=2):
        """Model initialisation

        Args:
            wells (str | Sequence[str]): wells used to form dataset
            features (Sequence[str], optional): features to be included. Defaults to ["ROC_VOLTAGE"].
            num_days (int, optional): number of days to combine to form one data instance. Defaults to 7.
            normalise_mode (str, optional): one of ['all','one']. If 'one', normalise each well individually; if 'all', normalise wells by their Voltage Group. Defaults to 'all'.
            label_mapping (_type_, optional): labels remapping rules. Defaults to {1:0,2:1,3:0,4:0,5:1}.
            drop_labels (Sequence[int], optional): labels to drop. Defaults to [0,6,7,8,9].
            split (bool, optional): whether to split the dataset to training and validation. Defaults to False.
            split_ratio (float, optional): training fraction. Defaults to 0.8.
            num_classes (int, optional): number of classes in the data. Defauls to 0.
        """
        self.wells = wells
        self.features = features 
        self.well_features = [f for f in self.features if f in well_features]
        self.weather_features = [f for f in self.features if f in weather_features]
        if len(self.weather_features)==0:
            self.weather_features=None
        self.num_days = num_days 
        self.normalise_mode = normalise_mode 
        self.label_mapping = label_mapping 
        self.drop_labels = drop_labels 
        self.split = split 
        self.split_ratio = split_ratio
        self.num_classes = num_classes
        self.last_day=last_day

    @staticmethod
    def get_scaler_from_config(mean: float|Sequence[float],var: float|Sequence[float],scale: float|Sequence[float]) -> StandardScaler:
        """Create a Standard Scaler object from known data statistics

        Args:
            mean (float | Sequence[float]): data mean
            var (float | Sequence[float]):  data variance
            scale (float | Sequence[float]):data scale

        Returns:
            StandardScaler: scaler object 
        """
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.var_ = var
        scaler.scale_= scale
        return scaler

    def get_scaler(self, mode: str, well: str, well_features:Sequence[str], weather_features:Sequence[str]) -> dict[StandardScaler]:
        """Get a dictionary of StandardScaler objects that correspond to statistics specified in config 

        Args:
            mode (str): either one or all. If one is selected, a scaler is specified for every individual well. If all is selected, a scaler is specified for each well group (12V or 24V)
            well(str): well name
            well_features (Sequence[str]): well features to be scaled.
            weather_features (Sequence[str]): weather features to be scaled.
        Returns:
            dict[StandardScaler]: dictionary of scaler for each feature of each well/well_group 
        """
        well_params = self.well_params_config[mode][well] if mode == 'one' else \
                      self.well_params_config[mode][self.well_type_config[well]]
        scaler_dict = {feature: self.get_scaler_from_config(well_params[feature]['mean'], 
                                                       well_params[feature]['var'], 
                                                       well_params[feature]['scale']) 
                       for feature in well_features if "Mask" not in feature}
        station = self.nearest_station_config[well]
        station_params = self.station_params_config[mode][station] if mode == 'one' else \
                         self.station_params_config[mode]
        scaler_dict.update({feature: self.get_scaler_from_config(station_params[feature]['mean'], 
                                                            station_params[feature]['var'], 
                                                            station_params[feature]['scale']) 
                            for feature in weather_features if "Mask" not in feature})
        return scaler_dict

    def _get_scaler(self):
        scaler = []
        for well in self.wells:
            scaler.append(self.get_scaler(self.normalise_mode, well, self.well_features, self.weather_features))
        return scaler 
    
    def setup(self):
        self.scaler = self._get_scaler()
        image_well, image_weather, label, TS = get_combined_data(well_name = self.wells, 
                                                                 well_features = self.well_features, 
                                                                 weather_features = self.weather_features,
                                                                 num_days =self.num_days, 
                                                                 scaler=self.scaler, 
                                                                 label_mapping=self.label_mapping, 
                                                                 drop_labels=self.drop_labels,
                                                                 last_day=self.last_day)
        label = tf.keras.utils.to_categorical(label,self.num_classes)

        if self.split:
            split_result = get_random_split_from_image_label(image_well=image_well, 
                                                             image_weather=image_weather,
                                                             label=label, 
                                                             TS=TS, 
                                                             train_ratio=self.split_ratio)
            train_image_well, train_image_weather, train_label, train_TS, \
            val_image_well, val_image_weather, val_label, val_TS = split_result
            self.train_image = [train_image_well, train_image_weather] if self.weather_features is not None else \
                               [train_image_well]
            self.train_label = train_label
            self.val_image   = [val_image_well, val_image_weather] if self.weather_features is not None else \
                               [val_image_well]
            self.val_label   = val_label
            self.TS = [train_TS, val_TS]
        else:
            self.image = [image_well, image_weather] if self.weather_features is not None else \
                         [image_well]
            self.label = label
            self.TS = TS
        
        logger.debug(f"Prepared dataset for wells: {self.wells} with split: {self.split}")
