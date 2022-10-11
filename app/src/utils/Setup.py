import yaml 
import logging 
import argparse
import datetime 
from src.utils.PathManager import Paths as Path

def logger_init(args: argparse.Namespace) -> None:
    """Set up log folder for saving model's training results and other data 
    Args:
        args (argparse.Namespace) - dictionary containing namespace 
    """ 
    if args.save_name is None: 
        args.save_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    logfolder = Path.model(args.save_name)
    logfolder.mkdir(parents=True, exist_ok=True)
    logfile = logfolder / 'log.txt'
    
    logging.basicConfig(filename=logfile,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    
    logging.debug("Save configs parameters")
    config_file = Path.model("config.yaml", args.save_name)
    with open(config_file, 'w+') as file:
        yaml.dump([vars(args)], file, default_flow_style=False)
    
def get_argparse(descriptor:str="ROC_Classification Model") -> argparse.Namespace:
    """Get argparse object from a meta file

    Args:
        descriptor (str, optional): parser description. Defaults to "ROC_Classification Model".

    Returns:
        argparse.Namespace: argparse object
    """
    parser=argparse.ArgumentParser(description=descriptor)
    parser.add_argument("-meta_config", type=str, default="meta_train_config.yaml", help="Name of meta_config file in the config directory. Name should be filename.yaml and does not include path.")
    args = parser.parse_args()
    with open(Path.config(args.meta_config),'r') as file:
        _config = yaml.safe_load(file)[0]
    vars(args).update(_config)
    return args