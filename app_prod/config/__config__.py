"""
Created on Tue Aug 30 2022
@author: Steve Lechowicz
"""
from utils.PathManager import PathManager

def init():
    base = {
        '_type' : 'file',
        '_info': {
            'connection_type' : 'file',
            'path' : PathManager.CONFIG_PATH,
            'file' : 'config.csv',
            'partition_mode' : None
        },
        '_sql': None,
        '_kwargs': {
        }
    }
    if base['_type'] == 'py':
        import imp
        config = imp.load_source(base['_name'], base['_path']).init()
    elif base['_type'] == 'file':
        import utils.advancedanalytics_util as aau
        sql = base['_sql']
        kwargs = base['_kwargs']
        config = aau.aauconnect_(base['_info']).read(sql=sql, args={}, edit=[], orient='config', do_raise=True, **kwargs)
    elif base['_type'] == 'ora':
        import utils.advancedanalytics_util as aau
        sql = base['_sql']
        kwargs = base['_kwargs']
        config = aau.aauconnect_(base['_info']).read(sql=sql, args={}, edit=[], orient='config', do_raise=True, **kwargs)
    return config
