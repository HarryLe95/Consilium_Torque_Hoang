try:
    import src.aau.advancedanalytics_util as aau
except:
    import advancedanalytics_util as aau
"""
Created on Tue Aug 30 2022
@author: Steve Lechowicz
"""
def init():
    base = {
        '_type' : 'file',
        '_info': {
            'connection_type' : 'file',
            'path' : 'C:/Sandbox/config',
            'file' : 'config.csv'
        },
        '_sql': None,
        '_kwargs': {
        }
    }
    if base['_type'] == 'py':
        import imp
        config = imp.load_source(base['_name'], base['_path']).init()
    elif base['_type'] == 'file':
        sql = base['_sql']
        kwargs = base['_kwargs']
        config = aau.aauconnect_(base['_info']).read(sql=sql, args={}, edit=[], orient='config', do_raise=True, **kwargs)
    elif base['_type'] == 'ora':
        sql = base['_sql']
        kwargs = base['_kwargs']
        config = aau.aauconnect_(base['_info']).read(sql=sql, args={}, edit=[], orient='config', do_raise=True, **kwargs)
    return config
