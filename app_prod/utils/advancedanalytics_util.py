# -*- coding: utf-8 -*-
"""
Created: 2022-11-09
@author: Steve Lechowicz

Santos Advanced Analytics 
SandboxProductionLoop data access utility

NOTE: Santos standards mandate NO HARD-CODING of credentials in any source code.
"""

# Compatibility
from __future__ import print_function

# Imports
from abc import ABC, abstractmethod
import os
import io
import ast
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas import Timestamp
import pytz
from pytz import timezone
import dateutil.parser
from random import randint
import inspect
from contextlib import contextmanager
from pandas.io.sql import to_sql, read_sql, execute
from sqlalchemy import create_engine
import re
from warnings import catch_warnings, filterwarnings
from sqlalchemy.exc import DatabaseError, ResourceClosedError
from sqlalchemy.pool import NullPool
from sqlalchemy import text
try:
    import utils.advancedanalytics_access as aa
except Exception:
    pass
try:
    import cx_Oracle
except Exception:
    pass
try:
    import pymssql
except Exception:
    pass
try:
    from p2 import P2ServerClient
except Exception:
    pass
try:
    import snowflake.connector as snow
except Exception:
    pass

class AAPandaSQLException(Exception):
    pass

class AAPandaSQL:
    def __init__(self, db_uri='sqlite:///:memory:', persist=False):
        """
        Initialize with a specific database.
        :param db_uri: SQLAlchemy-compatible database URI.
        :param persist: keep tables in database between different calls on the same object of this class.
        """
        self.engine = create_engine(db_uri, poolclass=NullPool)
        if self.engine.name not in ('sqlite', 'postgresql'):
            raise AAPandaSQLException('Currently only sqlite and postgresql are supported.')
        self.persist = persist
        self.loaded_tables = set()
        if self.persist:
            self._conn = self.engine.connect()
            self._init_connection(self._conn)

    def __call__(self, query, env=None):
        """
        Execute the SQL query.
        Automatically creates tables mentioned in the query from dataframes before executing.
        :param query: SQL query string, which can reference pandas dataframes as SQL tables.
        :param env: Variables environment - a dict mapping table names to pandas dataframes.
        If not specified use local and global variables of the caller.
        :return: Pandas dataframe with the result of the SQL query.
        """
        if env is None:
            env = get_outer_frame_variables()
        result = None
        with self.conn as conn:
            rtbl, wtbl = extract_table_names(query)
            if len(rtbl) > 0:
                for table_name in rtbl:
                    if table_name not in env:
                        # don't raise error because the table may be already in the database
                        continue
                    if self.persist and table_name in self.loaded_tables:
                        # table was loaded before using the same instance, don't do it again
                        continue
                    self.loaded_tables.add(table_name)
                    write_table(env[table_name], table_name, conn)
                try:
                    query = text(query)
                    prms = None
                    if 'sql_params' in env:
                        prms = env['sql_params']
                    result = read_sql(query, conn, params=prms)
                except DatabaseError as ex:
                    raise AAPandaSQLException(ex)
                except ResourceClosedError:
                    # query returns nothing
                    result = None
            if len(wtbl) > 0:
                for table_name in wtbl:
                    if table_name not in env:
                        # don't raise error because the table may be already in the database
                        continue
                    if self.persist and table_name in self.loaded_tables:
                        # table was loaded before using the same instance, don't do it again
                        continue
                    self.loaded_tables.add(table_name)
                    write_table(env[table_name], table_name, conn)
                prms = env['dst_data'].to_dict(orient='records')
                insert_sql(query, conn, args=prms)
                result = read_sql('SELECT * FROM {}'.format(table_name), conn)
        return result

    @property
    @contextmanager
    def conn(self):
        if self.persist:
            # the connection is created in __init__, so just return it
            yield self._conn
            # no cleanup needed
        else:
            # create the connection
            conn = self.engine.connect()
            conn.text_factory = str
            self._init_connection(conn)
            try:
                yield conn
            finally:
                # cleanup - close connection on exit
                conn.close()

    def _init_connection(self, conn):
        if self.engine.name == 'postgresql':
            conn.execute('set search_path to pg_temp')

def get_outer_frame_variables():
    """ Get a dict of local and global variables of the first outer frame from another file. """
    cur_filename = inspect.getframeinfo(inspect.currentframe()).filename
    outer_frame = next(f
                       for f in inspect.getouterframes(inspect.currentframe())
                       if f.filename != cur_filename)
    variables = {}
    variables.update(outer_frame.frame.f_globals)
    variables.update(outer_frame.frame.f_locals)
    return variables

def extract_table_names(query):
    """ Extract table names from an SQL query. """
    tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    rtables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    tables_blocks = re.findall(r'(?:INTO)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    wtables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return set(rtables), set(wtables)

def write_table(df, tablename, conn):
    """ Write a dataframe to the database. """
    with catch_warnings():
        filterwarnings('ignore',
                       message='The provided table name \'%s\' is not found exactly as such in the database' % tablename)
        to_sql(df, name=tablename, con=conn,
               index=not any(name is None for name in df.index.names))  # load index into db if all levels are named

def insert_sql(sql, conn, args=None):
    for v in args:
        for k in v:
            v[k] = str(v[k])
    tsql = text(sql)
    execute(tsql, con=conn, params=args)

def aasqldf(query, env=None, db_uri='sqlite:///:memory:'):
    """
    Query pandas data frames using sql syntax

    Parameters
    ----------
    query: string
        a sql query using DataFrames as tables
    env: locals() or globals()
        variable environment; locals() or globals() in your function
        allows sqldf to access the variables in your python environment
    db_uri: string
        SQLAlchemy-compatible database URI

    Returns
    -------
    result: DataFrame
        returns a DataFrame with your query's result

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
        "x": range(100),
        "y": range(100)
    })
    >>> from pandasql import sqldf
    >>> sqldf("select * from df;", globals())
    >>> sqldf("select * from df;", locals())
    >>> sqldf("select avg(x) from df;", locals())
    """
    return AAPandaSQL(db_uri)(query, env)

def aauconnect_(info, connection_type=None):
    if connection_type is None:
        connection_type = info['connection_type']
    # creates a connection to a data source / dest via the aau utility functions
    # connection type can be modified in configuration without changing code
    if connection_type == 's3':
        con = S3(info)
    elif connection_type == 'p2':
        con = P2(info)
    elif connection_type == 'file':
        con = File(info)
    elif connection_type == 'rts':
        con = RTS(info)
    elif connection_type == 'ora':
        con = Oracle(info)
    elif connection_type == 'sql':
        con = SQLServer(info)
    elif connection_type == 'sf':
        con = Snowflake(info)
    return con

class AAUConnection(ABC):
    def __init__(self, info):
        super().__init__()
        self.info = info
        self.client = None
        _ = self._connect_()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.client is not None:
            try:
                self.client.close()
                self.client = None
            except Exception:
                pass

    def __repr__(self):
        return 'advancedanalytics_util.Connection(%r)' % (self.info)

    def __str__(self):
        return 'AAUConnection: [INFO=%s]' % (self.info)

    def _error_handler_(self, error, do_raise):
        if do_raise:
            raise error
        else:
            print(error)

    @abstractmethod
    def _connect_(self, do_raise=True):
        pass

    def reconnect(self):
        return self._connect_()

    @abstractmethod
    def read(self, sql=None, args={}, edit=[], orient='list', do_raise=True, **kwargs):
        pass

    @abstractmethod
    def write(self, sql=None, args=[], edit=[], do_raise=True, **kwargs):
        pass

    @abstractmethod
    def write_many(self, sql, args=[], edit=[], do_raise=True, **kwargs):
        pass

    def get_filestartend(self, data_start, data_end, **kwargs):
        if 'partition_mode' in kwargs:
            pm = kwargs['partition_mode']
        else:
            pm = self.info['partition_mode']
        if pm in ['day', 'month', 'year', 'alltime']:
            y = data_start.year
            m = data_start.month
            d = data_start.day
            fstart = datetime(y, m, d)
            if pm == 'day':
                fend = fstart + timedelta(days=1)
            elif pm == 'month':
                fstart = datetime(y, m, 1)
                fend = fstart + timedelta(days=32)
                fend = datetime(fend.year, fend.month, 1)
            elif pm == 'year':
                fstart = datetime(y, 1, 1)
                fend = datetime(fstart.year+1, 1, 1)
            elif pm == 'alltime':
                fend = datetime(data_end.year, data_end.month, data_end.day) + timedelta(days=1)
            else:
                raise ValueError('Invalid partition mode')
                return
        elif pm is None:
            return None, None
        else:
            raise ValueError('Invalid partition mode')
            return
        return fstart, fend

    def get_filename(self, data_start, data_end, **kwargs):
        if 'partition_mode' in kwargs:
            pm = kwargs['partition_mode']
        else:
            pm = self.info['partition_mode']
        if 'file_prefix' in kwargs:
            fp = kwargs['file_prefix']
        else:
            fp = self.info['file_prefix']
        if 'file_suffix' in kwargs:
            fs = kwargs['file_suffix']
        else:
            fs = self.info['file_suffix']
        if pm in ['day', 'month', 'year', 'alltime']:
            fn = '{}{}_{}{}'.format(fp, data_start.strftime('%Y%m%d'), data_end.strftime('%Y%m%d'), fs)
        elif pm is None:
            return None
        else:
            raise ValueError('Invalid partition mode')
            return
        return fn

    def orient_and_parse(self, df, orient, timezone, **kwargs):
        args_ts = []
        if 'args_ts' in kwargs:
            args_ts = kwargs['args_ts']
        elif 'args_ts' in self.info:
            args_ts = self.info['args_ts']
        for c in df.columns:
            try:
                df[c] = df[c].replace({'None' : None})
                df[c] = df[c].replace({'nan' : None})
                df[c] = df[c].replace({'NaT' : None})
            except TypeError:
                pass
        if orient == 'records':
            recs = df.to_dict(orient=orient)
            if len(args_ts) > 0:
                for i, r in enumerate(recs):
                    for c in args_ts:
                        if c not in r:
                            continue
                        if pd.isnull(r[c]) is False and type(r[c]) not in [pd.Timestamp, datetime]:
                            if timezone is not None:
                                recs[i][c] = dateutil.parser.parse(r[c]).replace(tzinfo=timezone)
                            else:
                                recs[i][c] = dateutil.parser.parse(r[c])
            return recs
        elif orient == 'list':
            recs = df.to_dict(orient=orient)
            for c in args_ts:
                if c not in recs:
                    continue
                if timezone is not None:
                    recs[c] = [dateutil.parser.parse(rv).replace(tzinfo=timezone) if pd.isnull(rv) is False else rv for rv in recs[c]]
                else:
                    recs[c] = [dateutil.parser.parse(rv) if pd.isnull(rv) is False else rv for rv in recs[c]]
            return recs
        elif orient == 'config':
            config = {}
            recs = df.to_dict(orient='records')
            for rec in recs:
                key = rec['KEY']
                val = rec['VALUE']
                if val is None:
                    config[key] = None
                    continue
                val = str(val)
                vtype = rec['TYPE']
                if vtype == 'range':
                    r = []
                    for a, b in re.findall(r'(\d+)-?(\d*)', val):
                        r.extend(range(int(a), int(a)+1 if b=='' else int(b)+1))
                    if len(r) > 0:
                        config[key] = r[:-1]
                elif vtype == 'dict':
                    config[key] = ast.literal_eval(val)
                elif vtype == 'datetime':
                    config[key] = dateutil.parser.parse(val)
                elif vtype == 'str':
                    config[key] = val
                elif vtype == 'list':
                    config[key] = ast.literal_eval(val)
                elif vtype == 'float':
                    config[key] = float(val)
                elif vtype == 'int':
                    config[key] = int(val)
                elif vtype == 'bool':
                    config[key] = ast.literal_eval(val)
                else:
                    raise ValueError('Invalid config type')
                    return None
            return config
        else:
            for c in args_ts:
                if c not in df.columns:
                    continue
                if timezone is not None:
                    df[c] = pd.to_datetime(df[c])
                    df[c] = df[c].dt.tz_localize('UTC').dt.tz_convert(timezone)
                else:
                    df[c] = pd.to_datetime(df[c])
            return df

# P2 Connection Subclass
class P2Connection(AAUConnection):
    """
    P2 Wrapper
    Usage: 
        p2 = P2Connection()
            ...
    @param info  <dict>: P2 Server url

    """

    def __repr__(self):
        return 'advancedanalytics_util.P2Connection(%r)' % (self.info)

    def __str__(self):
        return 'P2Connection: [INFO=%s]' % (self.info)

    def _connect_(self, do_raise=True):
        """
        Connect to P2 server
        """
        try:
            self.client = P2ServerClient(self.info['p2server'], verify_cert=False)
            return True
        except Exception as e:
            self._error_handler_(e, do_raise)
        return False

    def read(self, sql=None, args={}, edit=[], orient='list', do_raise=True, **kwargs):
        """
        Execute query parsing return.
        @param sql       <str>: Query to execute (via SQLite - for compatibility with database connection classes). 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <dict>: Not used - for consistency with Oracle connection class
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param orient    <str>: String to control return structure. Mirrors pandas options. Default return is dataframe
        @param do_raise <bool>: Boolean flag to suppress errors.
        @param kwargs   <dict>: Dict of additonal args required by P2 to emulate database connection
        """
        if self.client is None:
            raise ValueError('P2 Connection is None')
            return None
        result = None
        try:
            tstart = kwargs['start']
            tend = kwargs['end']
            ptstart = Timestamp(tstart, tz=None)
            ptend = Timestamp(tend, tz=None)
            tags = kwargs['tagnames']
            cols = kwargs['colnames']
            dfns = kwargs['dfnames']
            ptype = kwargs['type']
            pint = kwargs['interval']
            timezone = None
            if 'timezone' in kwargs:
                tzname = kwargs['timezone']
                if tzname in ['UTC', 'utc', 'GMT', 'gmt']:
                    timezone = pytz.UTC
                elif tzname is not None and len(tzname) > 0:
                    timezone = timezone(tzname)
            elif 'timezone' in self.info:
                tzname = self.info['timezone']
                if tzname in ['UTC', 'utc', 'GMT', 'gmt']:
                    timezone = pytz.UTC
                elif tzname is not None and len(tzname) > 0:
                    timezone = timezone(tzname)
            data = self.client.get_data(tags, ptstart, ptend, ptype, pint)
            for i, df in enumerate(data):
                df.columns = [cols[i], 'TS', 'CONF', 'ERROR']
                df = df.loc[(df['TS'] >= tstart) & (df['TS'] <= tend)]
                df = df.loc[df['CONF'] == 100]
                df = df.where(df.notnull(), None)
                df.insert(0, 'TAG', tags[i][1:-1])
                if sql is None:
                    data[i] = self.orient_and_parse(df, orient, timezone, **kwargs)
                else:
                    df = self.orient_and_parse(df, 'df', timezone, **kwargs)
                    if i > 0 and dfns[i] in dfns[:i]:
                        df_ = locals()[dfns[i]]
                        locals()[dfns[i]] = pd.concat([df_, df])
                    else:
                        locals()[dfns[i]] = df
            if sql is None:
                return data
        except Exception as e:
            self._error_handler_(e, do_raise)
        locals()['sql_params'] = args
        df = aasqldf(sql.format(*edit), locals())
        return self.orient_and_parse(df, orient, timezone, **kwargs)
    
    def write(self, sql, args=[], edit=[], do_raise=True, **kwargs):
        return

    def write_many(self, sql, args=[], edit=[], do_raise=True, **kwargs):
        return

class P2(P2Connection):
    """
    Quicker shorthand for constructing the connection object. Has unique string representations.
    """
    def __repr__(self):
        return 'advancedanalytics_util.P2(%r)' % (self.info)

# CSV/Parquet Connection Class
class FileConnection(AAUConnection):
    """
    File Connection Wrapper
    Usage: 
        f = FileConnection()
            ...
    @param info  <dict>: File info
    
    """

    def __repr__(self):
        return 'advancedanalytics_util.FileConnection(%r)' % (self.info)

    def __str__(self):
        return 'FileConnection: [INFO=%s]' % (self.info)

    def _connect_(self):
        """
        Connect to File
        """
        if 'tzname' in self.info:
            if self.info['tzname'] in ['UTC', 'utc', 'GMT', 'gmt']:
                self.timezone = pytz.UTC
            elif self.info['tzname'] is not None and len(self.info['tzname']) > 0:
                self.timezone = timezone(self.info['tzname'])
            else:
                self.timezone = None
        else:
            self.timezone = None
        if 'do_raise' not in self.info:
            self.info['do_raise'] = True
        return True

    def get_files(self, path, prefix=None):
        flist = []
        files = os.listdir(path)
        files.sort()
        for file in files:
            if file.endswith(('csv','parquet')) is False:
                continue
            if prefix is not None and len(prefix) > 0 and file[:len(prefix)] != prefix:
                continue
            flist.append(file)
        return flist

    def read(self, sql=None, args={}, edit=[], orient='list', do_raise=True, **kwargs):
        """
        Execute query parsing return.
        @param sql       <str>: Query to execute (via SQLite - for compatibility with database connection classes). 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <dict>: Not used - for consistency with Oracle connection class
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param orient    <str>: String to control return structure. Mirrors pandas options. Default return is dataframe
        @param do_raise <bool>: Boolean flag to suppress errors.
        @param kwargs   <dict>: Dict of additonal args required by File to emulate database connection
        """
        if 'engine' in kwargs:
            engine = kwargs['engine']
        elif 'engine' in self.info:
            engine = self.info['engine']
        else:
            engine = 'c'
        fp = None
        if 'path' in kwargs:
            fp = kwargs['path']
        elif 'path' in self.info:
            fp = self.info['path']
        if 'partition_mode' in kwargs:
            partition_mode = kwargs['partition_mode']
        else:
            partition_mode = self.info['partition_mode']
        if partition_mode is None:
            try:
                if fp is None:
                    if 'file' in kwargs:
                        fp = kwargs['file']
                    elif 'file' in self.info:
                        fp = self.info['file']
                else:
                    if 'file' in kwargs:
                        fp = '{}/{}'.format(fp, kwargs['file'])
                    elif 'file' in self.info:
                        fp = '{}/{}'.format(fp, self.info['file'])
                if fp.endswith('parquet'):
                    df = pd.read_parquet(fp)
                elif fp.endswith('csv'):
                    df = pd.read_csv(fp, engine=engine, encoding='cp1252')
                else:
                    self._error_handler_(ValueError('File Connection - invalid file type'), do_raise)
                    return None
            except Exception as e:
                self._error_handler_(e, do_raise)
                return None
        elif partition_mode in ['day', 'month', 'year', 'alltime']:
            if 'data_index_column' in kwargs:
                idx = kwargs['data_index_column']
            else:
                idx = self.info['data_index_column']
            if 'start' in kwargs:
                data_start = kwargs['start']
            else:
                data_start = self.info['start']
            if 'end' in kwargs:
                data_end = kwargs['end']
            else:
                data_end = self.info['end']
            data = []
            fs, fe = self.get_filestartend(data_start, data_end, **kwargs)
            if data_end < fe:
                fn = self.get_filename(fs, fe, **kwargs)
                if fp is None:
                    fpn = fn
                else:
                    fpn = '{}/{}'.format(fp, fn)
                try:
                    if fn.endswith('parquet'):
                        data.append(pd.read_parquet(fpn))
                    elif fn.endswith('csv'):
                        data.append(pd.read_csv(fpn, engine=engine, encoding='cp1252'))
                    else:
                        self._error_handler_(ValueError('File Connection - invalid file type'), do_raise)
                        return None
                except Exception:
                    pass
            while (data_end >= fe):
                fn = self.get_filename(fs, fe, **kwargs)
                if fp is None:
                    fpn = fn
                else:
                    fpn = '{}/{}'.format(fp, fn)
                try:
                    if fn.endswith('parquet'):
                        data.append(pd.read_parquet(fpn))
                    elif fn.endswith('csv'):
                        data.append(pd.read_csv(fpn, engine=engine, encoding='cp1252'))
                    else:
                        self._error_handler_(ValueError('File Connection - invalid file type'), do_raise)
                        return None
                except Exception:
                    pass
                data_start = fe
                fs, fe = self.get_filestartend(data_start, data_end, **kwargs)
            if len(data) > 1:
                df = pd.concat(data).drop_duplicates([idx], keep='last')
            elif len(data) > 0 and len(data[0]) > 0:
                df = data[0]
            else:
                cols = None
                if 'colnames' in kwargs and len(kwargs['colnames']) > 0:
                    cols = kwargs['colnames']
                if orient == 'records':
                    return list()
                elif orient == 'list':
                    if cols is None:
                        return {}
                    return dict.fromkeys(cols, ())
                else:
                    df = pd.DataFrame(columns=cols)
                    return df
        if sql is None:
            return self.orient_and_parse(df, orient, self.timezone, **kwargs)
        try:
            if 'colnames' in kwargs and len(kwargs['colnames']) > 0:
                cols = kwargs['colnames']
                df.columns = cols
            tbl = kwargs['src_table']
            locals()[tbl] = self.orient_and_parse(df, 'df', self.timezone, **kwargs)
        except Exception as e:
            self._error_handler_(e, do_raise)
        locals()['sql_params'] = args
        df = aasqldf(sql.format(*edit), locals())
        result = self.orient_and_parse(df, orient, self.timezone, **kwargs)
        return result

    def write(self, sql=None, args=[], edit=[], do_raise=True, **kwargs):
        """
        Execute insert
        @param sql       <str>: Not used - for consistency with Oracle connection class
        @param args     <dict>: Not used - for consistency with Oracle connection class
        @param edit     <list>: Not used - for consistency with Oracle connection class
        @param do_raise <bool>: Boolean flag to suppress errors.
        """
        if 'append' in kwargs:
            append = kwargs['append']
        elif 'append' in self.info:
            append = self.info['append']
        else:
            append = True
        if sql is None:
            try:
                if isinstance(args, list) or isinstance(args, dict):
                    df = pd.DataFrame(args)
                else:
                    df = args
            except Exception as e:
                self._error_handler_(e, do_raise)
                return False
        else:
            try:
                destination_schema = {}
                for key, val in kwargs['dst_schema'].items():
                    if type(val) is str:
                        if val.startswith('pd.Series(dtype=') and val.endswith(')'):
                            tp = val[16:-1]
                            if tp == 'str':
                                destination_schema[key] = pd.Series(dtype=str)
                            elif tp == 'int':
                                destination_schema[key] = pd.Series(dtype=int)
                            elif tp == 'bool':
                                destination_schema[key] = pd.Series(dtype=bool)
                            elif tp == 'float':
                                destination_schema[key] = pd.Series(dtype=float)
                            elif tp == 'object':
                                destination_schema[key] = pd.Series(dtype=object)
                    else:
                        destination_schema[key] = val
                dst_df = pd.DataFrame(destination_schema)
                locals()[kwargs['dst_table']] = dst_df
                if isinstance(args, list) or isinstance(args, dict):
                    locals()['dst_data'] = pd.DataFrame(args)
                else:
                    locals()['dst_data'] = args
                data = aasqldf(sql.format(*edit), locals())
                df = self.orient_and_parse(data, 'df', None, **kwargs)
            except Exception as e:
                self._error_handler_(e, do_raise)
                return False
        fp = None
        if 'path' in kwargs:
            fp = kwargs['path']
        elif 'path' in self.info:
            fp = self.info['path']
        args_ts = ['TS']
        if 'args_ts' in kwargs:
            args_ts = kwargs['args_ts']
        elif 'args_ts' in self.info:
            args_ts = self.info['args_ts']
        if 'partition_mode' in kwargs:
            partition_mode = kwargs['partition_mode']
        else:
            partition_mode = self.info['partition_mode']
        if partition_mode is None:
            try:
                if fp is None:
                    if 'file' in kwargs:
                        fp = kwargs['file']
                    elif 'file' in self.info:
                        fp = self.info['file']
                else:
                    if 'file' in kwargs:
                        fp = '{}/{}'.format(fp, kwargs['file'])
                    elif 'file' in self.info:
                        fp = '{}/{}'.format(fp, self.info['file'])
                if fp.endswith('parquet'):
                    _ = df.to_parquet(path=fp, engine='auto', compression='snappy', index=False, partition_cols=None)
                elif fp.endswith('csv'):
                    if append is True:
                        if not os.path.isfile(fp):
                            df.to_csv(path_or_buf=fp, index=False)
                        else:
                            df.to_csv(path_or_buf=fp, mode='a', header=False, index=False)
                    else:
                        df.to_csv(path_or_buf=fp, index=False)
                else:
                    self._error_handler_(ValueError('File Connection - invalid file type'), do_raise)
                    return False
            except Exception as e:
                self._error_handler_(e, do_raise)
                return False
        else:
            if len(df) > 0:
                if 'data_index_column' in kwargs:
                    idx = kwargs['data_index_column']
                else:
                    idx = self.info['data_index_column']
                data_start = df.head(1).iloc[0][idx]
                data_end = df.tail(1).iloc[0][idx]
                fs, fe = self.get_filestartend(data_start, data_end, **kwargs)
                while (data_end >= fe):
                    fn = self.get_filename(fs, fe, **kwargs)
                    fdata = df.loc[(df[idx] >= fs) & (df[idx] < fe)]
                    if idx in args_ts:
                        fdata[idx] = fdata[idx].values.astype('<M8[s]')
                    if fp is None:
                        fpn = fn
                    else:
                        fpn = '{}/{}'.format(fp, fn)
                    if fn.endswith('parquet'):
                        _ = fdata.to_parquet(path=fpn, engine='auto', compression='snappy', index=False, partition_cols=None)
                    elif fn.endswith('csv'):
                        if append is True:
                            if not os.path.isfile(fpn):
                                fdata.to_csv(path_or_buf=fpn, index=False)
                            else:
                                fdata.to_csv(path_or_buf=fpn, mode='a', header=False, index=False)
                        else:
                            fdata.to_csv(path_or_buf=fpn, index=False)
                    else:
                        self._error_handler_(ValueError('File Connection - invalid file type'), do_raise)
                        return False
                    df = df.loc[(df[idx] >= fe)]
                    if len(df[idx]) == 0:
                        break
                    data_start = df.head(1).iloc[0][idx]
                    fs, fe = self.get_filestartend(data_start, data_end, **kwargs)
                if len(df[idx]) == 0:
                    return True
                fn = self.get_filename(fs, fe, **kwargs)
                fdata = df.loc[(df[idx] >= fs) & (df[idx] < fe)]
                if idx in args_ts:
                    fdata[idx] = fdata[idx].values.astype('<M8[s]')
                if fp is None:
                    fpn = fn
                else:
                    fpn = '{}/{}'.format(fp, fn)
                if fn.endswith('parquet'):
                    _ = fdata.to_parquet(path=fpn, engine='auto', compression='snappy', index=False, partition_cols=None)
                elif fn.endswith('csv'):
                    if append is True:
                        if not os.path.isfile(fpn):
                            fdata.to_csv(path_or_buf=fpn, index=False)
                        else:
                            fdata.to_csv(path_or_buf=fpn, mode='a', header=False, index=False)
                    else:
                        fdata.to_csv(path_or_buf=fpn, index=False)
                else:
                    self._error_handler_(ValueError('File Connection - invalid file type'), do_raise)
                    return False
            else:
                self._error_handler_(ValueError('File Connection - no data'), do_raise)
                return False
        return True

    def write_many(self, sql=None, args=[], edit=[], do_raise=True, **kwargs):
        return self.write(sql, args, edit, do_raise, **kwargs)

class File(FileConnection):
    """
    Quicker shorthand for constructing the connection object. Has unique string representations.
    """
    def __repr__(self):
        return 'advancedanalytics_util.File(%r)' % (self.info)

# AWS S3 Connection Class
class S3Connection(AAUConnection):
    """
    AWS S3 Wrapper
    Usage: 
        s3 = S3Connection(bucket, usr)
            ...
    @param bucket  <str>: s3 bucket name
    @param usr     <str>: user
    """

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.client.shutdown()
        except:
            pass
    
    def __repr__(self):
        return 'advancedanalytics_util.S3Connection(%r)' % (self.info)
    
    def __str__(self):
        return 'S3Connection: [INFO=%s]' % (self.info)

    def _connect_(self, do_raise=True):
        """
        Connect to S3 bucket
        """
        if 'tzname' in self.info:
            if self.info['tzname'] in ['UTC', 'utc', 'GMT', 'gmt']:
                self.timezone = pytz.UTC
            elif self.info['tzname'] is not None and len(self.info['tzname']) > 0:
                self.timezone = timezone(self.info['tzname'])
            else:
                self.timezone = None
        else:
            self.timezone = None
        if self.info['user'] is None:
            self.session = None
            self.s3 = boto3.resource('s3', region_name=self.info['region'])
            self.client = boto3.client('s3')
        else:
            pws = aa.AAPWSAccess(self.info['access']).get_pwd(self.info['bucket'], self.info['user'])
            pub = pws['public_key']
            pri = pws['private_key']
            if pri is not None:
                self.session = boto3.Session(
                    aws_access_key_id=pub,
                    aws_secret_access_key=pri,
                    aws_session_token=self.token
                )
                self.s3 = self.session.resource('s3', region_name=self.info['region'])
                self.client = boto3.client('s3', 
                    aws_access_key_id=pub, 
                    aws_secret_access_key=pri, 
                    aws_session_token=self.token,
                    region_name=self.info['region']
                )
            else:
                self.session = None
                self.s3 = None
                self.client = None
        if self.s3 is None or self.client is None:
            return False
        return True

    def get_buckets(self, name, pathfilter=[]):
        blist = []
        if self.s3 is None:
            return blist
        for b in self.s3.buckets.all():
            if b.name == name:
                tbucket = b
                for k in tbucket.objects.all():
                    p = k.key.split('/')
                    f = [True for fi, fv in enumerate(pathfilter) 
                         if len(p) > fi and 
                             ((fv['ftype'] == 'prefix' and p[fi].startswith(fv['filter'])) or
                              (fv['ftype'] == 'suffix' and p[fi].endswith(fv['filter'])) or
                              (fv['ftype'] == 'contains' and fv['filter'] in p[fi]) or
                              p[fi] == fv['filter'])]
                    if len(f) != len(pathfilter):
                        continue
                    blist.append(p)
        return blist

    def read(self, sql=None, args={}, edit=[], orient='list', do_raise=True, **kwargs):
        """
        Execute query parsing return.
        @param sql       <str>: Query to execute (via SQLite - for compatibility with database connection classes). 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <dict>: Not used - for consistency with Oracle connection class
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param orient    <str>: String to control return structure. Mirrors pandas options. Default return is dataframe
        @param do_raise <bool>: Boolean flag to suppress errors.
        @param kwargs   <dict>: Dict of additonal args required by File to emulate database connection
        """
        if self.s3 is None:
            self._error_handler_(ValueError('No S3 Connection'), do_raise)
            return False
        b = None
        if 'bucket' in kwargs:
            b = kwargs['bucket']
        elif 'bucket' in self.info:
            b = self.info['bucket']
        fp = None
        if 'path' in kwargs:
            fp = kwargs['path']
        elif 'path' in self.info:
            fp = self.info['path']
        if 'engine' in kwargs:
            engine = kwargs['engine']
        elif 'engine' in self.info:
            engine = self.info['engine']
        else:
            engine = 'c'
        if b is None:
            self._error_handler_(ValueError('S3 Connection - invalid path'), do_raise)
            return False
        read_string = False
        read_blob = False
        if 'read_string' in kwargs:
            read_string = kwargs['read_string']
        if 'read_blob' in kwargs:
            read_blob = kwargs['read_blob']
        if 'partition_mode' in kwargs:
            partition_mode = kwargs['partition_mode']
        else:
            partition_mode = self.info['partition_mode']
        if partition_mode is None:
            try:
                if fp is None:
                    if 'file' in kwargs:
                        fp = kwargs['file']
                    elif 'file' in self.info:
                        fp = self.info['file']
                else:
                    if 'file' in kwargs:
                        fp = '{}/{}'.format(fp, kwargs['file'])
                    elif 'file' in self.info:
                        fp = '{}/{}'.format(fp, self.info['file'])
                s3obj = self.client.get_object(Bucket=b, Key=fp)
                if read_string is True:
                    bio = io.BytesIO()
                    self.client.download_fileobj(b, fp, bio)
                    byte_str = bio.getvalue()
                    text_obj = byte_str.decode('UTF-8')
                    sio = io.StringIO(text_obj)
                    return sio
                elif read_blob is True:
                    bio = io.BytesIO()
                    self.client.download_fileobj(b, fp, bio)
                    return bio
                elif fp.endswith('parquet'):
                    df = pd.read_parquet(io.BytesIO(s3obj['Body'].read()))
                elif fp.endswith('csv'):
                    df = pd.read_csv(io.BytesIO(s3obj['Body'].read()), engine=engine, encoding='cp1252')
                else:
                    self._error_handler_(ValueError('S3 Connection - invalid type'), do_raise)
                    return None
            except Exception as e:
                self._error_handler_(e, do_raise)
                return None
        elif partition_mode in ['day', 'month', 'year', 'alltime']:
            if 'data_index_column' in kwargs:
                idx = kwargs['data_index_column']
            else:
                idx = self.info['data_index_column']
            if 'start' in kwargs:
                data_start = kwargs['start']
            elif 'start' in self.info:
                data_start = self.info['start']
            if 'end' in kwargs:
                data_end = kwargs['end']
            elif 'end' in self.info:
                data_end = self.info['end']
            data = []
            fs, fe = self.get_filestartend(data_start, data_end, **kwargs)
            while (data_end >= fe):
                fn = self.get_filename(fs, fe, **kwargs)
                if fp is None:
                    fpn = fn
                else:
                    fpn = '{}/{}'.format(fp, fn)
                try:
                    s3obj = self.client.get_object(Bucket=b, Key=fpn)
                    if fn.endswith('parquet'):
                        data.append(pd.read_parquet(io.BytesIO(s3obj['Body'].read())))
                    elif fn.endswith('csv'):
                        data.append(pd.read_csv(io.BytesIO(s3obj['Body'].read()), engine=engine, encoding='cp1252'))
                    else:
                        self._error_handler_(ValueError('S3 Connection - invalid type'), do_raise)
                        return None
                except Exception:
                    pass
                data_start = fe
                fs, fe = self.get_filestartend(data_start, data_end, **kwargs)
            if len(data) > 1:
                df = pd.concat(data).drop_duplicates([idx], keep='last')
            elif len(data) > 0 and len(data[0]) > 0:
                df = data[0]
            else:
                cols = None
                if 'colnames' in kwargs and len(kwargs['colnames']) > 0:
                    cols = kwargs['colnames']
                if orient == 'records':
                    return list()
                elif orient == 'list':
                    if cols is None:
                        return {}
                    return dict.fromkeys(cols, ())
                else:
                    df = pd.DataFrame(columns=cols)
                    return df
        if sql is None:
            return self.orient_and_parse(df, orient, self.timezone, **kwargs)
        try:
            if 'colnames' in kwargs and len(kwargs['colnames']) > 0:
                cols = kwargs['colnames']
                df.columns = cols
            tbl = kwargs['src_table']
            locals()[tbl] = self.orient_and_parse(df, 'df', self.timezone, **kwargs)
        except Exception as e:
            self._error_handler_(e, do_raise)
        locals()['sql_params'] = args
        df = aasqldf(sql.format(*edit), locals())
        return self.orient_and_parse(df, orient, self.timezone, **kwargs)

    def write(self, sql=None, args=[], edit=[], do_raise=True, **kwargs):
        """
        Execute insert.
        @param sql       <str>: Query to execute (via SQLite - for compatibility with database connection classes). 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <list, dict or dataframe>: Data to write.
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param do_raise <bool>: Boolean flag to suppress errors.
        @param kwargs   <dict>: Dict of additonal args required by File to emulate database connection
        """
        if self.s3 is None:
            self._error_handler_(ValueError('No S3 Connection'), do_raise)
            return False
        if sql is None:
            try:
                if isinstance(args, list) or isinstance(args, dict):
                    df = pd.DataFrame(args)
                else:
                    df = args
            except Exception as e:
                self._error_handler_(e, do_raise)
                return False
        else:
            try:
                destination_schema = {}
                for key, val in kwargs['dst_schema'].items():
                    if type(val) is str:
                        if val.startswith('pd.Series(dtype=') and val.endswith(')'):
                            tp = val[16:-1]
                            if tp == 'str':
                                destination_schema[key] = pd.Series(dtype=str)
                            elif tp == 'int':
                                destination_schema[key] = pd.Series(dtype=int)
                            elif tp == 'bool':
                                destination_schema[key] = pd.Series(dtype=bool)
                            elif tp == 'float':
                                destination_schema[key] = pd.Series(dtype=float)
                            elif tp == 'object':
                                destination_schema[key] = pd.Series(dtype=object)
                    else:
                        destination_schema[key] = val
                dst_df = pd.DataFrame(destination_schema)
                locals()[kwargs['dst_table']] = dst_df
                if isinstance(args, list) or isinstance(args, dict):
                    locals()['dst_data'] = pd.DataFrame(args)
                else:
                    locals()['dst_data'] = args
                data = aasqldf(sql.format(*edit), locals())
                df = self.orient_and_parse(data, 'df', None, **kwargs)
            except Exception as e:
                self._error_handler_(e, do_raise)
                return False
        b = None
        if 'bucket' in kwargs:
            b = kwargs['bucket']
        elif 'bucket' in self.info:
            b = self.info['bucket']
        fp = None
        if 'path' in kwargs:
            fp = kwargs['path']
        elif 'path' in self.info:
            fp = self.info['path']
        if 'engine' in kwargs:
            engine = kwargs['engine']
        elif 'engine' in self.info:
            engine = self.info['engine']
        else:
            engine = 'c'
        args_ts = ['TS']
        if 'args_ts' in kwargs:
            args_ts = kwargs['args_ts']
        elif 'args_ts' in self.info:
            args_ts = self.info['args_ts']
        if b is None:
            self._error_handler_(ValueError('S3 Connection - invalid path'), do_raise)
            return False
        if 'partition_mode' in kwargs:
            partition_mode = kwargs['partition_mode']
        else:
            partition_mode = self.info['partition_mode']
        if partition_mode is None:
            try:
                if fp is None:
                    if 'file' in kwargs:
                        fp = kwargs['file']
                    elif 'file' in self.info:
                        fp = self.info['file']
                else:
                    if 'file' in kwargs:
                        fp = '{}/{}'.format(fp, kwargs['file'])
                    elif 'file' in self.info:
                        fp = '{}/{}'.format(fp, self.info['file'])
                if fp.endswith('parquet'):
                    f = df.to_parquet(path=None, engine='auto', compression='snappy', index=False, partition_cols=None)
                elif fp.endswith('csv'):
                    f = df.to_csv(path_or_buf=None, index=False)
                else:
                    self._error_handler_(ValueError('S3 Connection - invalid file type'), do_raise)
                    return False
                s3obj = self.s3.Object(b, fp)
                s3obj.put(Body=f)
            except Exception as e:
                self._error_handler_(e, do_raise)
                return False
        else:
            if len(df) > 0:
                if 'data_index_column' in kwargs:
                    idx = kwargs['data_index_column']
                else:
                    idx = self.info['data_index_column']
                data_start = df.head(1).iloc[0][idx]
                data_end = df.tail(1).iloc[0][idx]
                fs, fe = self.get_filestartend(data_start, data_end, **kwargs)
                while (data_end >= fe):
                    fn = self.get_filename(fs, fe, **kwargs)
                    fdata = df.loc[(df[idx] >= fs) & (df[idx] < fe)]
                    if idx in args_ts:
                        fdata[idx] = fdata[idx].values.astype('<M8[s]')
                    if fp is None:
                        fpn = fn
                    else:
                        fpn = '{}/{}'.format(fp, fn)
                    if fn.endswith('parquet'):
                        f = fdata.to_parquet(path=None, engine='auto', compression='snappy', index=False, partition_cols=None)
                    elif fn.endswith('csv'):
                        f = fdata.to_csv(path_or_buf=None, index=False)
                    else:
                        self._error_handler_(ValueError('S3 Connection - invalid file type'), do_raise)
                        return False
                    s3obj = self.s3.Object(b, fpn)
                    s3obj.put(Body=f)
                    df = df.loc[(df[idx] >= fe)]
                    if len(df[idx]) == 0:
                        break
                    data_start = df.head(1).iloc[0][idx]
                    fs, fe = self.get_filestartend(data_start, data_end, **kwargs)
                if len(df[idx]) == 0:
                    return True
                fn = self.get_filename(fs, fe, **kwargs)
                fdata = df.loc[(df[idx] >= fs) & (df[idx] < fe)]
                if idx in args_ts:
                    fdata[idx] = fdata[idx].values.astype('<M8[s]')
                if fp is None:
                    fpn = fn
                else:
                    fpn = '{}/{}'.format(fp, fn)
                if fn.endswith('parquet'):
                    f = fdata.to_parquet(path=None, engine='auto', compression='snappy', index=False, partition_cols=None)
                elif fn.endswith('csv'):
                    f = fdata.to_csv(path_or_buf=None, index=False)
                else:
                    self._error_handler_(ValueError('S3 Connection - invalid file type'), do_raise)
                    return False
                s3obj = self.s3.Object(b, fpn)
                s3obj.put(Body=f)
            else:
                self._error_handler_(ValueError('File Connection - no data'), do_raise)
                return False
        return True

    def write_many(self, sql=None, args=[], edit=[], do_raise=True, **kwargs):
        return self.write(sql=sql, args=args, edit=edit, do_raise=do_raise, **kwargs)

class S3(S3Connection):
    """
    Quicker shorthand for constructing the connection object. Has unique string representations.
    """
    def __repr__(self):
        return 'advancedanalytics_util.S3(%r)' % (self.info)

# Real Time Simulator Connection Class
class RTSConnection(AAUConnection):
    """
    RTS Wrapper
    Usage: 
        rts = RTSConnection()
            ...
    @param info  <dict>: File info to use to emulate streaming data source
    
    """
    def __exit__(self, exc_type, exc_value, traceback):
        for v in self.info:
            if v['reader'] is not None:
                v['reader'].close()
                v['reader'] = None

    def __repr__(self):
        return 'advancedanalytics_util.RTSConnection(%r)' % (self.info)

    def __str__(self):
        return 'RTSConnection: [INFO=%s]' % (self.info)

    def get_files(self, info):
        flist = []
        files = info['con'].get_buckets(info['connection_info']['bucket'], info['pathfilter'])
        for fp in files:
            if len(fp) == 0:
                continue
            fn = fp[-1]
            if fn[-3:] != 'csv':
                continue
            flist.append(fn)
        return flist

    def get_reader(self, file_or_buf, read_size=None):
        if read_size is None:
            r = pd.read_csv(file_or_buf, iterator=True, low_memory=False)
        else:
            r = pd.read_csv(file_or_buf, chunksize=read_size, low_memory=False)
        return r
        
    def _connect_(self):
        """
        Simulating streaming data: This function returns the TextFileReader object to iterate through a file
        """
        first_files = []
        for i in self.info:
            if i['type'] == 's3':
                i['con'] = S3(i['connection_info'])
            elif i['type'] == 'file':
                i['con'] = File(i['connection_info'])
            if i['file'] is None:
                i['filelist'] = self.get_files(i)
            elif isinstance(i['file'], list):
                i['filelist'] = i['file']
            else:
                i['filelist'] = []
            if len(i['filelist']) > 0:
                i['file'] = i['filelist'][0]
                i['fileindex'] = 0
            elif i['file'] is None:
                return False
            if i['tzname'] in ['UTC', 'utc', 'GMT', 'gmt']:
                i['timezone'] = pytz.UTC
            elif i['tzname'] is not None and len(i['tzname']) > 0:
                i['timezone'] = timezone(i['tzname'])
            else:
                i['timezone'] = None
            kwargs = i['kwargs']
            if i['type'] == 's3':
                kwargs['read_string'] = False
                kwargs['file'] = i['file']
                kwargs['read_string'] = True
                fp = i['con'].read(sql=None, args={}, edit=[], orient=None, do_raise=True, **kwargs)
                first_files.append(i['file'])
            if isinstance(i['read_size'], dict):
                i['reader'] = self.get_reader(fp, read_size=None)
            else:
                i['reader'] = self.get_reader(fp, read_size=i['read_size'])
        return True

    def read(self, sql, args={}, edit=[], orient='list', do_raise=True, **kwargs):
        """
        Execute query parsing return.
        @param sql       <str>: Query to execute (via SQLite - for compatibility with database connection classes). 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <dict>: Not used - for consistency with Oracle connection class
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param orient    <str>: String to control return structure. Mirrors pandas options. Default return is dataframe
        @param do_raise <bool>: Boolean flag to suppress errors.
        @param kwargs   <dict>: Dict of additonal args required by RTS to emulate database connection
        """
        read_size = None
        timezone = None
        if 'timezone' in kwargs:
            tzname = kwargs['timezone']
            if tzname in ['UTC', 'utc', 'GMT', 'gmt']:
                timezone = pytz.UTC
            elif tzname is not None and len(tzname) > 0:
                timezone = timezone(tzname)
        elif 'timezone' in self.info:
            tzname = self.info['timezone']
            if tzname in ['UTC', 'utc', 'GMT', 'gmt']:
                timezone = pytz.UTC
            elif tzname is not None and len(tzname) > 0:
                timezone = timezone(tzname)
        next_files = []
        endoffiles = False
        for i in self.info:
            if i['reader'] is None:
                if do_raise is True:
                    raise ValueError('RTS Connection is None')
                    return None
                else:
                    continue
            if i['read_size'] is None:
                if do_raise is True:
                    raise ValueError('RTS Connection read_size is None')
                    return None
                else:
                    continue
            if 'colnames' in i and len(i['colnames']) > 0:
                edf = pd.DataFrame(columns=i['colnames'])
            else:
                edf = pd.DataFrame()
            df = None
            while (True):
                if isinstance(i['read_size'], dict):
                    if read_size is None:
                        read_size = randint(i['read_size']['min_read'], i['read_size']['max_read'])
                else:
                    read_size = i['read_size']
                if read_size == 0:
                    break
                try:
                    df = i['reader'].get_chunk(read_size)
                except StopIteration:
                    pass
                if df is None or df.empty:
                    # move to the next set of files in the series
                    df = None
                    if i['fileindex'] < len(i['filelist']) - 1:
                        i['fileindex'] += 1
                        i['file'] = i['filelist'][i['fileindex']]
                        next_files.append(i['file'])
                        if i['type'] == 's3':
                            kwargs['read_string'] = True
                            kwargs['file'] = i['file']
                            fp = i['con'].read(sql=None, args={}, edit=[], orient=None, do_raise=True, **kwargs)
                        if isinstance(i['read_size'], dict):
                            i['reader'] = self.get_reader(fp, read_size=None)
                        else:
                            i['reader'] = self.get_reader(fp, read_size=i['read_size'])
                    else:
                        endoffiles = True
                        break
                else:
                    break
            if df is None:
                if endoffiles is True:
                    return None
                df = edf
            locals()[i['src_table']] = self.orient_and_parse(df, 'df', timezone, **kwargs)
        locals()['sql_params'] = args
        df = aasqldf(sql.format(*edit), locals())
        return self.orient_and_parse(df, orient, None, **kwargs)

    def write(self, sql, args=[], edit=[], do_raise=True, **kwargs):
        return

    def write_many(self, sql, args=[], edit=[], do_raise=True, **kwargs):
        return

class RTS(RTSConnection):
    """
    Quicker shorthand for constructing the connection object. Has unique string representations.
    """
    def __repr__(self):
        return 'advancedanalytics_util.RTS(%r)' % (self.info)

# Oracle Database Connection Class
class OracleDatabaseConnection(AAUConnection):
    """
    Oracle Database Wrapper
    Usage: 
        db = OracleDatabaseConnection(db, usr)
            ...
    @param db  <str>: Database name
    @param usr <str>: Database user
    """

    def __init__(self, info):
        self.info = info
        self.con = None
        # This is the path to the ORACLE client files
        lib_dir = r"C:\Oracle\client_x64\instantclient_21_6"
        if 'lib_dir' in info and len(info['lib_dir']) > 0:
            lib_dir = info['lib_dir']
        try:
            cx_Oracle.init_oracle_client(lib_dir=lib_dir)
        except:
            pass
        _ = self._connect_()
        self.sets = dict(blob=cx_Oracle.BLOB, timestamp=cx_Oracle.TIMESTAMP)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.con is not None:
            try:
                self.con.close()
                self.con = None
            except:
                pass

    def __repr__(self):
        return 'advancedanalytics_util.OracleDatabaseConnection(%r)' % (self.info)

    def __str__(self):
        return 'OracleDatabaseConnection: [INFO=%s]' % (self.info)

    def _connect_(self, thread=False, do_raise=True):
        """
        Retrieve password and connect to database. Store connection.
        """
        try:
            pwd = aa.AAPWSAccess(self.info['access']).get_pwd(self.info['db'], self.info['usr'])['password']
            if pwd is not None:
                self.con = cx_Oracle.connect(self.info['usr'], pwd, self.info['db'], threaded=thread)
                return True
            else:
                self.con = None
        except Exception as e:
            self._error_handler_(e, do_raise)
        return False

    def _prepare_file_(self, cur, args, content):
        """
        Takes file content and reassigns into the dictionary as a correctly-constructed blob.

        @param cur      <obj>: Current cursor object.
        @param args    <dict>: Existing and prepared arguments dictionary.
        @param content <dict>: Dictionary of target bind names with the value holding the desired file content.
        """
        for arg in content:
            blob_var = cur.var(cx_Oracle.BLOB)
            blob_var.setvalue(0, content[arg])
            args[arg] = blob_var
        return args

    def _prepare_size_(self, **kwargs):
        """
        Prepare dictionary of bind variables and corresponding input sizes to set. 
        Done by providing lists via keyword arguments into the function.

        @kwargs <list>: Keyword arguments must match keys within the wrapper's sets definition.
                        Value of the keyword argument should be a list of bind names.
        """
        sets = {}
        for set in self.sets:
            if set in kwargs:
                for arg in kwargs[set]:
                    sets[arg] = self.sets[set]
        return sets

    def commit(self, do_raise=True):
        try:
            self.con.commit()
        except cx_Oracle.DatabaseError as e:
            self._error_handler_(e, do_raise)

    def execute(self, sql, args={}, args_ts=[], edit=[], do_raise=True, **kwargs):
        """
        Execute query without explicitly committing (may not be necessary).
        
        @param sql       <str>: Query to execute. 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <dict>: Dictionary of bind or positional variables and their values for replacement. Keyword arguments will also be prepared into this dictionary.
        @param args_ts  <list>: List of bind variables that must have their cursor input size set to timestamp. [NOT COMPATIBLE WITH POSITIONAL ARGUMENTS]
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param do_raise <bool>: Boolean flag to suppress errors.
        """
        cur = self.con.cursor()
        sets = self._prepare_size_(timestamp=args_ts)
        if sets:
            cur.setinputsizes(**sets)
        try:
            cur.execute(sql.format(*edit), args)
            cur.close()
        except cx_Oracle.DatabaseError as e:
            cur.close()
            self._error_handler_(e, do_raise)
            
    def execute_many(self, sql, args_list=[], args_ts=[], edit=[], do_raise=True):
        """
        Execute query for multiple records without explicitly committing (may not be necessary).
        
        @param sql        <str>: Query to execute. 
                                 Allows string replacement through another argument and the string.format() method.
                                 However, this may block regular expressions from being hardcoded into the query. 
                                 In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args_list <list>: List of dictionaries (if bind) or lists (if positional) variables and their values for replacement.
        @param args_ts   <list>: List of bind variables that must have their cursor input size set to timestamp. [NOT COMPATIBLE WITH POSITIONAL ARGUMENTS]
        @param edit      <list>: List of direct string replacement parameters in positional order.
        @param do_raise  <bool>: Boolean flag to suppress errors.
        """
        cur = self.con.cursor()
        sets = self._prepare_size_(timestamp=args_ts)
        if sets:
            cur.setinputsizes(**sets)
        try:
            cur.prepare(sql.format(*edit))
            cur.executemany(None, args_list)
            cur.close()
        except cx_Oracle.DatabaseError as e:
            cur.close()
            self._error_handler_(e, do_raise)

    def read(self, sql, args={}, edit=[], orient='list', do_raise=True, **kwargs):
        """
        Execute query parsing return.
        @param sql       <str>: Query to execute. 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <list>: Dictionary of bind or positional variables and their values for replacement. Keyword arguments will also be prepared into this dictionary.
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param orient    <str>: String to control return structure. Mirrors pandas options.
        @param do_raise <bool>: Boolean flag to suppress errors.
        """
        if self.con is None:
            raise ValueError('Oracle Connection is None')
            return None
        timezone = None
        if 'timezone' in kwargs:
            tzname = kwargs['timezone']
            if tzname in ['UTC', 'utc', 'GMT', 'gmt']:
                timezone = pytz.UTC
            elif tzname is not None and len(tzname) > 0:
                timezone = timezone(tzname)
        elif 'timezone' in self.info:
            tzname = self.info['timezone']
            if tzname in ['UTC', 'utc', 'GMT', 'gmt']:
                timezone = pytz.UTC
            elif tzname is not None and len(tzname) > 0:
                timezone = timezone(tzname)
        cur = self.con.cursor()
        try:
            cur.execute(sql.format(*edit), args)
            data = list(zip(*cur.fetchall()))
            head = list(list(zip(*cur.description))[0])
            cur.close()
            if not data or len(data) == 0:
                if orient == 'records':
                    return list()
                elif orient == 'list':
                    return dict.fromkeys(head, ())
                else:
                    df = pd.DataFrame(columns=head)
                    return df
            df = pd.DataFrame(dict(zip(head, data)))
            return self.orient_and_parse(df, orient, timezone, **kwargs)
        except cx_Oracle.DatabaseError as e:
            cur.close()
            self._error_handler_(e, do_raise)
    
    def write(self, sql, args={}, edit=[], do_raise=True, **kwargs):
        """
        Execute query with explicit commit.
        @param sql       <str>: Query to execute. 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <dict>: Dictionary of bind or positional variables and their values for replacement. Keyword arguments will also be prepared into this dictionary.
        @param args_ts  <list>: In kwargs - List of bind variables that must have their cursor input size set to timestamp. [NOT COMPATIBLE WITH POSITIONAL ARGUMENTS]
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param do_raise <bool>: Boolean flag to suppress errors.
        """
        if self.con is None:
            raise ValueError('Oracle Connection is None')
            return
        try:
            cur = self.con.cursor()
            sets = {}
            if 'args_ts' in kwargs:
                sets = self._prepare_size_(timestamp=kwargs['args_ts'])
        except cx_Oracle.DatabaseError as e:
            if cur is not None:
                cur.close()
            self._error_handler_(e, do_raise)
            return
        if sets:
            cur.setinputsizes(**sets)
        try:
            cur.execute(sql.format(*edit), args)
            self.con.commit()
            cur.close()
        except cx_Oracle.DatabaseError as e:
            cur.close()
            self._error_handler_(e, do_raise)

    def write_many(self, sql, args=[], edit=[], do_raise=True, **kwargs):
        """
        Execute query for multiple records with explicit commit.
        @param sql        <str>: Query to execute. 
                                 Allows string replacement through another argument and the string.format() method.
                                 However, this may block regular expressions from being hardcoded into the query. 
                                 In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args_list <list>: List of dictionaries (if bind) or lists (if positional) variables and their values for replacement.
        @param args_ts   <list>: In kwargs - List of bind variables that must have their cursor input size set to timestamp. [NOT COMPATIBLE WITH POSITIONAL ARGUMENTS]
        @param edit      <list>: List of direct string replacement parameters in positional order.
        @param do_raise  <bool>: Boolean flag to suppress errors.
        """
        if self.con is None:
            raise ValueError('Oracle Connection is None')
            return
        try:
            cur = self.con.cursor()
            sets = {}
            if 'args_ts' in kwargs:
                sets = self._prepare_size_(timestamp=kwargs['args_ts'])
        except cx_Oracle.DatabaseError as e:
            if cur is not None:
                cur.close()
            self._error_handler_(e, do_raise)
            return
        if sets:
            cur.setinputsizes(**sets)
        try:
            cur.prepare(sql.format(*edit))
            cur.executemany(None, args)
            self.con.commit()
            cur.close()
        except cx_Oracle.DatabaseError as e:
            cur.close()
            self._error_handler_(e, do_raise)

class Oracle(OracleDatabaseConnection):

    """
    Quicker shorthand for constructing the connection object. Has unique string representations.
    """
    
    def __repr__(self):
        return 'advancedanalytics_util.Oracle(%r, %r)' % (self.db, self.usr)

class ODBC(OracleDatabaseConnection):

    """
    Quicker shorthand for constructing the connection object. Has unique string representations.
    """
    
    def __repr__(self):
        return 'advancedanalytics_util.ODBC(%r, %r)' % (self.db, self.usr)
        
# SQL Server Database Connection Class            
class SQLServerDatabaseConnection(AAUConnection): # This is not compatible with multithreading.
    """
    SQLServer Database Wrapper
    Usage:
        db = SQLServerDatabaseConnection(srv, db, sid, usr)
            ...
    @param srv <str>: Server name
    @param  db <str>: Database name
    @param sid <str>: Service identifier
    @param usr <str>: Database user
    """
    def __exit__(self, exc_type, exc_value, traceback):
        if self.con is not None:
            try:
                self.con.close()
                self.con = None
            except:
                pass

    def __repr__(self):
        return 'advancedanalytics_util.SQLServerDatabaseConnection(%r)' % (self.info)

    def __str__(self):
        return 'SQLServerDatabaseConnection: [INFO=%s]' % (self.info)

    def _connect_(self, do_raise=True):
        try:
            pwd = aa.AAPWSAccess(self.info['access']).get_pwd(self.info['sid'], self.info['usr'])['password']
            if pwd is not None:
                self.con = pymssql.connect(self.info['srv'], self.info['usr'], pwd, self.info['db'])
                return True
            else:
                self.con = None
        except Exception as e:
            self._error_handler_(e, do_raise)
        return False

    def read(self, sql, args=(), edit=[], orient='list', do_raise=True, **kwargs):
        if self.con is None:
            raise ValueError('SQL Connection is None')
            return None
        timezone = None
        if 'timezone' in kwargs:
            tzname = kwargs['timezone']
            if tzname in ['UTC', 'utc', 'GMT', 'gmt']:
                timezone = pytz.UTC
            elif tzname is not None and len(tzname) > 0:
                timezone = timezone(tzname)
        elif 'timezone' in self.info:
            tzname = self.info['timezone']
            if tzname in ['UTC', 'utc', 'GMT', 'gmt']:
                timezone = pytz.UTC
            elif tzname is not None and len(tzname) > 0:
                timezone = timezone(tzname)
        try:
            cur = self.con.cursor()
            cur.execute(sql.format(*edit), args)
            data = list(zip(*cur.fetchall()))
            head = list(list(zip(*cur.description))[0])
            cur.close()
            if not data:
                if orient == 'records':
                    return list()
                elif orient == 'list':
                    return dict.fromkeys(head, ())
                else:
                    df = pd.DataFrame(columns=head)
                    return df
            df = pd.DataFrame(dict(zip(head, data)))
            return self.orient_and_parse(df, orient, timezone, **kwargs)
        except pymssql.DatabaseError as e:
            cur.close()
            self._error_handler_(e, do_raise)
            
    def write(self, sql, args={}, edit=[], do_raise=True, **kwargs):
        """
        Execute query with explicit commit.
        @param sql       <str>: Query to execute. 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <dict>: Dictionary of bind or positional variables and their values for replacement. Keyword arguments will also be prepared into this dictionary.
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param do_raise <bool>: Boolean flag to suppress errors.
        """
        if self.con is None:
            raise ValueError('SQL Connection is None')
            return
        try:
            cur = self.con.cursor()
        except pymssql.DatabaseError as e:
            if cur is not None:
                cur.close()
            self._error_handler_(e, do_raise)
            return
        try:
            cur.execute(sql.format(*edit), args)
            self.con.commit()
            cur.close()
        except pymssql.DatabaseError as e:
            cur.close()
            self._error_handler_(e, do_raise)

    def write_many(self, sql, args=[], edit=[], do_raise=True, **kwargs):
        """
        Execute query for multiple records with explicit commit.
        @param sql        <str>: Query to execute. 
                                 Allows string replacement through another argument and the string.format() method.
                                 However, this may block regular expressions from being hardcoded into the query. 
                                 In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args      <list>: List of dictionaries (if bind) or lists (if positional) variables and their values for replacement.
        @param edit      <list>: List of direct string replacement parameters in positional order.
        @param do_raise  <bool>: Boolean flag to suppress errors.
        """
        if self.con is None:
            raise ValueError('SQL Connection is None')
            return
        try:
            cur = self.con.cursor()
        except pymssql.DatabaseError as e:
            if cur is not None:
                cur.close()
            self._error_handler_(e, do_raise)
            return
        try:
            cur.prepare(sql.format(*edit))
            cur.executemany(None, args)
            self.con.commit()
            cur.close()
        except pymssql.DatabaseError as e:
            cur.close()
            self._error_handler_(e, do_raise)
            
class SQLServer(SQLServerDatabaseConnection):
    """
    Quicker shorthand for constructing the connection object. Has unique string representations.
    """
    def __repr__(self):
        return 'advancedanalytics_util.SQLServer(%r, %r, %r, %r)' % (self.srv, self.db, self.sid, self.usr)
        
class MSSQL(SQLServerDatabaseConnection):
    """
    Quicker shorthand for constructing the connection object. Has unique string representations.
    """
    def __repr__(self):
        return 'advancedanalytics_util.MSSQL(%r, %r, %r, %r)' % (self.srv, self.db, self.sid, self.usr)  

# Snowflake Connection Class
class SnowflakeConnection(AAUConnection):
    """
    Snowflake Wrapper
    Usage: 
            ...
    """
    def __repr__(self):
        return 'advancedanalytics_util.SnowflakeConnection(%r)' % (self.info)

    def __str__(self):
        return 'SnowlflakeConnection: [INFO=%s]' % (self.info)

    def _connect_(self, do_raise=True):
        try:
            pwd = aa.AAPWSAccess(self.info['access']).get_pwd(self.info['database'], self.info['user'])['password']
            if pwd is not None:
                if 'paramstyle' not in self.info:
                    self.info['paramstyle'] = 'numeric'
                snow.paramstyle = self.info['paramstyle']
                self.client = snow.connect(
                        user = self.info['user'],
                        password = pwd,
                        account = self.info['account_id'],
                        warehouse = self.info['warehouse'],
                        database = self.info['database'],
                        schema = self.info['schema'],
                        session_parameters = {
                                'TIMESTAMP_TYPE_MAPPING': 'TIMESTAMP_NTZ'
                            }
                    )
                return True
            else:
                self.client = None
        except Exception as e:
            self._error_handler_(e, do_raise)
        return False

    def read(self, sql, args={}, edit=[], orient='list', do_raise=True, **kwargs):
        """
        Execute query parsing return.
        @param sql       <str>: Query to execute. 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <list>: Dictionary of bind or positional variables and their values for replacement. Keyword arguments will also be prepared into this dictionary.
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param orient    <str>: String to control return structure. Mirrors pandas options.
        @param do_raise <bool>: Boolean flag to suppress errors.
        """
        if self.client is None:
            raise ValueError('Snowflake client is None')
            return None
        if 'paramstyle' in kwargs:
            if kwargs['paramstyle'] != self.info['paramstyle']:
                self.info['paramstyle'] = kwargs['paramstyle']
                self.reconnect()
        timezone = None
        if 'timezone' in kwargs:
            tzname = kwargs['timezone']
            if tzname in ['UTC', 'utc', 'GMT', 'gmt']:
                timezone = pytz.UTC
            elif tzname is not None and len(tzname) > 0:
                timezone = timezone(tzname)
        elif 'timezone' in self.info:
            tzname = self.info['timezone']
            if tzname in ['UTC', 'utc', 'GMT', 'gmt']:
                timezone = pytz.UTC
            elif tzname is not None and len(tzname) > 0:
                timezone = timezone(tzname)
        try:
            cur = self.client.cursor()
            cur.execute(sql.format(*edit), args)
            df = cur.fetch_pandas_all()
            cur.close()
            return self.orient_and_parse(df, orient, timezone, **kwargs)
        except Exception:
            try:
                sql = 'ALTER WAREHOUSE {} RESUME IF SUSPENDED'.format(self.info['warehouse'])
                cur.execute(sql)
                cur.execute(sql.format(*edit), args)
                df = cur.fetch_pandas_all()
                cur.close()
                return self.orient_and_parse(df, orient, timezone, **kwargs)
            except Exception as e:
                cur.close()
                self._error_handler_(e, do_raise)

    def write(self, sql, args=[], edit=[], do_raise=True, **kwargs):
        """
        Execute query with explicit commit.
        @param sql       <str>: Query to execute. 
                                Allows string replacement through another argument and the string.format() method.
                                However, this may block regular expressions from being hardcoded into the query. 
                                In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args     <dict>: Dictionary of bind or positional variables and their values for replacement. Keyword arguments will also be prepared into this dictionary.
        @param args_ts  <list>: In kwargs - List of bind variables that must have their cursor input size set to timestamp. [NOT COMPATIBLE WITH POSITIONAL ARGUMENTS]
        @param edit     <list>: List of direct string replacement parameters in positional order.
        @param do_raise <bool>: Boolean flag to suppress errors.
        """
        if self.client is None:
            raise ValueError('Snowflake client is None')
            return
        try:
            cur = self.client.cursor()
            tsidx = [1 if isinstance(v, pd.Timestamp) else 0 for v in args]
            args = [v if tsidx[i] == 0 else datetime(v.year, v.month, v.day, v.hour, v.minute, v.second) for i, v in enumerate(args)]
            cur.execute(sql.format(*edit), args)
            cur.close()
        except Exception as e1:
            try:
                sql = 'ALTER WAREHOUSE {} RESUME IF SUSPENDED'.format(self.info['warehouse'])
                cur.execute(sql)
                cur.execute(sql.format(*edit), args)
                cur.close()
            except Exception as e2:
                cur.close()
                self._error_handler_(e1, False)
                self._error_handler_(e2, do_raise)

    def write_many(self, sql, args=[], edit=[], do_raise=True, **kwargs):
        """
        Execute query for multiple records with explicit commit.
        @param sql        <str>: Query to execute. 
                                 Allows string replacement through another argument and the string.format() method.
                                 However, this may block regular expressions from being hardcoded into the query. 
                                 In this case, generate the string within a script variable and then use a bind to insert the expression into the query.
        @param args_list <list>: List of dictionaries (if bind) or lists (if positional) variables and their values for replacement.
        @param args_ts   <list>: In kwargs - List of bind variables that must have their cursor input size set to timestamp. [NOT COMPATIBLE WITH POSITIONAL ARGUMENTS]
        @param edit      <list>: List of direct string replacement parameters in positional order.
        @param do_raise  <bool>: Boolean flag to suppress errors.
        """
        if self.client is None:
            raise ValueError('Snowflake client is None')
            return
        try:
            data = None
            if isinstance(args, list):
                if len(args) == 0:
                    return
                if isinstance(args[0], list):
                    data = args
                else:
                    df = pd.DataFrame(args)
            if isinstance(args, dict):
                df = pd.DataFrame(args)
            else:
                df = args
            if data is None:
                for c in df.columns:
                    try:
                        df[c] = df[c].replace({np.nan: None})
                    except TypeError:
                        pass
                    try:
                        df[c] = df[c].replace({'None' : None})
                        df[c] = df[c].replace({'nan' : None})
                        df[c] = df[c].replace({'NaT' : None})
                    except TypeError:
                        pass
                data = df.values.tolist()
            if len(data) == 0:
                return
            rec0 = data[0]
            tsidx = [1 if isinstance(v, pd.Timestamp) else 0 for v in rec0]
            data1 = [] 

            for rec in data:
                data1.append([None if pd.isnull(v) else datetime(v.year, v.month, v.day, v.hour, v.minute, v.second) if tsidx[i] == 1 else v for i, v in enumerate(rec)])

            cur = self.client.cursor()
            cur.executemany(sql.format(*edit), data1)
            cur.close()
        except Exception as e1:
            try:
                sql = 'ALTER WAREHOUSE {} RESUME IF SUSPENDED'.format(self.info['warehouse'])
                cur.execute(sql)
                cur.executemany(sql.format(*edit), args)
                cur.close()
            except Exception as e2:
                cur.close()
                self._error_handler_(e1, False)
                self._error_handler_(e2, do_raise)

class Snowflake(SnowflakeConnection):

    """
    Quicker shorthand for constructing the connection object. Has unique string representations.
    """
    def __repr__(self):
        return 'advancedanalytics_util.Snowflake(%r, %r)' % (self.info)

class SF(SnowflakeConnection):

    """
    Quicker shorthand for constructing the connection object. Has unique string representations.
    """
    def __repr__(self):
        return 'advancedanalytics_util.SF(%r, %r)' % (self.info)
