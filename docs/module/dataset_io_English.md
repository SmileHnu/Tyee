# tyee.dataset.io

The `tyee.dataset.io` module provides a unified and efficient input/output interface, specifically designed for storing and managing the preprocessed results of various physiological signal datasets. It achieves convenient access and efficient utilization of large-scale physiological data by segmenting data storage and integrating it with metadata indexing. The core components of this module include `PhysioSignalIO` (responsible for reading and writing actual signal data) and `MetaInfoIO` (responsible for managing sample descriptive information and indexing).

## PhysioSignalIO

~~~python
class tyee.dataset.io.PhysioSignalIO(
	io_path: str,
    io_size: int = 1048576,
    io_mode: str = 'hdf5',
    io_chunks: int = None
)
~~~

A general-purpose and efficient signal I/O interface for converting various types of physiological signal datasets into data segments and storing them in a database.

**Usage Example**

~~~python
io = PhysioSignalIO('PATH')
eeg_signals = {
    'data': np.random.randn(32,128),
    'freq': 200.0,
    'channels': [i for i in range(0,32)]
}
key = io.write_signals(eeg_signals, signal_type='eeg')
eeg_signals = io.read_signals(signal_type='eeg', key=key, start=0, end=64)
eeg_signals['data'].shape
>>> (32,64)
~~~

**Parameters**

- `io_path (str)`: Storage location of the database.
- `io_size (int)`: Maximum capacity of the database. Initially set to `io_size`; when the capacity is exceeded, it increases by another `io_size`.
- `io_mode (str)`: Storage mode for signal data. The following options are supported:
  - `'hdf5'` (default): Tyee will read and write data in HDF5 format.
  - `'lmdb'`: LMDB-based signal storage.
  - `'pickle'`: File system-based Pickle serialized storage.
  - `'memory'`: Memory-based temporary storage.
- `io_chunks (int)`: Chunk size for the database. Allows specifying the chunk size; when using HDF5, data will be saved with this chunk size.

**Methods**

~~~python
signal_types()->list
~~~

Gets all signal types in this database.

Returns:

- list, a list of all signal types.

~~~python
keys()->list
~~~

Gets all keys in this database.

Returns:

- list, a list of all keys.

~~~python
signals(signal_type: str)->list
~~~

Gets all signal data for the specified signal type.

Parameters:

- `signal_type (str)`: The specified signal type.

Returns:

- list, a list containing all signal data for a given signal type.

~~~python
read_signals(signal_type:str, key:str, start:int, end:int)->dict
~~~

Gets a single sample's data for the specified signal type.

Parameters:

- `signal_type (str)`: The specified signal type.
- `key (str)`: The key corresponding to the data segment in the database.
- `start (int)`: The starting index of the sample within that data segment.
- `end (int)`: The ending index of the sample within that data segment.

Returns:

- dict, a dictionary containing information such as sample data, sampling rate, channel list, as shown below:
  - `data (numpy array)`: Sample data.
  - `freq (float)`: Sampling rate.
  - `channels (list)`: Channel list.

~~~python
write_signals(signal:dict, signal_type:str, key:str)->str
~~~

Writes a data segment of the specified signal type.

Parameters:

- `signal (dict)`: A dictionary containing signal data such as data, sampling rate, channel list.
- `signal_type (str)`: The type of the signal.
- `key (str)`: The key for the signal. If None, a key is automatically generated.

Returns:

- str, the key of the written signal.

## MetaInfoIO

~~~python
class tyee.dataset.io.MetaInfoIO(io_path: str)
~~~

Used in conjunction with `tyee.dataset.io.PhysioSignalIO` to store descriptive information for various datasets in tabular form, as well as the index mapping between samples of each signal type and their stored data segments, allowing users to analyze and use the database after its generation.

**Usage Example**

~~~python
io = MetaInfoIO('PATH')
key = io.write_info({
    'sample_id':0,
    'segment_id':0,
    'subject_id':0,
    'session_id':0,
    'trial_id':0,
})
info = io.read_info(key).to_dict()
>>>{
    'sample_id':0,
    'segment_id':0,
    'subject_id':0,
    'session_id':0,
    'trial_id':0,
}
~~~

**Parameters**

- `io_path (str)`: Storage location of the table.

**Methods**

~~~python
write_info(obj:dict)-> int
~~~

Inserts descriptive information for a single sample into the table.

Parameters:

- `obj (dict)`: Descriptive information to be written to the table, including metadata fields of the signal, such as sample ID, segment ID, subject ID, session ID, trial ID, etc..

Returns:

- int, the index of the written descriptive information in the table.

~~~python
read_info(key:int)-> pandas.DataFrame
~~~

Queries the descriptive information of physiological signals for the corresponding sample based on the index.

Parameters:

- `key (int)`: The index of the sample descriptive information to query.

Returns:

- pandas.DataFrame, descriptive information for the corresponding sample.

~~~python
read_all()-> pandas.DataFrame
~~~

Gets descriptive information for all samples in the table.

Returns:

- pandas.DataFrame, descriptive information for all samples.