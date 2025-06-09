# tyee.dataset.io

`tyee.dataset.io` 模块提供了一套统一且高效的输入输出接口，专用于存储和管理各类生理信号数据集的预处理结果。它通过将数据分段存储，并结合元信息索引，实现了对大规模生理数据的便捷访问和高效利用。该模块的核心组件包括 `PhysioSignalIO` (负责实际信号数据的读写) 和 `MetaInfoIO` (负责样本描述信息及索引的管理)。

## PhysioSignalIO

~~~python
class tyee.dataset.io.PhysioSignalIO(
	io_path: str,
    io_size: int = 1048576,
    io_mode: str = 'hdf5',
    io_chunks: int = None
)
~~~

一个通用高效的信号IO接口，用于将各种类型的生理信号数据集转换成数据段并将其存储在数据库中。

**使用样例**

~~~python
io = PhysioSignalIO('PATH')
eeg_signals = {
    'data': np.random.randn(32,128)
    'freq': 200.0
    'channels': [i for i in range(0,32)]
}
key = io.write_signals(eeg_signals, signal_type='eeg')
eeg_signals = io.read_signals(signal_type='eeg', key=key, start=0, end=64)
eeg_signals['data'].shape
>>> (32,64)
~~~



**参数**

- `io_path(str)`：数据库的存储位置
- `io_size(int)`：数据库的初始容量。当存储的数据超过此容量时，数据库大小会自动再增加一个 `io_size` 的量。
- `io_mode (str)`：信号数据的存储模式。支持以下选项：
  - `'hdf5'` (默认)：Tyee 将以 HDF5 格式进行数据读写。
  - `'lmdb'`：基于 LMDB 的信号存储。
  - `'pickle'`：基于文件系统的 Pickle 序列化存储。
  - `'memory'`：基于内存的临时存储。
- `io_chunks(int)`：数据库的分块大小。可指定分块大小，当使用hdf5时会给予分块大小进行保存。

**方法**

~~~python
signal_types()->list
~~~

获取该数据库所有的信号类型

返回：

- list，所有信号类型的列表

~~~python
keys()->list
~~~

获取该数据库所有的键

返回： 

- list, 所有键的列表

~~~python
signals(signal_type: str)->list
~~~

获取指定信号类型的所有信号数据

参数：

- `signal_type(str)`：指定的信号类型

返回： 

- list, 包含某信号的所有信号数据的列表

~~~python
read_signals(signal_type:str, key:str, start:int, end:int)->dict
~~~

获取指定信号类型的单个样本数据

参数：

- `signal_type(str)`：指定的信号类型
- `key(str)`：数据库中对应数据段的键
- `start(int)`： 样本在该数据段的起始索引
- `end(int)`： 样本在该数据段的终止索引

返回：

- dict, 包含样本数据、采样率、通道列表等信息的字典,如下所示：
  - `data(numpy array)`: 样本数据
  - `freq(float)`：采样率
  - `channels(list)`：通道列表 

~~~python
write_signals(signal:dict, signal_type:str, key:str)->str
~~~

写入指定信号类型的数据段

参数：

- `signal(dict)`：包含数据、采样率、通道列表等信号数据的字典

- `signal_type(str)`：信号的类型

- `key(str)`：信号的键。如果为None则自动生成键

返回：

- str, 写入信号的键

## MetaInfoIO

~~~python
class tyee.dataset.io.MetaInfoIO(io_path: str)
~~~

配合tyee.dataset.io.PhysioSignalIO进行使用，以表格的形式存储各种数据集的描述信息，以及其中每个信号类型的样本与存储数据段的索引映射，以便用户在数据库生成后可以分析和使用。

**使用样例**

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



**参数**

- `io_path(str)`：表格存储的位置

**方法**

~~~python
write_info(obj:dict)-> int
~~~

向表格中插入单个样本的描述信息

参数：

- `obj(dict)`：要写入表格的描述信息，包含信号的元数据字段，如样本id，数据段id，受试者id，会话id，试验id等

返回： 

- int, 写入的描述信息在表格中的索引

~~~python
read_info(key:int)-> pandas.DataFrame
~~~

根据索引查询对应样本的生理信号描述信息

参数：

- `key(int)`：要查询的样本描述信息的索引

返回： 

- pandas.DataFrame， 对应样本的描述信息

~~~python
read_all()-> pandas.DataFrame
~~~

获取表格中所有样本的描述信息

返回：

- pandas.DataFrame, 所有样本的描述信息