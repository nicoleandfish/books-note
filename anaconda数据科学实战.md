# 2 
spyder
julia
octave
 - pip install octave_kernel

# 3 数据基础
## 3.1 数据源
一般来说，有两种类型的数据，公共数据和私有数据。私有或专有的数据库相当昂贵，例如证券价格研究中心（Center for Research in Security Prices, CRSP）数据库，由芝加哥大学制作并维护的一个金融数据库。公共或免费数据哭，例如，UCI提供许多有用的数据集。
## 3.2 UCI机器学习
公共数据源举例
名称|数据类型
---|---
UCI|用于机器学习的数据
世界卫生组织|医疗保健数据
亚马逊Web服务|Web应用
Data.gov（美国政府公开数据）|农业、气候、消费者等
GitHub|个人提供的许多数据集
开放数据网络|开放的数据集
政府健康数据|医疗保健相关数据集
谷歌公共数据|世界发展指标

经济学领域的开放数据来源
名称|数据类型
美联储经济数据|利率和AAA/AA级债券的利率
美国人口统计局|人口普查数据
美国财政部|美国财政收益
美国劳工统计局|通货膨胀、就业、失业、工资和福利等
美国经济分析局|GDP、个人收入支出、企业利润等
美国国家经济调查局|商业周期、重要统计、美国总统报告等

金融和会计领域开放数据来源
名称|数据类型
雅虎财经|历史定价数据、期权、资产负债表和损益表
谷歌财经|当前和历史交易价格
市场观察|财务报表
美国证券交易委员会归档文件|资产负债表、损益表、持有量等
French教授的数据库|Fama/French因子、市场指数、无风险利润、行业分类等

http://canisius.edu/~yany/data/ff2Monthly.csv

## 3.4 输入数据
### 3.4.1 使用R
```
>path<-"http://archive.ics.uci.edu/ml/machine-learning-databases/"
>dataSet<-"iris/bezdekIris.data"
>a<-paste(path,dataSet,sep='')
>x<-read.csv(a,header=F)
>colnames(x)<-c("sepalLength","sepalWidth","petalLength","petalWidth","Class")
```

dim()查看数据集大小、head()查看前几行观察值、tail()查看最后几行观察值。
read.table()、read.delim()、read.fwf()可以从外部数据文件来输入数据。可以使用apropos()来查看以read.开头的函数。>apropos(read.)

### 3.4.2 使用Python
```
import pandas as pd
path="http://archive.ics.uci.edu/ml/machine-learning-databases/"
dataset="iris/bezdekIris.data"
inFile=path+dataset
data=pd.read_csv(inFile,header=None)
data.columns=["sepalLength","sepalWidth","petalLength","petalWidth","Class"]
#查看前两个实例
print(data.head(2))
```
pandas中提取数据的函数
函数|描述
---|---
read_clipboard|从剪切板读取文本并传送到read_table
read_csv|将CSV文件读入DataFrame
read_excel|将Excel表读入panda DataFrame
read_feather|从文件路径加载feather格式对象
read_fwf|将固定宽度的格式化行表读入DataFrame
read_gbq|从谷歌BigQuery加载数据
read_hdf|检索存储在文件中的panda对象
read_html|将HTML表读入DataFrame对象列表
read_json|将JSON字符串转换为panda对象
read_msgpack|从指定的文件路径加载pandas msgpack对象
read_pickle|从指定文件路径加载一个序列化的pandas文件（或任何其他序列化对象）
read_sas|读取存储为XPORT或SAS7DBAT格式文件的SAS文件
read_sql|将SQL查询或数据库表读入DataFrame
read_sql_query|将SQL查询读入DataFrame
read_sql_table|将SQL数据库表读入DataFrame
read_stata|将Stata文件读入DataFrame
read_table|将带分隔符的普通文件读入DataFrame

## 3.5 Quandl数据分发平台简介
Quandl是包含多个免费数据集的数据分发平台。
```
>library(Quandl)
>x<-Quandl.dataset.get("WIKI/ibm",list(rows=50))
>head(x,2)

>y<-Quandl.dataset.get("DY4/601628",list(rows=50))
>head(y,2)

#获取Quandl应用程序密钥
>library(Quandl)
>Quandl.api_key(myQuandlKey)
>x<-Quandl.dataset.get("WIKI/wmt",list(rows=50))
```
python
```
import quandl
x=quandl.get("FRED/GDP")
x.head()
#上述返回的是pandas数据帧，可以自定义数据类型
import quandl
x=quandl.get("FRED/GDP",returns="pandas")
x=quandl.get("FRED/GDP",returns="numpy")
```
# 5 在anaconda中统计建模
最简单的统计模型：单因子线性模型（the one-factor linear model)，此模型的应用：著名的金融模型资本资产定价模型（capital asset pricing model,CAPM)
