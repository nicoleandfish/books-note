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

Quandl数据平台上不同数据源的模式
模式|例子|说明
---|---|---
WIKI|WIKI/IBM|美国股市收盘价
DY4|DY4/60618|中国数据
FRED|FRED/GDP|美联储经济数据
EIA|EIA/PET_RWTC_D|美国能源部
NSE|NSE/OIL|石油价格
ZEP|ZEP/ibm|Zacks股票价格、股息历史（美国）
CBOE|CBOE/VXU2018|芝加哥期权交易所

```
import quandll as qd
quandl.ApiConfig.api_key='YOUR_API_KEY'
y.qd.get("WIKI/WMT")
y.head(2)
y.tail(1)
```
## 3.6处理缺失数据
语言|用于表示缺失的代码|说明或例子
---|---|---
R|NA|NA代表不可用
Python|nan|import scipy as sp /n misingCode=sp.nan
Julia|missing|julia>missing+5 /n missing
Octave|NaN|和Matlab相同

R
```
set.seed(123)
n=500
x<-rnorm(n)
x2<-x
m=100
y<-as.integer(runif(m)*n)
x[y]<-0
z<-matrix(x,n/5,5)
outFile<-"c:/temp/z.csv"
write.table(z,file=outFile,quote=F,row.names=F,col.names=F,sep=',')
```
python
```
import scipy as sp
import pandas as pd
path="http://canisius.edu/~yany/data/"
dataSet="z.csv"
infile=path+dataset
#infile="c:/temp/z.csv"
x=pd.read_csv(infile,header=None)
print(x.head())
print((x[[1,2,3,4,5]] == 0).sum())

#使用scipy.NaN来替换0
x2=x
x2[[1,2,3,4,5]] =x2[[1,2,3,4,5]].replace(0.sp.NaN)
print(x2.head())

#使用平均值来替换NaN
x3=x2
x3.fillna(x3.mean(),inplace=True)
print(x3.head())
```

## 3.7数据排序
R
```
> set.seed(123)
> x<-rnorm(100)
> head(x)
> y<-sort(x)
> head(y)

#安装包命令
install.packages('fImport')

#另一种排序方法，nyseListing是数据集，在fImport包里
> library(fImport)
> data(nyseListing)
> dim(nyseListing)
> head(nyseListing)
#按名字排序
> x<-nyseListing[order(nyseListing$Name),]
> head(x)
#先按ID再按RET排序
> x<-c(1,3,1,0.1,0.3,-0.4,100,300,30)
> y<-data.frame(matrix(x,3,3))
> colnames(y)<-c("ID","RET","Data1")
> y
  ID  RET Data1
1  1  0.1   100
2  3  0.3   300
3  1 -0.4    30

#可以使用order
> z<-y[order(y$ID,y$RET),]
> z
  ID  RET Data1
3  1 -0.4    30
1  1  0.1   100
2  3  0.3   300

#使用decreasing进行降序排列
> z2<-y[order(y$ID,decreasing=TRUE,y$RET),]
> z2
  ID  RET Data1
2  3  0.3   300
1  1  0.1   100
3  1 -0.4    30
```

python
```
import pandas as pd
a = pd.DataFrame([[8,3],[8,2],[1,-1]],columns=['X','Y'])
print(a)
#sort by A ascending,then B descending
b=a.sort_values(['X','Y'],ascending=[1,0])
print(b)
#sort by A and B, both ascending
c=a.sort_values(['X','Y'],ascending=[1,1])
print(c)
```

### 3.7.1切割数据集
R
```
> con<-url("http://canisius.edu/~yany/RData/marketCap.RData")
> load(con)
> head(.marketCap)

```
## 3.8 cbsodata
数据集的API-cbsodata，问题，安装了依旧无法导入。
datadotworld也安装不成功。

'''
> library(haven)
> library(foreign)
> x<-1:100
> y<-matrix(x,50,2)
> z<-data.frame(y)
> colnames(z)<-c("a","b")
> write_sas(z,"/Users/Nicole/python/tt.sas7dbat")
#write_spss和write_stata不成功
> write_spss(z,"/Users/Nicole/python/tt.sav")
> write_stata(z,"/Users/Nicole/python/tt.dta")
'''

## 3.11 R包dslabs
R包dslabs是数据科学实验室（Data Science Labs）的简称。
```
> install.packages("dslabs")
> library(dslabs)
> data(murders)
> head(murders)
       state abb region population total
1    Alabama  AL  South    4779736   135
2     Alaska  AK   West     710231    19
3    Arizona  AZ   West    6392017   232
4   Arkansas  AR  South    2915918    93
5 California  CA   West   37253956  1257
6   Colorado  CO   West    5029196    65
```

dslabs中包含的数据集
数据集名称|说明
---|---
admissions|加州大学伯克利分校研究生入学的性别偏差
divorce_margarine|离婚率和人造黄油消费数据
ds_theme_set|dslabs主题集
gapminder|Gapminder数据
heights|身高数据
murders|2010年美国各州的枪杀案数据
na_example|含有缺失值的计数数据
oecd|Gapminder数据
opec|Gapminder数据
outlier_example|包含异常值、以英尺为单位的成年男性身高
poll_us_election_2016|2016年Fivethirtyeight发布的民调数据
raw_data_research_funding_rates|荷兰研究经费中的性别偏差（原始数据）
reported_heights|自报告的身高
research_funding_rates|荷兰研究经费中的性别偏差
results_us_election_2016|2016年Fivethirtyeight民调数据处理结果
rfalling_object|模拟落体数据
take_poll|通过民意测验得出的模型结果
us_contagious_diseases|美国各州传染病数据

## 3.12 生成Python的数据集
使用pandas的to_pickle函数。

## 3.13 生成R数据集
R
```
> path<-"http://archive.ics.uci.edu/ml/machine-learning-databases/"
> dataSet<-"iris/bezdekIris.data"
> a<-paste(path,dataSet,sep="")
> iris<-read.csv(a,header=F)
> colnames(iris)<-c("sepalLength","sepalWidth","petalLength","petalWidth","Class")
> save(iris,file="/Users/Nicole/python/iris.RData")
> load("/Users/Nicole/python/iris.RData")
> head(iris)
  sepalLength sepalWidth petalLength petalWidth       Class
1         5.1        3.5         1.4        0.2 Iris-setosa
2         4.9        3.0         1.4        0.2 Iris-setosa
3         4.7        3.2         1.3        0.2 Iris-setosa
4         4.6        3.1         1.5        0.2 Iris-setosa
5         5.0        3.6         1.4        0.2 Iris-setosa
6         5.4        3.9         1.7        0.4 Iris-setosa
```
# 4 可视化
## 4.2 R数据可视化
```
> plot(cos,-2*pi,2*pi)
#直方图
> set.seed(333)
> data<-rnorm(5000)
> hist(data)

> x<-seq(-3,3,length=100)
> y<-dnorm(x,mean=0,sd=1)
> title<-"Area under standard normal dist & x less than -2.33"
> plot(x,y,lwd=3,col="black",main=title,xlab=xLebal,ylab=yLebal)
> plot(x,y,type="l",lwd=3,col="black",main=title,xlab=xLebal,ylab=yLebal)
> x<-seq(-3,-2.33,length=100)
> y<-dnorm(x,mean=0,sd=1)
> polygon(c(-4,x,-2.33),c(0,y,0),col="red")
Error in polygon(c(-4, x, -2.33), c(0, y, 0), col = "red") : 
  plot.new has not been called yet


> install.packages("rattle")
```

## 4.9 动态可视化Dynamic visualization
```
> library(animation)
> ani.options(interval=0.05,nmax=30)
animation option 'nmax' changed: 50 --> 30
> a<-c(3,3,2,0.5)
> b<-c(2,0.5,0)
> name<-"Brownian Motion"
> par(mar=a,mgp=b,tcl=-0.3,cex.axis=0.8,cex.lab=0.8,cex.main=1)
> brownian.motion(pch=21,cex=5,col="red",bg="yellow",main=name)
```
# 5 在anaconda中统计建模
最简单的统计模型：单因子线性模型（the one-factor linear model)，此模型的应用：著名的金融模型资本资产定价模型（cital asset pricing model,CAPM)
