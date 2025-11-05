# 第173期 [实战项目]基于Python和LSTM的人工智能股市预测
![b2baN4](https://raw.githubusercontent.com/errolyan/tuchuang/master/uPic/b2baN4.png)


## 引言：利用人工智能预测股市
股票价格波动频繁，人工预测几乎难以实现。而人工智能与深度学习的出现改变了这一局面。借助长短期记忆（LSTM）网络，你可以对时间序列数据进行建模，从而预测股票趋势。



我个人曾构建过能准确预测短期趋势的人工智能交易模型。LSTM网络作为循环神经网络的一种，非常适合处理股票价格这类序列数据。结合Python的机器学习库，你可以打造出可投入实际生产的预测系统。

实用提示：人工智能交易的成功关键在于系统化和纪律性——切勿依赖直觉判断。


## 步骤1：搭建环境
执行以下命令安装所需库：
```
pip install numpy pandas matplotlib yfinance scikit-learn tensorflow keras
```
- TensorFlow/Keras——深度学习框架
- YFinance——获取历史股票数据
- Scikit-learn——数据预处理与模型评估
- Matplotlib——数据可视化


## 步骤2：获取股票数据
```python
import yfinance as yf

symbol = "AAPL"  # 股票代码，此处以苹果公司为例
data = yf.download(symbol, period="5y", interval="1d")  # 获取5年日度数据
data = data[["Open", "High", "Low", "Close", "Volume"]]  # 保留开盘价、最高价、最低价、收盘价和成交量
print(data.tail())  # 查看数据的最后几行
```
历史数据是训练LSTM模型的关键要素，你也可以将股票代码替换为任意股票或加密资产的代码。


## 步骤3：数据预处理
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 将收盘价数据归一化到[0,1]区间
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

# 划分训练集与测试集（8:2比例）
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 构建LSTM输入序列（时间步长设为60，即利用前60天数据预测第61天价格）
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])  # 输入：前60天数据
        y.append(dataset[i + time_step, 0])    # 输出：第61天数据
    return np.array(X), np.array(y)

# 生成训练集和测试集的输入输出序列
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# 调整数据形状以适配LSTM模型（样本数，时间步长，特征数）
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
```
对股票价格进行归一化处理可提升LSTM模型的学习效果，而构建序列数据则是为了满足LSTM对输入格式的要求。


## 步骤4：构建LSTM模型
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 初始化序贯模型
model = Sequential()
# 第一层LSTM（50个神经元，返回序列以连接下一层LSTM）
model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
model.add(Dropout(0.2))  # 防止过拟合，随机丢弃20%的神经元
# 第二层LSTM（50个神经元，不返回序列）
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))  # 再次加入Dropout层
# 全连接层（25个神经元）
model.add(Dense(25))
# 输出层（1个神经元，预测次日收盘价）
model.add(Dense(1))

# 编译模型（优化器为adam，损失函数为均方误差）
model.compile(optimizer="adam", loss="mean_squared_error")
# 查看模型结构摘要
model.summary()
```
LSTM层的核心作用是捕捉数据中的时间依赖关系，Dropout层用于防止模型过拟合，而全连接层（Dense层）则负责输出最终的预测股票价格。


## 步骤5：训练模型
```python
# 训练模型（批量大小64，迭代次数50，使用测试集作为验证数据）
history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test))
```
迭代次数（epochs）和批量大小（batch size）是影响训练效率的关键参数，而验证数据的作用是确保模型在未见过的数据上也能保持稳健的预测性能。


## 步骤6：进行预测
```python
# 用测试集数据进行预测
predictions = model.predict(X_test)
# 将预测结果反归一化，还原为真实股票价格尺度
predictions = scaler.inverse_transform(predictions)
# 将测试集真实值反归一化
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))
```
由于之前对数据进行了归一化处理，此处需要通过反归一化操作，将预测结果和真实值还原为实际的股票价格，以便后续对比分析。


## 步骤7：结果可视化
```python
import matplotlib.pyplot as plt

# 设置图表大小
plt.figure(figsize=(12,6))
# 绘制真实价格曲线
plt.plot(y_test_actual, label="实际价格")
# 绘制预测价格曲线
plt.plot(predictions, label="预测价格")
# 添加图表标题（包含股票代码）
plt.title(f"{symbol}股票价格预测")
# 添加坐标轴标签
plt.xlabel("时间")
plt.ylabel("价格")
# 显示图例
plt.legend()
# 展示图表
plt.show()
```
通过可视化对比实际价格与预测价格的走势，既能直观评估模型的预测准确性，也能根据趋势变化识别潜在的买卖时机。


## 步骤8：实时预测
```python
# 获取最新60天的股票数据（用于预测次日收盘价）
latest_data = yf.download(symbol, period="60d", interval="1d")
# 对最新收盘价数据进行归一化
scaled_latest = scaler.transform(latest_data["Close"].values.reshape(-1,1))
# 调整数据形状以适配模型输入
X_input = np.array([scaled_latest[:,0]])
X_input = X_input.reshape(1, X_input.shape[1], 1)

# 预测次日收盘价
next_day_prediction = model.predict(X_input)
# 反归一化得到真实价格
next_day_price = scaler.inverse_transform(next_day_prediction)
# 打印预测结果
print(f"预测次日收盘价：{next_day_price[0][0]}")
```
该步骤可实现次日收盘价的实时预测，其结果可作为人工智能交易信号的重要参考依据。


## 步骤9：结合FastAPI部署API
```python
from fastapi import FastAPI
from pydantic import BaseModel

# 初始化FastAPI应用
app = FastAPI()

# 定义请求数据模型（需传入股票代码）
class StockInput(BaseModel):
    symbol: str

# 定义POST接口，用于接收请求并返回预测结果
@app.post("/predict/")
def predict_stock(data: StockInput):
    symbol = data.symbol  # 获取请求中的股票代码
    # 获取该股票最新60天数据
    latest_data = yf.download(symbol, period="60d", interval="1d")
    # 归一化处理
    scaled_latest = scaler.transform(latest_data["Close"].values.reshape(-1,1))
    # 调整输入形状
    X_input = scaled_latest.reshape(1, scaled_latest.shape[0],1)
    # 预测
    prediction = model.predict(X_input)
    # 反归一化
    price = scaler.inverse_transform(prediction)
    # 返回预测价格（以浮点数格式）
    return {"predicted_price": float(price[0][0])}
```
通过FastAPI部署API后，可获得实时预测的接口服务，该接口可进一步与数据仪表盘或交易机器人集成，实现自动化应用。


## 步骤10：模型回测
```python
import pandas as pd  # 需提前导入pandas库

# 创建包含实际值和预测值的DataFrame
test_results = pd.DataFrame({
    "实际价格": y_test_actual.flatten(),
    "预测价格": predictions.flatten()
})
# 计算预测价格的日收益率
test_results["日收益率"] = test_results["预测价格"].pct_change()
# 计算累计收益率
test_results["累计收益率"] = (1 + test_results["日收益率"]).cumprod()
# 查看结果的最后几行
print(test_results.tail())
```
回测的核心目的是评估基于模型预测进行交易可能获得的潜在收益，同时也可通过回测结果调整模型参数和特征，进一步优化模型性能。


## 步骤11：提升模型性能
- 增加特征维度：可引入成交量（Volume）、简单移动平均线（SMA）、指数移动平均线（EMA）、布林带（Bollinger Bands）等金融指标
- 采用更先进的模型架构：如门控循环单元（GRU）、双向LSTM（Bi-LSTM）、基于Transformer的模型等
- 超参数调优：对学习率、迭代次数、批量大小等关键超参数进行优化调整

实用提示：即使是微小的特征工程改进，也可能显著提升LSTM模型的预测效果。


## 步骤12：实际应用场景
- 人工智能驱动的股票交易机器人
- 加密货币市场预测仪表盘
- 投资组合管理人工智能助手
- 金融研究与模型回测
- 市场异常情况投资者预警系统