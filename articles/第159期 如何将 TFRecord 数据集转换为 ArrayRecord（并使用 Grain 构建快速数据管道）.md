# 第159期 如何将 TFRecord 数据集转换为 ArrayRecord（并使用 Grain 构建快速数据管道）



若你曾使用 GPU 或 TPU 训练过模型，一定体会过这种 frustration：加速器速度很快，但数据加载器却慢吞吞的。

模型在等待。
GPU 在“叹气”。
而你只能眼睁睁看着训练进度缓慢爬行。

解决方案是什么？答案是 **ArrayRecord 与 Grain 结合使用**。

谷歌悄然开发了这两款工具，旨在让数据管道更快速、更简洁——尤其适用于大规模训练场景。在本文中，我将清晰地讲解其用法（当然，还会附上代码）。


![](https://fastly.jsdelivr.net/gh/bucketio/img14@main/2025/10/24/1761317033913-951af6e4-d161-4a39-815a-de95804e6d24.png)


图片由 Unsplash 平台上的 Claudio Schwarz 提供
## 到底什么是 ArrayRecord？
可以将 ArrayRecord 视为一种新的数据集存储方式——它与 TFRecord 类似，但在大规模数据读取场景下速度显著更快。该格式专为大规模语言模型（LLM）训练、密集型计算机视觉管道等工作负载设计。

若把 TFRecord 比作可靠的老式卡车，那么 ArrayRecord 就是电动跑车版本——用途相同，但针对速度进行了优化。

## 方案1：转换标准 TFDS 数据集（如 CIFAR-10）
如果你使用的是 TensorFlow Datasets（TFDS）中的数据集（如 CIFAR-10、ImageNet 等），转换过程会非常简单。

首先安装 TensorFlow Datasets：
```bash
pip install -q --upgrade tfds-nightly
```
然后运行以下命令：
```bash
# 将 CIFAR-10 转换为 ArrayRecord 格式
tfds build cifar10 --file_format=array_record
```
操作到此结束。
你可以在 `~/tensorflow_datasets/` 目录下找到生成的 ArrayRecord 文件。
整个过程无需复杂脚本，也无需手动转换。

## 方案2：转换自定义 TFRecord 数据集（可扩展方法）
如果你的数据集是自定义的 TFRecord 格式，那么应该结合使用 Apache Beam 和 ArrayRecord Beam SDK。

这种组合支持跨多个工作节点处理大型数据集（当然，也可以在 Google Cloud Dataflow 上运行）。

### 步骤1：安装依赖库
```bash
pip install -q apache-beam
pip install -q array-record-beam-sdk
```

### 步骤2：转换脚本
以下是一个简洁、独立的脚本，可完成所有转换工作：
```python
from apache_beam.options import pipeline_options
from array_record.beam.pipelines import convert_tf_to_arrayrecord_disk_match_shards

# 输入路径（TFRecord 文件）和输出路径（ArrayRecord 文件）
input_pattern = 'gs://your-bucket/data/records-*.tfrecord'
output_path = 'gs://your-bucket/converted/records'
args = {'input': input_pattern, 'output': output_path}

# 配置管道选项
local_pipeline_options = pipeline_options.PipelineOptions()

def main():
    print("开始 TFRecord → ArrayRecord 转换...")
    # 执行转换
    convert_tf_to_arrayrecord_disk_match_shards(
        args=args,
        pipeline_options=local_pipeline_options,
    ).run()
    print(f"转换完成！文件已保存至 {output_path}")

if __name__ == "__main__":
    main()
```

**提示**：对于超大型数据集，可将运行器切换为 DataflowRunner，让 Google Cloud 处理繁重的计算工作。

## 使用 Grain 构建数据管道
既然数据已转换为 ArrayRecord 格式，接下来就可以将其输入到 Grain（谷歌推出的快速且灵活的数据集 API）中。

### 步骤1：加载数据集
```python
import grain

# ArrayRecord 文件路径（以 CIFAR-10 为例）
file_paths = ["~/tensorflow_datasets/cifar10/3.0.2/cifar10-train.array_record-00000-of-00001"]
# 创建数据源
data_source = grain.sources.ArrayRecordDataSource(file_paths)
# 构建基础数据集
dataset = grain.MapDataset.source(data_source)
```
这就是基础数据集——接下来我们可以对其进行转换操作。

### 步骤2：添加转换操作（打乱、映射、批处理）
```python
def parse_and_transform(record):
    # 在此处对数据进行解码和处理
    return {"record": record}

# 批处理大小
BATCH_SIZE = 32

# 链式调用转换操作
dataset = (
    dataset.shuffle(seed=42)  # 打乱数据，保证随机性
           .map(parse_and_transform)  # 应用自定义数据处理逻辑
           .batch(batch_size=BATCH_SIZE, drop_remainder=True)  # 按批次划分，丢弃不足一批的数据
)
```
每个转换步骤都会返回一个新的数据集，因此可以灵活地链式调用——类似 PyTorch 的 DataLoader，但专为大规模场景设计。

### 步骤3：迭代读取数据
```python
# 创建数据迭代器
data_iterator = iter(dataset)

# 遍历批次数据
for batch in data_iterator:
    # 在此处执行训练步骤
    pass
```
操作并不复杂，既简洁又高效。

## 让速度再快一点（大幅提升）
这部分内容会很有意思。如果希望训练步骤之间无延迟，可以使用多进程预取（multiprocessing prefetching）。

Grain 用一行代码就能实现这一功能：
```python
dataset = (
    grain.MapDataset.source(data_source)
    .shuffle(seed=42)  # 打乱数据
    .map(parse_and_transform)  # 数据处理
    .to_iter_dataset()  # 转换为可迭代数据集
    .batch(batch_size=BATCH_SIZE, drop_remainder=True)  # 批处理
    .mp_prefetch(
        # 配置多进程选项
        grain.multiprocessing.MultiprocessingOptions(
            num_workers=16  # 若 CPU 性能允许，可增加工作节点数量
        )
    )
)
```
现在，CPU 可以在 GPU 训练的同时加载数据——无需等待，也不会卡顿。
如果感觉 GPU 处于“空闲”状态，只需增加 `num_workers` 的值，你会立即感受到速度的变化。

## 想深入了解更多？
以下是值得收藏的官方文档：
- Grain 文档
- ArrayRecord GitHub 仓库
- Apache Beam 文档
- TensorFlow Datasets 文档

如果你想了解谷歌如何在大规模 LLM 训练中使用这套工具，可以查看 MaxText——这是一个基于 JAX 的开源模型训练系统，其数据输入环节就用到了 Grain 和 ArrayRecord。
- MaxText GitHub 仓库
- MaxText 数据管道指南

## 总结思考
数据管道往往是训练流程的瓶颈，而非模型本身。GPU 处理数据的速度其实比你提供数据的速度更快——除非你搭建了高效的数据管道。

因此，如果你仍在使用传统的 TFRecord 格式，且未结合预取或多进程技术，那么你的训练效率就如同“拉着手刹赛跑”。