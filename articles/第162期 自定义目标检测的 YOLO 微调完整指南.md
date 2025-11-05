 

# 第162期 自定义目标检测的 YOLO 微调完整指南

目标检测技术已在众多领域中变得至关重要，从自动驾驶到医学影像均有其身影。训练能够检测特定领域内目标的模型，可显著提升自动化分析与决策流程的效率。本文将带你完整了解如何针对自定义目标检测任务微调 YOLO（You Only Look Once，实时目标检测算法）模型。

## 一、引言

YOLO 系列模型凭借其速度与精度优势，已成为实时目标检测任务中最受欢迎的选择之一。本文将探讨如何在自定义数据集上微调 YOLO12（较新变体）或 YOLOv8，以检测与你的应用场景相关的目标。

本文将涵盖以下内容：

- • 数据集结构搭建
- • 数据验证与预处理
- • 采用优化超参数训练模型
- • 模型性能评估
- • 新图像测试

## 二、前置条件

若要跟随本文操作，你需要准备以下工具与知识：

- • Python 3.8 及以上版本
- • PyTorch 框架
- • Ultralytics YOLO 软件包
- • 带标注的图像数据集（含图像与标注文件）
- • 目标检测基础概念认知

## 三、数据集结构

进行 YOLO 模型训练时，需按以下结构组织数据：

```
  Dataset/
├── data.yaml        # 数据集配置文件
├── train/
│   ├── images/      # 训练集图像
│   └── labels/      # 训练集标注（YOLO 格式）
├── valid/
│   ├── images/      # 验证集图像
│   └── labels/      # 验证集标注
└── test/
    ├── images/      # 测试集图像
    └── labels/      # 测试集标注
```

其中，`data.yaml` 文件需包含以下内容：

```yaml
  train: /path/to/train/images  # 训练集图像路径
val: /path/to/val/images      # 验证集图像路径
test: /path/to/test/images    # 测试集图像路径
nc: 3                         # 类别数量
names: ['class1', 'class2', 'class3']  # 类别名称列表
```

## 四、核心代码模块

### 1. 数据集验证

```python
  def verify_dataset_structure():
    # 检查 data.yaml 是否存在，不存在则创建
    if not os.path.exists(data_yaml_path):
        train_images_path = os.path.join(train_path, "images")
        val_images_path = os.path.join(val_path, "images")
        test_images_path = os.path.join(test_path, "images")
        
        yaml_data = {
            'train': train_images_path,
            'val': val_images_path, 
            'test': test_images_path,
            'nc': len(class_names),  # 类别数量
            'names': class_names     # 类别名称
        }
        
        # 写入 yaml 文件
        with open(data_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
    
    # 统计并验证图像与标注文件数量
    train_images_dir = os.path.join(train_path, "images")
    train_labels_dir = os.path.join(train_path, "labels")
    
    # 统计训练集图像数量（支持 jpg、jpeg、png、bmp 格式）
    train_images = len([f for f in os.listdir(train_images_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    # 统计训练集标注文件数量（仅 txt 格式）
    train_labels = len([f for f in os.listdir(train_labels_dir) if f.endswith('.txt')])
    
    # 确保训练集存在图像与标注文件
    return train_images > 0 and train_labels > 0
```

### 2. 生成验证集

```python
  def create_validation_set(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir):
    # 创建验证集图像与标注文件夹（若不存在）
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # 获取训练集所有图像文件
    image_files = [f for f in os.listdir(train_images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # 固定随机种子以确保结果可复现，打乱图像文件顺序
    np.random.seed(42)
    np.random.shuffle(image_files)
    # 按 20% 比例划分验证集
    val_split = int(0.2 * len(image_files))
    val_files = image_files[:val_split]
    
    # 将选中的文件复制到验证集目录
    for img_file in val_files:
        # 复制图像文件
        src_img = os.path.join(train_images_dir, img_file)
        dst_img = os.path.join(val_images_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        # 复制对应标注文件（标注文件名与图像文件名一致，后缀为 txt）
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(train_labels_dir, label_file)
        dst_label = os.path.join(val_labels_dir, label_file)
        
        # 若标注文件存在，则复制
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
```

### 3. 模型训练

```python
  def train_yolo_model(epochs=50, batch_size=16, img_size=640, lr0=0.01):
    # 检查 CUDA 可用性（优先使用 GPU 训练）
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # 生成时间戳，用于模型文件唯一命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'train_{timestamp}'
    
    # 加载模型（优先加载 YOLO12n，失败则加载 YOLOv8n）
    try:
        model = YOLO('yolo12n.pt')
        model_type = 'yolo12n'
    except Exception:
        model = YOLO('yolov8n.pt')
        model_type = 'yolov8n'
    
    # 模型训练
    results = model.train(
        data=data_yaml_path,    # 数据集配置文件路径
        epochs=epochs,          # 训练轮次
        batch=batch_size,       # 批次大小
        imgsz=img_size,         # 输入图像尺寸
        patience=10,            # 早停机制（10 轮无提升则停止）
        save=True,              # 保存训练过程中的模型
        device=device,          # 训练设备（GPU/CPU）
        project=os.path.join(base_dir, 'runs'),  # 训练结果保存根目录
        name=run_name,          # 本次训练结果文件夹名称
        lr0=lr0,                # 初始学习率
        lrf=0.01,               # 最终学习率（为初始学习率的 1%）
        plots=True,             # 生成训练过程可视化图表
        save_period=5           # 每 5 轮保存一次模型
    )
    
    # 模型保存路径
    model_save_path = os.path.join(model_save_dir, f"{model_type}_{timestamp}.pt")
    
    # 尝试多种方式保存模型
    try:
        model.model.save(model_save_path)
    except AttributeError:
        try:
            model.save(model_save_path)
        except Exception:
            # 若上述方式失败，复制训练过程中保存的最优模型
            best_model_path = os.path.join(base_dir, 'runs', run_name, 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, model_save_path)
    
    return model
```

### 4. 模型验证

```python
  def validate_model(model):
    # 模型验证（基于验证集）
    metrics = model.val(
        data=data_yaml_path,    # 数据集配置文件路径
        split='val',            # 验证集标识
        project=os.path.join(base_dir, 'runs'),  # 验证结果保存根目录
        name='val'              # 验证结果文件夹名称
    )
    
    # 计算 F1 分数（综合精确率与召回率）
    f1_score = 2 * metrics.box.precision * metrics.box.recall / (metrics.box.precision + metrics.box.recall + 1e-6)
    
    return metrics
```

### 5. 图像测试

```python
  def test_on_images(model, conf_threshold=0.25):
    # 生成时间戳，用于测试结果文件夹唯一命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, 'runs', f'detect_{timestamp}')
    # 创建测试结果保存文件夹（若不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试集图像目录
    test_images_dir = os.path.join(test_path, "images")
    # 获取所有测试图像文件
    image_files = [f for f in os.listdir(test_images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # 从 data.yaml 中读取类别名称
    with open(data_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    class_names = yaml_data.get('names', ['Unknown'])
    
    # 选取前 10 张图像进行可视化（若不足 10 张则取全部）
    viz_images = image_files[:min(10, len(image_files))]
    
    # 逐张图像进行测试与结果可视化
    for img_file in viz_images:
        img_path = os.path.join(test_images_dir, img_file)
        
        # 模型推理（置信度阈值设为 0.25）
        results = model(img_path, conf=conf_threshold)
        
        # 读取原始图像并转换颜色空间（OpenCV 默认 BGR，转为 RGB 用于可视化）
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 在图像上绘制检测框与类别标签
        for box in results[0].boxes:
            # 获取检测框坐标（左上角 x1,y1，右下角 x2,y2）
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # 检测置信度
            conf = box.conf[0].item()
            # 类别索引
            cls = int(box.cls[0].item())
            
            # 获取类别名称（若索引超出范围则标记为 Unknown）
            class_name = class_names[cls] if cls < len(class_names) else f"Unknown-{cls}"
            
            # 为每个类别生成唯一颜色（基于类别索引计算）
            color = ((cls * 70) % 256, (cls * 50) % 256, (cls * 30) % 256)
            # 绘制检测框（线宽为 2）
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制类别标签（包含置信度，保留 2 位小数）
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存带检测结果的图像
        output_path = os.path.join(output_dir, img_file)
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        plt.axis('off')  # 隐藏坐标轴
        plt.tight_layout()  # 调整布局
        plt.savefig(output_path)
        plt.close()
```

### 6. 主函数

```python
  def main():
    # 验证数据集结构是否合法
    dataset_valid = verify_dataset_structure()
    if not dataset_valid:
        return  # 数据集不合法则终止程序
    
    # 生成本次训练的时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 训练模型（默认 50 轮、批次大小 16、初始学习率 0.01）
    model = train_yolo_model(epochs=50, batch_size=16, lr0=0.01)
    
    # 若模型训练成功，执行验证与测试
    if model is not None:
        validate_model(model)
        test_on_images(model)

if __name__ == "__main__":
    main()
```

## 五、关键超参数解析

### 1. 训练轮次（Epochs）

```python
  epochs=50  # 训练轮次（完整遍历数据集的次数）
```

- • **选择原因**：对于大多数数据集，50 轮训练能在训练时间与模型收敛效果间取得平衡。

- • **其他选择**：

- - • 较少轮次（20-30）：适用于大型数据集或使用强数据增强时；
    - • 较多轮次（100-300）：适用于小型数据集或复杂模型训练；

- • **早停机制**：`patience=10` 参数可实现早停——若 10 轮训练后模型性能无提升，将自动停止训练。

### 2. 批次大小（Batch Size）

```python
  batch_size=16  # 单次前向/反向传播处理的图像数量
```

- • **选择原因**：16 的批次大小能在内存占用与训练稳定性间达到平衡。

- • **其他选择**：

- - • 较小批次（4-8）：适用于 GPU 内存不足或处理超高分辨率图像时；
    - • 较大批次（32-64）：GPU 内存充足时使用，可能加速模型收敛；

- • **注意事项**：批次大小会影响梯度估计——较大批次的梯度更稳定，但可能降低模型探索性。

### 3. 图像尺寸（Image Size）

```python
  img_size=640  # 输入图像分辨率
```

- • **选择原因**：640×640 是 YOLO 模型的标准训练分辨率，兼顾细节保留与处理速度。

- • **其他选择**：

- - • 较小尺寸（416-512）：训练与推理速度更快，适用于检测大型目标；
    - • 较大尺寸（1024-1280）：适用于检测小型目标或需保留精细细节的场景；

- • **多尺度训练**：可添加 `--multi-scale` 参数，让模型在多种分辨率下训练，提升泛化能力。

### 4. 学习率（Learning Rate）

```python
  lr0=0.01    # 初始学习率
lrf=0.01    # 最终学习率（为初始学习率的比例）
```

- • **选择原因**：

- - • 0.01 的初始学习率足够“激进”，可加速模型收敛；
    - • 最终学习率为 0.0001（初始学习率的 1%），便于训练后期精细调优。

- • **其他选择**：

- - • 较低初始学习率（0.001）：适用于模型微调或训练不稳定时；
    - • 余弦退火策略：添加 `--cos-lr` 参数，实现学习率余弦式衰减；
    - • 学习率查找工具：使用 `lr_find()` 等工具快速测试最优学习率。

### 5. 进阶超参数

#### （1）两阶段训练

为提升迁移学习效果，可采用“先特征提取、后全量微调”的两阶段训练：

```python
  # 阶段 1：特征提取（冻结骨干网络层）
model = YOLO('yolov8n.pt')
model.train(data=data_yaml_path, epochs=10, lr0=0.001, freeze=[0, 1, 2, 3, 4, 5])
# 阶段 2：全量微调（解冻所有层）
model.train(data=data_yaml_path, epochs=40, lr0=0.01)
```

- • **设计思路**：冻结早期网络层可保留预训练模型学到的通用特征，仅训练检测头；后续解冻所有层，让整个网络适配自定义数据集。

#### （2）数据增强

针对小型数据集，可启用增强策略提升模型泛化能力：

```python
  # 小型数据集增强配置
results = model.train(
    # ... 其他已有参数
    augment=True,    # 启用增强
    fliplr=0.5,      # 水平翻转概率（50%）
    flipud=0.1,      # 垂直翻转概率（10%，若目标方向敏感可设为 0）
    mosaic=1.0,      # 马赛克增强（融合 4 张图像，提升目标多样性）
    mixup=0.1,       # 混合增强（融合 2 张图像，提升泛化能力）
    copy_paste=0.1   # 复制粘贴增强（提升目标实例学习效果）
)
```

- • **选择原因**：

- - • 马赛克增强可增加目标多样性；
    - • 混合增强帮助模型更好地泛化；
    - • 复制粘贴增强提升目标实例学习效果。

- • **其他选择**：

- - • HSV 增强：`hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`，提升模型对光照与颜色变化的鲁棒性；
    - • 平移/缩放增强：`translate=0.1, scale=0.5`，提升模型对目标位置变化的适应性；
    - • 关闭增强：若数据集本身多样性充足，可设 `augment=False`。

#### （3）类别权重（Class Weighting）

针对类别不平衡数据集，可通过权重调整缓解模型偏向多数类的问题：

```python
  # 为少数类设置更高权重（权重与类别频率成反比）
class_weights = [1.0, 2.0, 3.0]
results = model.train(
    # ... 其他已有参数
    class_weights=class_weights
)
```

- • **重要性**：类别不平衡会导致模型偏向多数类，通过权重设置，可让模型对少数类的预测错误“代价更高”，从而提升少数类检测效果。

## 六、YOLO 模型变体选择

不同 YOLOv8 变体在模型大小、速度与精度上存在差异，需根据需求选择：

| 模型    | 参数量  | 速度 | 精度 | 适用场景           |
| :------ | :------ | :--- | :--- | :----------------- |
| YOLOv8n | 320 万  | 最快 | 最低 | 边缘设备、实时应用 |
| YOLOv8s | 1120 万 | 较快 | 中等 | 平衡型应用         |
| YOLOv8m | 2590 万 | 中等 | 良好 | 标准检测任务       |
| YOLOv8l | 4370 万 | 较慢 | 优秀 | 高精度需求场景     |
| YOLOv8x | 6820 万 | 最慢 | 最佳 | 精度优先的关键场景 |

可根据优先级选择模型：

```python
  # 根据需求选择模型
if speed_critical:  # 速度优先
    model = YOLO('yolov8n.pt')  # 纳米级模型（最快）
elif balanced_needed:  # 平衡速度与精度
    model = YOLO('yolov8m.pt')  # 中型模型（平衡）
else:  # 精度优先
    model = YOLO('yolov8x.pt')  # 超大级模型（精度最高）
```

## 七、总结

微调 YOLO 模型需理解关键超参数及其对性能的影响，本文提供的代码与指南可作为自定义目标检测模型开发的基础框架。需注意，最优方案往往需要通过实验探索——建议从本文推荐的参数开始，结合验证集结果逐步调整，以适配你的具体应用场景。

 