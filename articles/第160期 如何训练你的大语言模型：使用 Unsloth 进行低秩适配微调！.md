# 第160期 如何训练你的大语言模型：使用 Unsloth 进行低秩适配微调！  

![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2025/10/24/1761318125699-9f42c90c-e32f-4095-9b45-d234da72d898.png)


大语言模型（LLM）微调就像训练一条龙。图片由ChatGPT根据作者提示生成  
训练大语言模型的体验，很像在《驯龙高手》里教“无牙”（Toothless）飞行：“无牙”无疑是条强大的龙，但它少了一只尾翼；同理，基础大语言模型看似功能强大，但若未经训练、无法获取你的领域知识，就容易出现“幻觉”（生成虚假信息），给出不可靠的回复。  

我们的期望是什么？希望通过少量微调，让模型学习你的数据——最终在需要时能“精准咬合”（输出符合需求的结果）。  

在本文中，我将一步步拆解微调流程：从数据集准备到训练执行，并分享实际操作中的结果。剧透一下：第一次尝试并没有“喷火”般的惊艳效果，但这正是实验最有意思的地方。  

所有代码均可在我的GitHub仓库中获取。  


## 一、先聊聊大语言模型微调  
这部分内容至关重要，在写一行代码之前，我们先来深入探讨一下。  

大语言模型的微调，和其他AI模型的微调截然不同。因为大语言模型在规模和能力上都独树一帜：它们通过海量文本语料训练，能够形成对语言的理解能力。  

### 1. 大语言模型的“理解能力”  
随便问ChatGPT西兰花的健康益处，你都会为之惊叹——它的回答不仅快速、准确，还带有惊人的细节丰富度。  

按回车键或点击可查看完整尺寸图片  

大语言模型的能力甚至可以延伸到“推理”：尤其当它们在包含“思维链”（Chain-of-Thought, CoT）推理的数据集上训练时，推理能力会更突出。如今，大多数主流大语言模型都融入了这类复杂的训练机制。  

不过，大语言模型微调仍是一个存在争议的话题，学术论文中支持与反对的观点并存。一些研究警告，微调可能导致“灾难性遗忘”（模型忘记原有知识）、安全护栏弱化、幻觉风险增加以及隐私问题；更令人担忧的是，在教模型学习特定领域知识时，可能会无意间让它丧失推理能力。尽管有研究提出了缓解这些问题的技术，但持续的争议表明，解决方案并非那么简单。  

### 2. 微调的核心突破：参数高效微调（PEFT）  
大语言模型微调领域的一项重大突破，是“参数高效微调”（Parameter-Efficient Fine-Tuning, PEFT）。这是一种足以让人拍案叫绝的“跨时代思路”，如今已被快速纳入开源框架，用于生产环境。  

大语言模型的规模正在飞速增长。例如，Meta的Llama 4 Behemoth模型据称拥有2万亿个参数，极少有机构能拥有重训这类超大模型的计算资源。此外，为每个下游任务单独存储和部署微调模型的成本极高——因为每个微调模型的规模几乎与原始模型相当。  

PEFT的目标就是解决这两个核心问题：参数负担与部署成本，同时也能在一定程度上缓解前文提到的风险（尽管争议表明效果参差不齐）。PEFT技术仅对少量新增参数进行微调，而将预训练模型的大部分参数“冻结”（不更新），这大大降低了计算和存储需求。  

### 3. 低秩适配（LoRA）登场  
在各类PEFT方法中，“低秩适配”（Low-Rank Adaptation, LoRA）是目前应用最广泛的一种。它不更新模型的所有权重，而是冻结预训练模型的参数，通过注入可训练的低秩矩阵，实现任务特定的学习。  

LoRA的工作原理如下：  
- **确定目标层**：LoRA通常应用于线性层（例如Transformer注意力机制中的查询层、键层、值层和输出投影层）；  
- **添加低秩矩阵**：对于每个冻结的目标权重矩阵W₀，LoRA引入两个更小的矩阵A和B，使得它们的乘积BA具有较低的秩r（r远小于W₀的维度）；  
- **训练低秩矩阵**：微调过程中，仅更新矩阵A和B的参数，原始权重W₀保持不变，最终的有效权重矩阵为W = W₀ + BA；  
- **推理时可选合并矩阵**：训练完成后，可将低秩矩阵BA合并回原始权重矩阵W₀，形成新的权重矩阵W。这意味着，推理时的计算量与全量微调模型完全一致，无需额外开销。  

### 4. LoRA会影响大语言模型的激活值吗？  
答案是肯定的。低秩矩阵A和B用于计算原始权重矩阵的更新量（BA）；当输入经过LoRA适配层时，输出结果（以及后续的激活值）会与未适配的原始模型产生差异。  

本质上，LoRA通过注入可训练的低秩矩阵，修改了大语言模型的有效权重。这些修改后的权重会在推理时产生新的激活模式——且这种模式是针对特定任务定制的，同时还能保持较低的计算和内存成本。这是一种实用、精妙的优化手段，正在重塑大语言模型的实际应用方式。  

### 5. 要不要微调？关键看模型规模  
在我看来，微调的收益在“小型大语言模型”上体现得最明显。这类模型通常通过“知识蒸馏”从更大规模的“兄弟模型”中学习：尽管能继承基础的语言理解能力，但深度和广度往往不及大型模型。  

但这一点反而可能成为优势。实证研究表明，小型大语言模型的“灾难性遗忘”现象更轻微——微调时覆盖原有预训练知识的风险更低。作为回报，你只需付出远低于大型模型的训练、部署和推理成本，就能获得模型的“领域特定能力”。  

那么，“小型”具体指多大？在本文中，我们将微调Llama 3.2 10亿参数模型（Llama 3.2 1bn）——它是目前主流大语言模型中，规模较小但能力足够的代表之一。  


## 二、步骤1：构建训练集与测试集  
关于大语言模型微调，有一个常见误区：“训练数据永远不够”。但事实并非如此——因为你完全可以用另一个大语言模型来生成训练数据！  

首先，我们来搭建 notebook 环境：  

```python
# ./notebooks/training_dataset_gen.ipynb

import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.ollama import Ollama

# 创建数据目录并下载示例文档（关于“非传统资质”的论文）
!mkdir  -p ../data
!wget "https://arxiv.org/pdf/2405.00247.pdf" -O "../data/non_traditional_credentials.pdf"

# 加载文档
docs = SimpleDirectoryReader("../data/").load_data(show_progress=True)
```

我们将使用LlamaIndex的`RagDatasetGenerator`工具来构建数据集，具体代码如下：  

```python
# ./notebooks/training_dataset_gen.ipynb

data_gen = RagDatasetGenerator.from_documents(
    docs,
    llm=Ollama("qwen2.5"),  # 使用Qwen 2.5模型生成数据
    # 生成指令：模拟教师/教授，基于文档上下文生成1个问题及对应答案
    question_gen_query="You are a teacher/professor. Using the provided context, formulate a single question and its answer",
    num_questions_per_chunk=10  # 每个文档片段生成10个问题
)
# 从文档节点生成问答数据集
qa_dataset = data_gen.generate_dataset_from_nodes()
```

💡 **实用技巧**：生成问答对时，以及评估待微调的小型模型性能时，一定要用更强大（规模更大）的大语言模型。如果这个“强模型”对小型模型的输出满意，就说明你成功打造了一个“小而强”的紧凑模型。  

为了避免API调用次数限制和降低成本，我选择使用本地部署的Qwen 2.5 7B模型生成训练数据——它的能力足以支撑数据生成，且无需承担调用云端API的额外开销。  

接下来，我们将数据集划分为训练集和保留集（用于后续评估），并将保留集导出为CSV文件：  

```python
# ./notebooks/training_dataset_gen.ipynb

from sklearn.model_selection import train_test_split
from llama_index.core.llama_dataset import LabelledRagDataset
import json
import pandas as pd

# 提取所有示例
all_examples = qa_dataset.examples
# 按8:2比例划分训练集和测试集（随机种子确保可复现）
train_examples, test_examples = train_test_split(
    all_examples,
    test_size=0.2,          # 20%数据作为保留集
    random_state=42,        # 随机种子，保证结果可复现
    shuffle=True
)
print(f"训练集样本数：{len(train_examples)}，测试集样本数：{len(test_examples)}")

# 构建带标签的RAG数据集
training_dataset = LabelledRagDataset(examples=train_examples)
holdout_dataset = LabelledRagDataset(examples=test_examples)

# 将保留集转换为DataFrame并导出为CSV
records = []
for ex in holdout_dataset.examples:
    records.append({
        "query": ex.query,
        # 对参考上下文列表进行JSON编码
        "reference_contexts": json.dumps(ex.reference_contexts),
        "reference_answer": ex.reference_answer,
        # 对CreatedBy对象进行JSON编码
        "query_by": ex.query_by.model_dump_json(),
        "reference_answer_by": ex.reference_answer_by.model_dump_json(),
    })

df = pd.DataFrame.from_records(records)
df.to_csv("holdout_dataset.csv", index=False)

print(f"训练集样本数：{len(train_examples)}，测试集样本数：{len(test_examples)}")
```

按回车键或点击可查看完整尺寸图片  

最后，我们将训练集序列化为训练所需的JSONL格式：  

```python
# ./notebooks/training_dataset_gen.ipynb

def serialize_to_jsonl(examples, out_path="train.jsonl"):
    """
    将带标签的RAG数据示例序列化为JSONL格式
    参数：
        examples: LabelledRagDataExample列表，每个示例包含.query和.reference_answer字段
        out_path: JSONL文件的输出路径
    """
    def strip_prefix(text):
        # 移除文本开头的"**Question:**"或"**Answer:**"前缀（若存在）
        for p in ("**Question:**", "**Answer:**"):
            if text.strip().startswith(p):
                return text.strip()[len(p):].strip()
        return text

    with open(out_path, "w", encoding="utf8") as f:
        for ex in examples:
            q_raw = ex.query or ""
            a_raw = getattr(ex, "reference_answer", None)
            # 仅序列化包含"Question"前缀且有答案的示例
            if q_raw.lower().startswith("**question") and a_raw:
                q = strip_prefix(q_raw)
                a = a_raw.strip()
                # 构建符合格式的消息对象
                obj = {
                    "messages": [
                        {"role": "user",      "content": q},
                        {"role": "assistant", "content": a}
                    ]
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 序列化训练集
serialize_to_jsonl(train_examples)
```

按回车键或点击可查看完整尺寸图片  

大功告成！我们的训练集已经准备好了。  


## 三、步骤2：先评估“未微调”的基础模型性能  
在开始微调前，必须先评估基础模型的“原生性能”——这是重要的基准。我们将使用之前保存的保留集，进行一次简单的“检索增强生成”（RAG）评估。  

### 1. 加载保留集（从CSV文件）  
```python
# ./notebooks/training_dataset_gen.ipynb

from llama_index.core.llama_dataset import (
    LabelledRagDataset,
    LabelledRagDataExample,
    CreatedBy,
)

def get_rag_dataset_from_csv(csv_path: str):
    """从CSV文件加载带标签的RAG数据集"""
    # 定义CSV列的转换器（处理JSON格式数据）
    converters = {
        "reference_contexts":    lambda s: json.loads(s),
        "query_by":             lambda s: CreatedBy.model_validate_json(s),
        "reference_answer_by":  lambda s: CreatedBy.model_validate_json(s),
    }
    # 读取CSV文件
    df = pd.read_csv(csv_path, converters=converters)
    examples = []
    for _, row in df.iterrows():
        # 构建LabelledRagDataExample对象
        examples.append(
            LabelledRagDataExample(
                query=row["query"],
                query_by=row["query_by"],                      # 转换为CreatedBy对象
                reference_contexts=row["reference_contexts"],   # 转换为字符串列表
                reference_answer=row["reference_answer"],
                reference_answer_by=row["reference_answer_by"], # 转换为CreatedBy对象
            )
        )
    # 构建并返回带标签的RAG数据集
    dataset = LabelledRagDataset(examples=examples)
    return dataset

# 加载保留集
holdout_dataset = get_rag_dataset_from_csv("holdout_dataset.csv")
```

### 2. 构建RAG引擎  
```python
# ./notebooks/training_dataset_gen.ipynb

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex

# 初始化嵌入模型（用于文本向量转换）
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
# 从文档构建向量存储索引
index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
# 构建查询引擎（使用待微调的基础模型Llama 3.2 1B）
query_engine = index.as_query_engine(
      similarity_top_k=6,  # 检索Top 6个相似文档片段
      llm=Ollama("llama3.2:1b")  # 基础模型（后续将微调此模型）
)
```

### 3. 执行RAG评估  
```python
# ./notebooks/training_dataset_gen.ipynb

from llama_index.core.llama_pack import download_llama_pack

# 下载RAG评估工具包
RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./pack")
# 初始化评估器
rag_evaluator = RagEvaluatorPack(
    query_engine=query_engine, 
    rag_dataset=holdout_dataset,
    # 使用生成数据集时的同一模型（Qwen 2.5）作为评估裁判
    judge_llm=Ollama("qwen2.5", request_timeout=120.0),
    embed_model=OllamaEmbedding(model_name="nomic-embed-text")
)
# 运行评估并获取结果
benchmark_df = rag_evaluator.run()
```

评估指标说明：  
- “正确性”（Correctness）：评分范围为1-5分；  
- 其他指标（如相关性、忠实度、流畅度）：评分范围为0-1分。  

整体来看，基础模型的表现不算差，但“忠实度”（Faithfulness）得分明显偏低——这是一个关键短板。  

“忠实度”衡量的是生成回答与检索到的上下文之间的“事实一致性”：只有当回答中的每一个主张都能被检索上下文直接支持时，才算“忠实”。忠实度低，意味着模型存在“幻觉”（生成无依据内容）或“过度泛化”（超出上下文范围推断）的问题——这正是我们希望通过微调解决的核心痛点。  


## 四、步骤3：终于到微调环节！  
接下来就是最有趣的部分：微调！这里要介绍一个非常出色的框架——Unsloth.ai。  

Unsloth的核心优势在于，它通过手动推导反向传播步骤，绕开了PyTorch的标准自动求导（autograd）系统。这种优化大幅降低了计算需求，显著加快了训练速度，非常适合在本地或资源有限的硬件上运行。  

需要明确的是，Unsloth.ai的核心创新并非“替代训练框架”，而是“优化LLM模块”——专门针对PEFT微调场景进行改写。事实上，Unsloth的大多数示例代码（cookbook）都能在免费的Google Colab实例上运行，甚至能微调参数规模达140亿的模型！  

实际微调过程中，我们仍会使用Hugging Face的“Transformer强化学习”（TRL）框架——这是一个基于PyTorch的高层抽象框架，支持：  
- 监督式微调（包括推理密集型数据集）；  
- PEFT（参数高效微调）方法；  
- 基于强化学习的微调（如GRPO）。  

Unsloth与TRL的组合，为资源有限场景下的大语言模型训练提供了高效且灵活的解决方案。  

### 1. 加载数据集与模型  
```python
# ./notebooks/finetune_llama32_1bn.ipynb

import json
from datasets import Dataset
from unsloth import FastLanguageModel
import torch

# 加载数据集（添加系统提示词）
def load_messages_with_system(path, system_content="You are a helpful assistant."):
    examples = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            obj = json.loads(line)
            # 构建系统提示词消息
            sys_msg = {"role": "system", "content": system_content}
            # 获取用户-助手消息对
            ua_msgs = obj.get("messages", [])
            # 组合系统消息与用户-助手消息
            examples.append({"messages": [sys_msg] + ua_msgs})
    return examples

# 加载训练集（JSONL格式）
examples = load_messages_with_system("train.jsonl")
# 转换为Hugging Face Dataset格式
dataset = Dataset.from_list(examples)

# 加载模型与分词器（使用Unsloth的FastLanguageModel包装器）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",  # Unsloth预训练的Llama 3.2 1B模型
    max_seq_length=2048,  # 长上下文长度（可按需设置）
    load_in_4bit=True,   # 4位量化，减少内存占用
    load_in_8bit=False,  # 8位量化（精度略高，但内存占用翻倍）
    full_finetuning=False,  # 关闭全量微调（启用PEFT）
)
```

步骤说明：  
- 首先将JSONL格式的训练集加载为Hugging Face Dataset格式；  
- 然后通过Unsloth的`FastLanguageModel`包装器加载基础模型与分词器——`model_name`参数对应Unsloth在Hugging Face Hub上发布的模型，可根据需求浏览并选择兼容模型。  

### 2. 配置LoRA参数（启用PEFT）  
```python
# ./notebooks/finetune_llama32_1bn.ipynb

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 低秩矩阵的秩（常用值：8、16、32、64、128）
    # LoRA适配的目标层（不同模型可能不同，需参考官方文档）
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,  # LoRA缩放因子
    lora_dropout=0,  # LoRA dropout率（设为0以优化性能）
    bias="none",     # 偏置项处理（"none"最节省内存）
    use_gradient_checkpointing="unsloth",  # 使用Unsloth梯度检查点，减少30%显存占用
    random_state=3407,  # 随机种子，保证可复现
    use_rslora=False,   # 关闭秩稳定LoRA（按需启用）
    loftq_config=None,  # 关闭LoftQ量化（按需配置）
)
```

通过`FastLanguageModel.get_peft_model()`，我们将基础模型转换为支持PEFT的模型。其中，`target_modules`定义了LoRA将适配的层——不同模型的目标层可能不同，建议参考Unsloth官方示例代码，确认对应模型的目标层设置。  

### 3. 格式化训练数据（适配Llama 3.2指令格式）  
Llama 3.2的指令微调需要特定的对话格式，示例如下：  
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

知识截止日期：2023年12月
当前日期：2025年5月1日

你是一个乐于助人的助手。<|eot_id|><|start_header_id|>user<|end_header_id|>

1+1等于多少？<|eot_id|><|start_header_id|>assistant<|end_header_id|>

2<|eot_id|>
```

注意：该格式是“模型专属”的，但Unsloth提供了对话模板映射工具，可简化这一步骤：  

```python
# ./notebooks/finetune_llama32_1bn.ipynb

def formatting_prompts_func(batch):
    """将对话消息转换为Llama 3.2所需的格式"""
    convos = batch["messages"]  # 对话消息列表（每个元素是一条对话）
    # 应用分词器的对话模板
    texts = [
        tokenizer.apply_chat_template(convo,
                                      tokenize=False,
                                      add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

# 批量处理数据集（不删除原始"messages"字段）
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)
```

现在，数据集已完成格式转换，可用于训练！  

### 4. 配置并启动微调  
```python
# ./notebooks/finetune_llama32_1bn.ipynb
import os
from trl import SFTTrainer, SFTConfig

# 初始化监督式微调（SFT）训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,  # 可按需配置评估集
    args=SFTConfig(
        dataset_text_field="text",  # 数据集中文本字段的名称
        per_device_train_batch_size=2,  # 单设备训练批次大小
        gradient_accumulation_steps=4,  # 梯度累积步数（用于模拟更大批次）
        warmup_steps=5,  # 预热步数
        max_steps=60,  # 总训练步数（也可注释此参数，设置num_epochs=1）
        learning_rate=2e-4,  # 学习率
        logging_steps=1,  # 日志记录步数
        optim="adamw_8bit",  # 优化器（8位AdamW，节省内存）
        weight_decay=0.01,  # 权重衰减（防止过拟合）
        lr_scheduler_type="linear",  # 学习率调度器类型
        seed=2025,  # 随机种子
        report_to="none",  # 关闭日志上报（如需使用WandB等工具，可修改此参数）
    ),
)
# 启动训练
trainer_stats = trainer.train()

# 将微调后的模型推送到Hugging Face Hub
model.push_to_hub(
  "tituslhy/retrained_llama32-1bn-finetuned", 
  token=os.environ["HUGGINGFACE_ACCESS_TOKEN"]  # 从环境变量获取Hub令牌
) 
# 将分词器推送到Hugging Face Hub
tokenizer.push_to_hub(
  "tituslhy/retrained_llama32-1bn-finetuned", 
  token=os.environ["HUGGINGFACE_ACCESS_TOKEN"]
) 
```

训练配置说明：  
- 微调的核心是`SFTConfig`：它控制所有训练参数。本文中，我们选择按“固定步数”（60步）训练，而非按“完整轮次”（epoch）——按轮次训练耗时更长，按步数训练更适合快速原型验证；  
- Hugging Face的TRL框架封装了PyTorch的底层细节（如`zero_grad()`和自动求导调用），因此只需一行代码配置训练器、一行代码启动训练。  

训练启动后，你会看到Unsloth标志性的控制台输出：  
```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 16 | Num Epochs = 30 | Total steps = 60
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 4 x 1) = 8
 "-____-"     Trainable parameters = 11,272,192/1,000,000,000 (1.13% trained)
```

没错，输出里有一个树懒图案！🦥 在10亿个参数中，我们仅训练了1127万个参数——这就是LoRA低秩适配的魔力。  

训练完成后，可通过`model.push_to_hub()`将模型推送到Hugging Face Hub，方便后续调用。  

### 5. （可选）推送量化版本模型  
如果需要推送量化版本的模型，Unsloth底层使用llama.cpp实现该功能（注意：需确保llama.cpp安装正确，我曾在此步骤遇到过问题，重新安装几次后才解决）。代码如下：  

```python
# ./notebooks/finetune_llama32_1bn.ipynb
model.push_to_hub_gguf(
    "tituslhy/retrained_llama32-1bn-finetuned", 
    tokenizer,
    quantization_method=["q4_k_m", "q8_0", "q5_k_m"],  # 量化方式
    token=os.environ["HUGGINGFACE_ACCESS_TOKEN"], 
)
```

另外，你可能会注意到控制台输出显示“30轮训练”，但我们明明只设置了60步——这是因为训练数据集规模过小：少量样本需要循环多次，才能达到设定的步数。  

现在，关键问题来了：这种训练方式到底好不好？让我们通过评估来寻找答案。  


## 五、步骤4：使用Ollama本地部署模型  
首先，在Hugging Face Hub上找到你的模型，点击“Use this model”，再选择“Ollama”。  

按回车键或点击可查看完整尺寸图片  

按回车键或点击可查看完整尺寸图片  

你可以直接将页面提供的命令复制到终端运行，也可以稍作修改：将`run`改为`pull`，将模型下载到本地Ollama环境中。  


## 六、步骤5：评估微调后的模型  
### 1. 基于RAG的评估  
```python
# ./notebooks/finetune_llama32_1bn.ipynb

# 构建新的查询引擎（使用微调后的模型）
query_engine2 = index.as_query_engine(
    similarity_top_k=6, 
    llm=Ollama("hf.co/tituslhy/retrained_llama32-1bn-finetuned:Q4_K_M")
)
# 初始化新的评估器
rag_evaluator2 = RagEvaluatorPack(
    query_engine=query_engine2, 
    rag_dataset=holdout_dataset,
    # 仍使用Qwen 2.5作为评估裁判（与生成数据集时一致）
    judge_llm=Ollama("qwen2.5", request_timeout=120.0),
    embed_model=OllamaEmbedding(model_name="nomic-embed-text")
)
# 运行评估（异步方式）
benchmark_df = await rag_evaluator.arun()
```

结果令人意外：微调效果并不明显——尽管忠实度得分略有提升（达到0.31/1），但整体仍处于较低水平。  

原因可能在于：我们在“RAG模式”下评估微调模型——此时模型依赖外部检索到的文本片段生成答案，而非仅依靠自身内化的知识；但大语言模型微调的核心目标，是让模型“内化领域知识”，而非“提升使用外部上下文的能力”。  

### 2. 直接评估模型（无检索，仅依赖内化知识）  
因此，我们需要换一种正确的评估方式：不进行检索，让模型直接回答问题，仅依靠微调时学到的领域知识。  

```python
# ./notebooks/finetune_llama32_1bn.ipynb

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from tqdm import tqdm

# 初始化微调后的模型与嵌入模型
llm = Ollama("hf.co/tituslhy/retrained_llama32-1bn-finetuned:Q4_K_M")
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 读取保留集（处理JSON格式字段）
converters = {
        "reference_contexts":   lambda s: json.loads(s),
        "query_by":             lambda s: CreatedBy.model_validate_json(s),
        "reference_answer_by":  lambda s: CreatedBy.model_validate_json(s),
    }
df = pd.read_csv("holdout_dataset.csv", converters=converters)

# 初始化语义相似度评估器（相似度阈值设为0.5）
evaluator = SemanticSimilarityEvaluator(
    similarity_threshold=0.5, 
    embed_model=embed_model
)

# 筛选出包含"Question"的有效查询与参考答案
queries = [df.iloc[i]['query'] for i in range(len(df)) if "Question" in df.iloc[i]['query']]
references = [df.iloc[i]['reference_answer'] for i in range(len(df)) if "Question" in df.iloc[i]['query']]
# 构建DataFrame存储查询与参考答案
df_answers = pd.DataFrame({'queries': queries, 'reference_answers': references})

# 存储模型回答与相似度得分
answers, similarity_scores = [], []
# 遍历所有查询，让模型直接回答
for idx, (query, reference) in tqdm(df_answers.iterrows()):
    answer = llm.complete(query)
    answers.append(str(answer))
    
    # 计算模型回答与参考答案的语义相似度
    similarity_score = evaluator.evaluate(
        response=str(answer),
        reference=reference
    )
    similarity_scores.append(round(similarity_score.score, 2))

# 将结果添加到DataFrame
df_answers['answers'] = answers
df_answers['similarity_scores'] = similarity_scores
```

### 3. 评估结果分析  
我们来拆解这段代码的逻辑：  
1. 初始化微调后的模型与嵌入模型；  
2. 加载保留集，筛选出有效查询（包含“Question”的条目）；  
3. 让模型不依赖任何检索上下文，直接回答每个查询；  
4. 计算模型回答与参考答案之间的语义相似度（得分范围0-1）。  

按回车键或点击可查看完整尺寸图片  

从相似度得分来看，似乎有了提升！但别急着下结论——我们需要仔细阅读模型的回答内容，而非仅看分数。  

按回车键或点击可查看完整尺寸图片  

问题很快浮现：以第一个问题为例，模型的回答非常冗长，而参考答案只是一句话。这种“高分”其实具有误导性——更长的回答包含更多token和信息点，自然更容易与参考答案产生语义重叠。  

更深入分析“得分最高”的回答后，发现情况更不乐观。例如，有一个非常模糊的查询：“Which study are we referring to here?”（我们这里指的是哪项研究？）——这个问题本身表述不明确、信息不完整。  

按回车键或点击可查看完整尺寸图片  

我们来看看模型是如何处理的：  

按回车键或点击可查看完整尺寸图片  

从表面上看，模型似乎学到了文档中的一些信息——即使面对模糊的提示，也给出了连贯的回答。但问题在于：我们从未在训练数据中提及“该研究的名称”，因此这个回答很可能是“幻觉”。  

👉 我将这种情况称为“假阳性”：回答看起来正确，但实际上基于假设或虚构的上下文。  


## 七、实验反思  
总体而言，这次微调实验不能算“成功”，但却带来了宝贵的经验。以下是几点关键启示：  

1. **数据集质量至关重要**  
我们的训练集仅包含49个示例，且没有对数据质量进行校验（只是盲目相信大语言模型生成的问答对）。实际场景中，这类数据往往存在“What format is this document?”（这是什么格式的文档？）这类无意义问题，需要手动剔除。正因如此，生成训练数据集时，务必使用性能优异的大语言模型！  

2. **数据集规模同样关键**  
样本量过少时，微调很可能是“错误选择”——我们的实验就出现了性能未达预期的情况。更优的路径（我将在下一篇文章中探讨）是“微调整个RAG系统”，敬请关注！  

3. **训练配置决定结果成败**  
理想的训练配置应包含：  
- 单独的评估集（同样需要精心筛选或生成）；  
- 多组超参数实验（测试不同参数组合的效果）；  
- 借助Weights & Biases或MLflow等工具进行完整的训练跟踪。  

但在本次实验中，我们仅进行了一次“一次性训练”，且几乎没有跟踪记录——悄悄说一句，可别让我的教授知道这事。  

4. **微调比想象中简单，归功于开源社区**  
我们能轻松完成这次实验，离不开开源社区的贡献。Unsloth.AI、Hugging Face和Ollama等工具，将曾经复杂且高门槛的大语言模型微调，变得如今触手可及。没有这些出色的工具，本次实验的代码编写和运行会困难得多。  

虽然我们没能成功“教无牙学会咬合”（让模型精准输出），但这次实验仍是一次宝贵的“大语言模型微调入门”——我们不仅学到了大量知识，还得到了可复用的代码，而不仅仅是复制Unsloth的示例。  

目前，Hugging Face和Kaggle上有许多高质量的领域数据集可供使用。相信只要优化数据质量、设计更合理的训练方案，我们一定能构建出更精准、更可靠的微调模型。  

下一篇文章中，我将探索“微调整个RAG系统”——非常期待后续的结果，敬请关注！