# 第169期 如何从零开始学习大语言模型（LLMs）
## 五步学习路线图
![JouwiR](https://raw.githubusercontent.com/errolyan/tuchuang/master/uPic/JouwiR.png)

今天这篇文章旨在探讨如何从零开始学习大语言模型（LLMs）。这里所说的“学习”，并非指如何使用API接口，而是要真正理解其底层运作机制——从核心数学原理到架构细节，再到模型训练后的优化与对齐全过程。  

我们的学习计划分为五个阶段。  

完成所有阶段的学习后，你将清晰掌握从零构建小型GPT风格模型原型的方法，并且了解模型规模化扩展所需的关键要素。  



### 第一阶段：基础数学
假设你已掌握非常基础的数学知识，并且会使用Python编程，具备这些基础就足够开启学习了。  

因此，我们首先要做的是学习足够的微积分、线性代数和概率论知识，以便能够理解后续涉及的各类计算过程。  

如果你是完全的新手，建议先从建立直观认知入手。可以观看3Blue1Brown频道的《线性代数的本质》（Essence of Linear Algebra）和《微积分的本质》（Essence of Calculus）系列视频。这些视频以可视化、概念化的方式呈现内容，能帮助你真正理解数学知识在底层的实际作用。  

建立起一定的直观认知后，就可以进入更系统的学习阶段：  

- Coursera平台上由DeepLearning.AI开设的“机器学习数学基础”专项课程（Math for Machine Learning specialization）对新手十分友好，它会以循序渐进的节奏涵盖所有核心主题——微积分、线性代数和概率论。  
- 若你希望进一步提升手工计算能力，可以在完成DeepLearning.AI的专项课程后，继续学习伦敦帝国理工学院（Imperial College London）“机器学习数学”专项课程中的前两门。这些课程会提供大量导数计算和矩阵运算的练习，确保你能完全跟上后续课程的学习进度。  

顺便提一句——Coursera目前正在推出一项重磅优惠活动：Coursera Plus年度订阅服务可享受6折优惠，订阅后你能免费学习平台上的所有课程并获取证书，无需额外付费。此外，订阅还包含来自谷歌（Google）、国际商业机器公司（IBM）、微软（Microsoft）等合作方的超过10,000门课程，所有资源都包含在同一订阅服务中。该优惠活动仅在未来几周内有效，强烈建议你去了解一下！

### 第二阶段：神经网络
掌握了必备的数学知识后，我们就可以进入深度学习基础的学习了。  

和往常一样，我建议先通过可视化内容建立直观认知。我强烈推荐3Blue1Brown频道“神经网络”播放列表中的前四个视频，首先从《什么是神经网络？| 深度学习第一章》（But what is a neural network? | Deep learning chapter 1）开始观看。  

如果你希望在学习更高级的课程前，先对内容有一个通俗易懂的概览，StatQuest频道的深度学习基础播放列表也是不错的选择。具体来说，你可以观看其“神经网络核心思想”（The Essential Main Ideas of Neural Networks）播放列表中的第1至18个视频。  

你可以选择一次性看完整个播放列表，也可以在观看我接下来推荐的更正式课程前，先查看StatQuest频道中对应主题的讲解视频。这种学习方式对我非常有效，因为在进入数学和代码密集型的课程前，我已经掌握了核心概念。  

谈到代码实现，此时不得不推荐安德烈·卡帕西（Andrej Karpathy）的经典视频。这些视频清晰、逐步地讲解了反向传播（backpropagation）和神经网络训练的原理，具体可参考《神经网络与反向传播详解：构建micrograd》（The spelled-out intro to neural networks and backpropagation: building micrograd）。  

到这里，你已经掌握了足够的深度学习基础知识，能够开始理解作为大语言模型（LLMs）核心的Transformer架构了！  

但如果有人希望先建立更扎实的基础，我推荐学习DeepLearning.AI的“深度学习”专项课程，以及伊恩·古德费洛（Ian Goodfellow）等人编写的《深度学习》（Deep Learning）一书。Coursera上的这门专项课程之所以成为经典，是有其原因的，而且它会不断更新以保持内容的时效性。

### 第三阶段：Transformer架构与预训练
完成以上学习后，你就具备了理解支撑现代大语言模型（LLMs）的核心架构——Transformer架构所需的基础。  

我相信你肯定不会感到意外，我还是会建议从高层次的可视化介绍开始学习。;-)  

以下是一些优质资源：  

- 3Blue1Brown“神经网络”播放列表中第5个及之后的视频，以及StatQuest“神经网络”播放列表中的第19至22个视频。  
- 如果你更喜欢阅读，也可以查看博客《图解Transformer》（Illustrated Transformer）。  

更深入一层的学习，可以参考安德烈·卡帕西（Andrej Karpathy）的经典教程。他详细讲解了如何从零构建GPT模型以及GPT分词器（Tokenizer），具体课程包括：  
- 《从零开始编写代码：详解GPT构建过程》（Let’s build GPT: from scratch, in code, spelled out）  
- 《构建GPT分词器》（Let’s build the GPT Tokenizer）  

此外，阅读Transformer的原始论文《Attention Is All You Need》也非常有价值。  

如果你在理解论文时遇到困难，强烈推荐观看扬尼克·基尔彻（Yannic Kilcher）讲解该论文的视频。  

完成这些学习后，你就能理解“预训练”（pre-training）的概念了——预训练指的是在大规模语料库上训练模型，使其具备预测下一个token的能力。但现代大语言模型还需要经历更多步骤，接下来让我们继续学习。

### 第四阶段：微调（Fine-tuning）
经过预训练的模型虽然理解语言的运作方式，但尚未掌握在特定领域发挥实际作用的能力，而微调正是解决这一问题的关键环节。  

微调的过程是：以预训练的基础模型为起点，在特定领域的数据集上进一步训练模型，使其能够在法律、医疗、金融等特定领域，或其他任意专业领域中表现出色。  

要理解微调，我建议从DeepLearning.AI开设的短期课程《大语言模型微调》（Finetuning Large Language Models）开始学习。  

若想更深入地钻研，可以阅读《使用Transformer进行自然语言处理》（Natural Language Processing with Transformers）一书，以及2024年8月发布的技术综述《大语言模型微调完全指南：从基础到突破》（The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs）。  

与该领域的所有技术一样，微调技术也在不断发展。值得花时间了解一些更新的、参数效率更高的微调方法，例如LoRA（低秩适应，Low-Rank Adaptation）和QLoRA（量化低秩适应，Quantized Low-Rank Adaptation）。

### 第五阶段：对齐（Alignment）
至此，你已经拥有了一个经过微调、知道该“说什么”的模型。但如何让模型以“有用”（helpful）且“无害”（harmless）的方式输出内容呢？这就需要“基于人类反馈的强化学习”（Reinforcement Learning with Human Feedback，简称RLHF）技术——它能训练模型生成人类真正偏好的响应。  

以下是学习RLHF的推荐路径：  

1. 首先，若StatQuest频道有相关主题的视频，那它一定是我的首选推荐。所以，我们从这个概览视频开始：《基于人类反馈的强化学习（RLHF）：清晰解读！》（Reinforcement Learning with Human Feedback (RLHF), Clearly Explained!!!）。  
2. HuggingFace平台也有一个不错的入门视频：《从零基础到ChatGPT：基于人类反馈的强化学习》（Reinforcement Learning from Human Feedback: From Zero to chatGPT）。  
3. 你还应该阅读OpenAI发布的《深度强化学习入门》（Spinning Up in Deep RL）。这是一本非常适合新手的强化学习通用入门资料，虽然不专门针对RLHF，但它能为你理解对齐过程中使用的奖励建模（reward modeling）奠定扎实基础。  
4. 最后，如果你想进行全面深入的学习，可以阅读OpenAI的InstructGPT论文。该论文详细阐述了完整的RLHF流程，包括有监督微调（Supervised Fine-Tuning，简称SFT）、奖励建模和优化。建议搭配奥马尔·贾米尔（Umar Jamil）的视频讲解一起学习，该视频包含所有推导过程和代码，能帮助你建立扎实的知识体系。  

最近，“直接偏好优化”（Direct Preference Optimization，简称DPO）作为一种比RLHF更稳定的替代方法，正受到越来越多的关注。  