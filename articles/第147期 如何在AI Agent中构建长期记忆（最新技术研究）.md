# 第147期 如何在AI Agent中构建长期记忆（最新技术研究）

基于智能代理或检索增强生成（RAG）的解决方案通常依赖于两层内存系统，该系统使智能代理或大语言模型（LLM）既能专注于当前上下文，又能保留过去的经验。

![](https://fastly.jsdelivr.net/gh/bucketio/img4@main/2025/10/18/1760758728205-c6627e4a-73ee-4756-9003-7311924b53d6.png)

短期记忆管理活跃会话或对话中的即时信息。长期记忆跨会话存储和检索知识，随着时间的推移实现连续性和学习。这些层次共同作用，使智能代理显得更加连贯、有上下文感知且智能。让我们来直观了解一下这些内存组件在现代人工智能架构中的位置……

## 智能代理架构中的内存系统
因此，有两种类型的内存层：
1. **线程级内存（短期）**： 这种内存作用于一个对话线程内，它跟踪在该会话期间已经发生的消息、上传的文件、检索的文档，以及智能代理与之交互的任何其他内容。你可以将其视为智能代理的 “工作内存”。它帮助智能代理理解上下文，并自然地继续讨论，而不会忘记前面的步骤。LangGraph会自动管理这种内存，通过检查点保存进度。一旦对话结束，这种短期内存就会被清除，下一个会话重新开始。
2. **跨线程内存（长期）**： 第二种类型的内存设计为不止在一次聊天中持续存在。这种长期内存存储智能代理可能需要跨多个会话记住的信息，如用户偏好、早期决策或在此过程中了解到的重要事实。LangGraph将这些数据作为JSON文档保存在内存存储中。信息使用命名空间（类似于文件夹）和键（类似于文件名）进行整齐地组织。由于这种内存在对话结束后不会消失，智能代理可以随着时间的推移积累知识，并提供更一致、更个性化的响应。

在这篇博客中，我们将探索一个生产级人工智能系统如何使用LangGraph（一个用于构建可扩展且有上下文感知的人工智能工作流程的流行框架）来管理长期内存流。

这篇博客是基于LangGraph智能代理指南创作的。所有代码都可以在这个GitHub代码库中找到：

[GitHub - FareedKhan-dev/langgraph-long-memory: A detail Implementation of handling long-term memory…](https://github.com/FareedKhan-dev/langgraph-long-memory)：在智能代理人工智能中处理长期记忆的详细实现 - FareedKhan-dev/langgraph-long-memory

### 目录
- LangGraph数据持久层
    - 1. 内存存储（用于笔记本和快速测试）
    - 2. 本地开发存储（使用langgraph dev命令）
    - 3. 生产存储（LangGraph平台或自托管）
- 使用内存特性
- 构建智能代理架构
    - 定义我们的模式
    - 创建代理提示
    - 定义工具和实用程序
- 内存函数和图节点
- 使用人类在环捕获反馈
- 组装图工作流程
- 测试带内存的智能代理
    - 测试用例1：基线 - 接受提案
    - 测试用例2：从直接编辑中学习
- 长期内存系统是如何工作的？

### LangGraph数据持久层
LangGraph是处理智能代理内存的最常用组件，LangGraph中最常见的特性之一是存储特性，它根据你运行项目的环境来管理内存的保存、检索和更新方式。

LangGraph提供了不同类型的存储实现，平衡了简单性、持久性和可扩展性。每个选项都适用于特定的开发或部署阶段。

![24lWzU](https://raw.githubusercontent.com/errolyan/tuchuang/master/uPic/24lWzU.png)

#### Langgraph数据持久层
让我们来了解如何以及何时相应地使用每种类型。
1. **内存存储（用于笔记本和快速测试）**：这是最简单的存储选项，非常适合短期实验或演示。按回车键或点击以查看完整尺寸的图片。它使用`from langgraph.store.memory import InMemoryStore`导入语句，创建一个完全在内存中运行的存储，使用标准的Python字典。由于它不会将数据写入磁盘，所以当进程结束时，所有信息都会丢失。然而，它速度非常快且易于使用，非常适合测试工作流程或尝试新的图配置。如果需要，也可以添加语义搜索功能，如语义搜索指南中所述。
2. **本地开发存储（使用langgraph dev命令）**：这个选项的行为与内存版本类似，但包括会话之间的基本持久性。按回车键或点击以查看完整尺寸的图片。当你使用`langgraph dev`命令运行应用程序时，LangGraph会自动使用Python的pickle格式将存储保存到本地文件系统。这意味着在重新启动开发环境后，你的数据会被恢复。它轻量级且方便，不需要外部数据库。如果需要，你仍然可以启用语义搜索功能，如语义搜索文档中所解释的那样。这种设置非常适合开发工作，但不适用于生产环境。
3. **生产存储（LangGraph平台或自托管）**：对于大规模或生产部署，LangGraph使用集成了pgvector的PostgreSQL数据库，以实现高效的向量存储和语义检索。按回车键或点击以查看完整尺寸的图片。这种设置提供了完整的数据持久性、内置的可靠性，以及处理更大工作量或多用户系统的能力。语义搜索是开箱即用的，默认的相似性度量是余弦相似度，不过你可以根据特定需求进行定制。这种配置确保你的内存数据被安全存储，并且即使在高流量或分布式工作负载下，也能在各个会话中保持可用。

现在我们已经了解了基础知识，就可以开始逐步编码整个工作架构了。

### 使用内存特性
我们将在这篇博客中实现的类别是内存特性，这是在基于人工智能的系统中管理内存的最常见方法。

它以顺序方式工作，在逐步构建或测试技术过程时很有用。

![gYSuIO](https://raw.githubusercontent.com/errolyan/tuchuang/master/uPic/gYSuIO.png)

它允许我们在运行代码时临时存储数据，并有助于理解LangGraph中内存处理的工作方式。

我们可以从从LangGraph导入`InMemoryStore`开始。这个类让我们可以直接在内存中存储记忆，而无需任何外部数据库或文件系统。
```python
# 导入InMemoryStore类，用于在内存中存储记忆（无持久性）
from langgraph.store.memory import InMemoryStore
# 初始化一个内存存储实例，用于本笔记本中的使用
in_memory_store = InMemoryStore()
```
在这里，我们基本上创建了`InMemoryStore`的一个实例。在我们进行示例操作时，它将保存我们的临时数据。由于这仅在内存中运行，所以一旦进程停止，所有存储的数据都将被清除。

LangGraph中的每个记忆都保存在一个称为命名空间的东西里面。

命名空间就像一个标签或文件夹，有助于组织记忆。它被定义为一个元组，可以有一个或多个部分。在这个例子中，我们使用一个包含用户ID和一个名为“memories”标签的元组。
```python
# 定义用于内存存储的用户ID
user_id = "1"
# 设置用于存储和检索记忆的命名空间
namespace_for_memory = (user_id, "memories")
```
命名空间可以代表任何内容，并不总是必须基于用户ID。你可以根据应用程序的结构，按照自己的意愿使用它来对记忆进行分组。

接下来，我们将一个记忆保存到存储中。为此，我们使用`put`方法。这个方法需要三样东西：命名空间、唯一键和实际的记忆值。

这里，键将是使用`uuid`库生成的唯一标识符，记忆值将是一个存储一些信息的字典——在这个例子中，是一个简单的偏好。
```python
import uuid
# 为记忆生成唯一ID
memory_id = str(uuid.uuid4())
# 创建一个记忆字典
memory = {"food_preference": "I like pizza"}
# 将记忆保存到定义的命名空间中
in_memory_store.put(namespace_for_memory, memory_id, memory)
```
这将我们的记忆条目添加到我们之前定义的命名空间下的内存存储中。

一旦我们存储了记忆，就可以使用`search`方法将其取回。这个方法会在命名空间内查找，并以列表形式返回属于该命名空间的所有记忆。
![DQllHC](https://raw.githubusercontent.com/errolyan/tuchuang/master/uPic/DQllHC.png)
每个记忆都是一个`Item`对象，包含其命名空间、键、值和时间戳等详细信息。我们可以将其转换为字典，以便更清楚地查看数据。
```python
# 检索给定命名空间中存储的所有记忆
memories = in_memory_store.search(namespace_for_memory)
# 查看最新的记忆
memories[-1].dict()
```
当我们在笔记本中运行这段代码时，得到了以下输出：
```json
{
 "namespace": ["1", "memories"],
 "key": "c8619cd4-3d3f-4108-857c-5c8c12f39e87",
 "value": {"food_preference": "I like pizza"},
 "created_at": "2025-10-08T15:46:16.531625+00:00",
 "updated_at": "2025-10-08T15:46:16.531625+00:00",
 "score": null
}
```
输出显示了存储的记忆详细信息。这里最重要的部分是`value`字段，它包含了我们保存的实际信息。其他字段有助于识别和管理记忆的创建时间和位置。

一旦存储准备就绪，我们可以将其连接到一个图，以便内存和检查点能够协同工作。我们在这里使用两个主要组件：
- `InMemorySaver`用于管理线程之间的检查点。
- `InMemoryStore`用于存储跨线程内存。
```python
# 启用线程（对话）
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
# 启用跨线程内存
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
# 使用检查点和存储编译图
# graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)
```
它使图能够使用相同的内存机制记住线程内的对话上下文（短期），并跨线程保留重要信息（长期）。

这是在转向生产级存储之前测试内存行为的一种简单有效的方法。

### 构建智能代理架构
在我们看到内存系统的工作流程之前，需要构建使用它的智能代理。由于本指南侧重于内存管理，我们将构建一个中等复杂度的电子邮件助手。这将使我们能够在现实场景中探索内存是如何工作的。



我们将从头开始构建这个系统，定义其数据结构、“大脑”（提示）和能力（工具）。最后，我们将拥有一个不仅能回复电子邮件，还能从我们的反馈中学习的智能代理。

#### 定义我们的模式
为了处理任何数据，我们需要定义其形状。模式是我们智能代理信息流的蓝图，它们确保一切都是结构化的、可预测的且类型安全的。

首先，我们将编写`RouterSchema`。我们需要它的原因是为了使我们的初始分类步骤可靠。当我们期望得到明确的决策时，不能冒险让大语言模型返回非结构化文本。

这个Pydantic模型将强制大语言模型给我们一个干净的JSON对象，其中包含其推理过程和一个严格为“ignore”（忽略）、“respond”（回复）或“notify”（通知）之一的分类。
```python
# 从Pydantic和Python的typing模块导入必要的库
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
# 定义我们路由器结构化输出的Pydantic模型
class RouterSchema(BaseModel):
    """分析未读电子邮件并根据其内容进行分类。"""
    # 添加一个字段，让大语言模型解释其逐步推理过程
    reasoning: str = Field(description="分类背后的逐步推理。")
    # 添加一个字段来保存最终分类
    # Literal类型将输出限制为这三个特定字符串之一
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="电子邮件的分类。"
    )
```
我们正在为我们的分类大语言模型创建一个契约。稍后当我们将其与LangChain的`.with_structured_output()`方法配对时，我们保证输出将是一个可预测的Python对象，我们可以使用它，使我们图中的逻辑更加健壮。

接下来，我们需要一个地方来存储我们智能代理单次运行的所有信息。这就是`State`的目的。它就像一个中央白板，我们图的每个部分都可以从中读取和写入信息。
```python
# 从LangGraph导入基础状态类
from langgraph.graph import MessagesState
# 定义我们图的中央状态对象
class State(MessagesState):
    # 这个字段将保存初始的原始电子邮件数据
    email_input: dict
    # 这个字段将存储我们分类路由器做出的决策
    classification_decision: Literal["ignore", "respond", "notify"]
```
我们继承自LangGraph的`MessagesState`，它会自动给我们一个消息列表来跟踪对话历史。然后我们添加自己的自定义字段。随着流程从一个节点移动到另一个节点，这个`State`对象将被传递，并积累信息。

最后，我们将定义一个小但重要的`StateInput`模式，来定义我们图的第一个输入应该是什么样子。
```python
# 定义整个工作流程初始输入的TypedDict
class StateInput(TypedDict):
    # 工作流程必须以包含'email_input'键的字典开始
    email_input: dict
```
这个简单的模式从我们应用程序的入口点就提供了清晰性和类型安全性，确保对我们图的任何调用都以正确的数据结构开始。

#### 创建代理提示
我们正在使用一种提示方法来指导大语言模型的行为。对于我们的智能代理，我们将定义几个提示，每个提示用于特定的任务。

在智能代理从我们这里学到任何东西之前，它需要一组基线指令。这些默认字符串将在第一次运行时加载到内存存储中，为智能代理的行为提供一个起点。

首先，让我们定义`default_background`，给我们的智能代理一个角色设定。
```python
# 为智能代理定义默认角色设定
default_background = """ 
我是兰斯，LangChain的一名软件工程师。
"""
```
接下来，是`default_triage_instructions`。这些是我们的分类路由器最初用于分类电子邮件的规则。
```python
# 为分类大语言模型定义初始规则
default_triage_instructions = """
不值得回复的电子邮件：
- 营销时事通讯和促销电子邮件
- 垃圾邮件或可疑电子邮件
- 在仅供参考的线程中被抄送且没有直接问题的邮件

需要通知但无需回复的电子邮件：
- 团队成员生病或休假
- 构建系统通知或部署

需要回复的电子邮件：
- 团队成员的直接问题
- 需要确认的会议请求
"""
```
现在，是`default_response_preferences`，它定义了智能代理最初的写作风格。
```python
# 定义智能代理撰写电子邮件的默认偏好
default_response_preferences = """
使用专业简洁的语言。
如果电子邮件提到截止日期，确保在回复中明确提及并引用该截止日期。
在回复会议安排请求时：
- 如果提出了时间，确认日历可用性并确定一个时间。
- 如果没有提出时间，查看你的日历并提出多个选项。
"""
```
最后，是`default_cal_preferences`，用于指导其日程安排行为。
```python
# 定义安排会议的默认偏好
default_cal_preferences = """
首选30分钟的会议，但15分钟的会议也可以接受。
"""
```
现在我们创建将使用这些默认值的提示。首先是`triage_system_prompt`。
```python
# 为初始分类步骤定义系统提示
triage_system_prompt = """
<角色>
你的角色是根据背景和指令对传入的电子邮件进行分类。
</角色>
<背景>
{background}
</背景>
<指令>
将每封电子邮件分类为“忽略”、“通知”或“回复”。
</指令>
<规则>
{triage_instructions}
</规则>
"""
```
这个提示模板赋予我们的分类路由器角色和指令。`{background}`和`{triage_instructions}`占位符将用我们刚刚定义的默认字符串填充。
![NdStvm](https://raw.githubusercontent.com/errolyan/tuchuang/master/uPic/NdStvm.png)

接下来是`triage_user_prompt`，它是一个简单的模板，用于将原始电子邮件内容结构化为大语言模型可以轻松解析的干净格式。
```python
# 为分类定义用户提示，它将格式化原始电子邮件
triage_user_prompt = """
请确定如何处理以下电子邮件：
发件人：{author}
收件人：{to}
主题：{subject}
{email_thread}
"""
```
现在对于主要组件，我们必须创建`agent_system_prompt_hitl_memory`，因为它将包含我们到目前为止编写的角色和其他类型的指令。
```python
# 导入datetime库，以便在提示中包含当前日期
from datetime import datetime
# 为响应代理定义主要的系统提示
agent_system_prompt_hitl_memory = """
<角色>
你是一位顶尖的执行助理。 
</角色>
<工具>
你可以使用以下工具：{tools_prompt}
</工具>
<指令>
1. 仔细分析电子邮件内容。
2. 每次只调用一个工具，直到任务完成。
3. 使用“Question”向用户寻求澄清。
4. 使用“write_email”起草电子邮件。
5. 对于会议，检查可用性并相应安排。
    - 今天的日期是 """ + datetime.now().strftime("%Y-%m-%d") + """
6. 发送电子邮件后，使用“Done”工具。