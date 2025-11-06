# 第184期 Code execution with MCP：打造更高效的智能体（节省tokens）


直接调用工具时，每个工具的定义和执行结果都会占用上下文空间。而智能体通过编写代码来调用工具，能实现更好的扩展。本文将介绍如何结合MCP实现这一方案。

模型上下文协议（Model Context Protocol，简称MCP）是一种用于将AI智能体连接到外部系统的开放式标准。传统方式下，要将智能体与工具、数据相连，需为每一对组合开发定制化集成方案，这会导致系统碎片化且存在大量重复工作，难以构建真正可扩展的互联系统。而MCP提供了通用协议——开发者只需在智能体中实现一次MCP，就能解锁一整套集成生态。

自2024年11月MCP推出以来，其 adoption（采用率）增长迅速：社区已搭建数千台MCP服务器，适用于所有主流编程语言的SDK（软件开发工具包）均已上线，MCP也成为行业内智能体连接工具与数据的默认标准。

如今，开发者通常会构建可访问数十台MCP服务器上数百乃至数千种工具的智能体。但随着所连接工具数量的增加，预先加载所有工具定义以及通过上下文窗口传递中间结果的操作，会拖慢智能体运行速度并增加成本。

在本文中，我们将探讨代码执行如何帮助智能体更高效地与MCP服务器交互，在支持更多工具的同时减少token（令牌）消耗。

## 工具导致的token过度消耗降低智能体效率
随着MCP使用规模扩大，有两种常见情况会增加智能体的成本与延迟：
- 工具定义占用过多上下文窗口空间；
- 工具的中间结果消耗额外token。

### 1. 工具定义占用过多上下文窗口空间
大多数MCP客户端会将所有工具定义预先直接加载到上下文中，并通过直接工具调用语法向模型展示这些定义。工具定义示例如下：

**gdrive.getDocument（谷歌云端硬盘获取文档工具）**
- 描述：从谷歌云端硬盘中获取文档
- 参数：
  - documentId（必填，字符串类型）：待获取文档的ID
  - fields（可选，字符串类型）：需返回的特定字段
- 返回值：包含标题、正文内容、元数据、权限等信息的文档对象

**salesforce.updateRecord（ Salesforce更新记录工具）**
- 描述：在Salesforce（客户关系管理平台）中更新记录
- 参数：
  - objectType（必填，字符串类型）：Salesforce对象类型（如潜在客户、联系人、客户等）
  - recordId（必填，字符串类型）：待更新记录的ID
  - data（必填，对象类型）：需更新的字段及其新值
- 返回值：包含确认信息的更新后记录对象

工具描述会占用大量上下文窗口空间，导致响应时间延长、成本增加。若智能体连接了数千种工具，在处理请求前，需先处理数十万token。

![9Y5QZv](https://raw.githubusercontent.com/errolyan/tuchuang/master/uPic/9Y5QZv.png)

### 2. 工具的中间结果消耗额外token
大多数MCP客户端允许模型直接调用MCP工具。例如，你可能会向智能体下达指令：“从谷歌云端硬盘下载我的会议记录，并将其附加到Salesforce的潜在客户信息中”。

此时模型会执行如下调用：
```
工具调用：gdrive.getDocument(documentId: "abc123")
        → 返回结果：“讨论了第四季度目标...\n[完整会议记录文本]”
           （该结果会加载到模型上下文中）

工具调用：salesforce.updateRecord(
			objectType: "SalesMeeting"（销售会议）,
			recordId: "00Q5f000001abcXYZ",
  			data: { "Notes"（备注）: "讨论了第四季度目标...\n[完整会议记录文本]" }
		)
		（模型需再次将完整会议记录写入上下文）
```


每一个中间结果都必须经过模型处理。在上述示例中，完整的会议记录会两次流经模型。对于一场2小时的销售会议，这意味着需额外处理5万个token。若文档规模更大，甚至可能超出上下文窗口的容量限制，导致工作流程中断。

此外，当处理大型文档或复杂数据结构时，模型在工具调用之间复制数据时出错的概率也会增加。

MCP客户端会将工具定义加载到模型的上下文窗口中，并编排一个消息循环——在不同操作之间，每一次工具调用及其结果都需经过模型处理。

## 结合MCP的代码执行提升上下文效率
随着代码执行环境在智能体中的应用愈发普遍，一种解决方案应运而生：将MCP服务器作为代码API（应用程序编程接口）而非直接工具调用对象。这样一来，智能体就能通过编写代码与MCP服务器交互。该方案可同时解决上述两个问题：智能体只需加载当前所需的工具，并能在将结果返回给模型前，在执行环境中对数据进行处理。

实现该方案的方式有多种。其中一种是从已连接的MCP服务器中生成所有可用工具的文件树，以下是使用TypeScript（编程语言）实现的示例：
```
servers（服务器文件夹）
├── google-drive（谷歌云端硬盘文件夹）
│   ├── getDocument.ts（获取文档工具文件）
│   ├── ...（其他工具文件）
│   └── index.ts（索引文件）
├── salesforce（Salesforce文件夹）
│   ├── updateRecord.ts（更新记录工具文件）
│   ├── ...（其他工具文件）
│   └── index.ts（索引文件）
└── ...（其他服务器文件夹）
```


每个工具都对应一个文件，例如：
```typescript
// ./servers/google-drive/getDocument.ts（文件路径）
import { callMCPTool } from "../../../client.js";（从客户端文件中导入调用MCP工具的函数）

// 定义获取文档工具的输入参数类型
interface GetDocumentInput {
  documentId: string;（文档ID，字符串类型）
}

// 定义获取文档工具的返回结果类型
interface GetDocumentResponse {
  content: string;（文档内容，字符串类型）
}

/* 从谷歌云端硬盘读取文档的函数 */
export async function getDocument(input: GetDocumentInput): Promise<GetDocumentResponse> {
  return callMCPTool<GetDocumentResponse>('google_drive__get_document', input);（调用MCP工具获取文档）
}
```


前文提到的“从谷歌云端硬盘下载会议记录并附加到Salesforce潜在客户信息”示例，可通过以下代码实现：
```typescript
// 从谷歌文档读取会议记录，并添加到Salesforce潜在客户信息中
import * as gdrive from './servers/google-drive';（导入谷歌云端硬盘工具模块）
import * as salesforce from './servers/salesforce';（导入Salesforce工具模块）

// 从谷歌云端硬盘获取会议记录内容
const transcript = (await gdrive.getDocument({ documentId: 'abc123' })).content;
// 将会议记录更新到Salesforce的销售会议记录中
await salesforce.updateRecord({
  objectType: 'SalesMeeting',（对象类型：销售会议）
  recordId: '00Q5f000001abcXYZ',（记录ID）
  data: { Notes: transcript }（待更新数据：备注为会议记录）
});
```


智能体通过浏览文件系统发现工具：先列出“./servers/”目录下的可用服务器（如谷歌云端硬盘、Salesforce），再读取完成当前任务所需的特定工具文件（如getDocument.ts、updateRecord.ts），以了解每个工具的接口。这种方式让智能体仅加载当前任务所需的工具定义，将token用量从15万个减少到2000个，时间和成本均节省了98.7%。

Cloudflare（云计算服务公司）也发布了类似研究结果，并将结合MCP的代码执行称为“代码模式（Code Mode）”。核心思路一致：大语言模型（LLMs）擅长编写代码，开发者应利用这一优势，构建能更高效与MCP服务器交互的智能体。

## 结合MCP的代码执行优势
结合MCP的代码执行能让智能体更高效地利用上下文——按需加载工具、在数据到达模型前进行过滤、一步执行复杂逻辑。此外，该方案在安全性和状态管理方面也具备优势。

### 1. 渐进式信息披露
模型在浏览文件系统方面表现出色。将工具以代码形式存放在文件系统中，模型可按需读取工具定义，而非一次性加载所有定义。

此外，还可在服务器中添加“search_tools（工具搜索）”工具，用于查找相关定义。例如，在使用前文提到的Salesforce服务器时，智能体可搜索“salesforce”关键词，仅加载当前任务所需的工具。在“search_tools”工具中加入“细节级别”参数，允许智能体选择所需的信息详细程度（如仅工具名称、工具名称及描述、包含模式的完整定义等），也能帮助智能体节省上下文空间，更高效地查找工具。

### 2. 上下文高效的工具结果处理
处理大型数据集时，智能体可在代码中对结果进行过滤和转换后再返回给模型。以获取包含1万行数据的电子表格为例：

```
// 无代码执行情况——所有行数据均需流经上下文
工具调用：gdrive.getSheet(sheetId: 'abc123')
        → 返回1万行数据到上下文，需手动过滤

// 有代码执行情况——在执行环境中过滤数据
const allRows = await gdrive.getSheet({ sheetId: 'abc123' });（获取所有行数据）
const pendingOrders = allRows.filter(row => 
  row["Status"] === 'pending'（筛选状态为“待处理”的订单）
);
console.log(`Found ${pendingOrders.length} pending orders`);（打印待处理订单数量）
console.log(pendingOrders.slice(0, 5)); // 仅打印前5条数据供查看
```


此时智能体只需处理5行数据，而非1万行。类似思路也适用于数据聚合、多数据源关联、特定字段提取等场景——所有操作均不会占用过多上下文窗口空间。

### 3. 更强大且上下文高效的控制流
循环、条件判断和错误处理可通过熟悉的代码模式实现，无需串联多个单独的工具调用。例如，若需在Slack（即时通讯工具）中获取部署通知，智能体可编写如下代码：
```javascript
let found = false;（初始化“是否找到通知”标记为false）
while (!found) {（循环查找通知）
  const messages = await slack.getChannelHistory({ channel: 'C123456' });（获取Slack频道历史消息）
  found = messages.some(m => m.text.includes('deployment complete'));（判断消息中是否包含“部署完成”）
  if (!found) await new Promise(r => setTimeout(r, 5000));（未找到则等待5秒后重试）
}
console.log('Deployment notification received');（打印“已收到部署通知”）
```


这种方式比在智能体循环中交替执行MCP工具调用和休眠命令更高效。此外，通过编写可执行的条件判断树，还能减少“首token生成时间”延迟——无需等待模型解析if语句，智能体可直接让代码执行环境处理该逻辑。

### 4. 隐私保护操作
智能体结合MCP执行代码时，中间结果默认保留在执行环境中。这样一来，智能体仅能看到你明确记录或返回的数据，意味着无需与模型共享的数据可在工作流程中流转，且永远不会进入模型的上下文。

对于敏感度更高的任务，智能体工具可自动对敏感数据进行令牌化处理。例如，若需将电子表格中的客户联系方式导入Salesforce，智能体可编写如下代码：
```javascript
const sheet = await gdrive.getSheet({ sheetId: 'abc123' });（获取电子表格数据）
for (const row of sheet.rows) {（遍历表格每行数据）
  await salesforce.updateRecord({
    objectType: 'Lead',（对象类型：潜在客户）
    recordId: row.salesforceId,（记录ID）
    data: { 
      Email: row.email,（邮箱）
      Phone: row.phone,（电话）
      Name: row.name（姓名）
    }
  });
}
console.log(`Updated ${sheet.rows.length} leads`);（打印“已更新的潜在客户数量”）
```


MCP客户端会在数据到达模型前对其进行拦截，并将个人身份信息（PII）令牌化。例如，若智能体记录表格行数据，其看到的内容如下：
```
[
  { salesforceId: '00Q...', email: '[EMAIL_1]', phone: '[PHONE_1]', name: '[NAME_1]' },
  { salesforceId: '00Q...', email: '[EMAIL_2]', phone: '[PHONE_2]', name: '[NAME_2]' },
  ...
]
```


当数据在另一次MCP工具调用中共享时，MCP客户端会通过查找令牌，将其还原为原始数据。真实的邮箱地址、电话号码和姓名会从谷歌表格流转到Salesforce，但全程不会经过模型。这能防止智能体意外记录或处理敏感数据，同时也可通过该功能定义确定性安全规则，指定数据的流转范围。

### 5. 状态持久化与技能沉淀
具备文件系统访问权限的代码执行功能，可让智能体在不同操作间保持状态。智能体能将中间结果写入文件，从而实现任务续接和进度跟踪，示例代码如下：
```javascript
// 获取Salesforce中的潜在客户数据
const leads = await salesforce.query({ 
  query: 'SELECT Id, Email FROM Lead LIMIT 1000'（查询前1000条潜在客户的ID和邮箱）
});
// 将数据转换为CSV格式
const csvData = leads.map(l => `${l.Id},${l.Email}`).join('\n');
// 将CSV数据写入文件
await fs.writeFile('./workspace/leads.csv', csvData);

// 后续执行时可从文件中读取数据，续接任务
const saved = await fs.readFile('./workspace/leads.csv', 'utf-8');
```


智能体还能将自身代码保存为可复用函数。一旦智能体开发出可完成某任务的代码，就能将该实现保存下来，供后续使用。例如：
```typescript
// ./skills/save-sheet-as-csv.ts（技能文件：将表格保存为CSV格式）
import * as gdrive from './servers/google-drive';（导入谷歌云端硬盘工具）
export async function saveSheetAsCsv(sheetId: string) {
  const data = await gdrive.getSheet({ sheetId });（获取表格数据）
  const csv = data.map(row => row.join(',')).join('\n');（转换为CSV格式）
  await fs.writeFile(`./workspace/sheet-${sheetId}.csv`, csv);（写入文件）
  return `./workspace/sheet-${sheetId}.csv`;（返回文件路径）
}

// 后续任意智能体执行过程中，均可调用该函数
import { saveSheetAsCsv } from './skills/save-sheet-as-csv';
const csvPath = await saveSheetAsCsv('abc123');（调用函数，将指定表格保存为CSV）
```


这与“技能（Skills）”概念高度契合——“技能”是包含可复用指令、脚本和资源的文件夹，能帮助模型提升处理特定任务的性能。在这些保存的函数中添加SKILL.md文件，可构建结构化技能库，供模型参考和使用。长期来看，智能体可通过这种方式积累一套高阶能力工具箱，逐步完善高效工作所需的架构支持。

需注意的是，代码执行也会带来新的复杂性。运行智能体生成的代码，需要具备适当沙箱隔离、资源限制和监控功能的安全执行环境。这些基础设施需求会增加运营成本，同时带来直接工具调用所没有的安全考量。因此，在采用代码执行方案时，需权衡其优势（降低token成本、减少延迟、提升工具组合能力）与实现成本。

## 总结
MCP为智能体连接各类工具和系统提供了基础协议。但当连接的服务器数量过多时，工具定义和执行结果会消耗大量token，导致智能体效率下降。

尽管文中讨论的问题（上下文管理、工具组合、状态持久化）看似新颖，但在软件工程领域已有成熟解决方案。代码执行将这些既定模式应用于智能体，让智能体能通过熟悉的编程结构，更高效地与MCP服务器交互。若你采用了该方案，我们鼓励你向MCP社区分享你的实践经验。