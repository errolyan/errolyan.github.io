# 模型上下文协议（MCP）——使用Java构建SQL数据库代理（MCP代理教程）



在本实操教程中，我们将使用纯Java基于**模型上下文协议（Model Context Protocol，MCP）** 构建一个SQL数据库代理（SQL Database Agent）。该代理能让大型语言模型（LLMs）通过结构化操作（如创建表、插入数据、查询数据等）以编程方式与SQL数据库交互，全程无需手动编写SQL语句。

我此前已撰写过多篇关于MCP、智能体间通信（Agent-to-Agent，A2A）及其对比的深度文章。为聚焦实操，本文不再重复这些概念，而是直接使用a2ajava库进行实现。

✅ 本教程非常适合希望通过Java，借助智能体（Intelligent Agents）将数据库与LLM集成的开发者。


![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2025/10/20/1760939261500-b61c3eac-8de9-40b5-af07-0390c7f59d1c.png)


![](https://fastly.jsdelivr.net/gh/bucketio/img3@main/2025/10/20/1760939272822-4087141a-2235-45d3-b862-795d8e7d49f5.png)

## 🛠 编写Java类或Spring Bean——其余工作交给MCP即可
模型上下文协议（MCP）的一大核心优势是能与工具无缝集成，但本文要实现的功能是将Java方法转换为工具：你只需编写常规的Java类（更推荐编写Spring Boot服务），a2a库会自动将其转换为可用于MCP或A2A（智能体间）任务的工具。

以下是使用a2ajava和Spring的真实案例，该代理可处理内存中的Apache Derby数据库的SQL操作：

```java
package io.github.vishalmysore.service;

import com.t4a.annotations.Action;
import com.t4a.annotations.Agent;
import com.t4a.detect.ActionCallback;
import io.github.vishalmysore.a2a.domain.Task;
import io.github.vishalmysore.a2a.domain.TaskState;
import io.github.vishalmysore.data.*;

import lombok.extern.java.Log;
import org.springframework.stereotype.Service;

import java.sql.*;
import java.util.*;

@Log
@Service
@Agent(groupName = "Database related actions") // 标注为数据库相关操作的智能体
public class DerbyService {

    // Derby内存数据库的JDBC连接地址，若数据库不存在则自动创建
    private static final String JDBC_URL = "jdbc:derby:memory:myDB;create=true";

    private ActionCallback callback;

    // 启动数据库服务器的操作
    @Action(description = "start database server")
    public String startServer(String serverName) {
        log.info("Derby server started.");
        return "为" + serverName + "启动Derby服务器";
    }

    // 创建数据库的操作
    @Action(description = "Create database")
    public String createDatabase(String databaseName) {
        // 将任务状态设置为“已完成”，并添加数据库创建信息
        ((Task)callback.getContext()).setDetailedAndMessage(TaskState.COMPLETED, "Creating database: " + databaseName);
        try (Connection conn = DriverManager.getConnection(JDBC_URL)) {
            // 若连接成功则返回创建成功信息，否则返回失败信息
            return conn != null ? "数据库创建成功。" : "数据库创建失败。";
        } catch (SQLException e) {
            e.printStackTrace();
            return "数据库创建失败。";
        }
    }

    // 创建数据表的操作
    @Action(description = "Create tables")
    public String createTables(TableData tableData) {
        // 【仅作演示用：实际使用前请务必对输入进行安全校验！】
        StringBuilder createTableSQL = new StringBuilder("CREATE TABLE ");
        createTableSQL.append(tableData.getTableName()).append(" (");

        // 遍历表的列信息，拼接SQL语句
        for (ColumnData column : tableData.getHeaderList()) {
            createTableSQL.append(column.getColumnName())
                    .append(" ")
                    .append(column.getSqlColumnType())
                    .append(", ");
        }
        // 移除SQL语句末尾多余的逗号和空格
        createTableSQL.setLength(createTableSQL.length() - 2);
        createTableSQL.append(")");

        try (Connection conn = DriverManager.getConnection(JDBC_URL);
             Statement stmt = conn.createStatement()) {
            // 执行创建表的SQL语句
            stmt.execute(createTableSQL.toString());
            return tableData.getTableName() + "表创建成功。";
        } catch (SQLException e) {
            e.printStackTrace();
            return "创建表时出错：" + e.getMessage();
        }
    }

    // 向数据表插入新数据的操作
    @Action(description = "Insert new data in database table")
    public String insertDataInTable(TableData tableData) {
        StringBuilder insertSQL = new StringBuilder("INSERT INTO ");
        insertSQL.append(tableData.getTableName()).append(" (");

        // 获取表的列信息（从第一行数据中提取）
        List<ColumnData> columns = tableData.getRowDataList().get(0).getColumnDataList();
        for (ColumnData column : columns) {
            insertSQL.append(column.getColumnName()).append(", ");
        }
        // 移除末尾多余的逗号和空格，拼接VALUES子句
        insertSQL.setLength(insertSQL.length() - 2);
        insertSQL.append(") VALUES (");
        // 根据列数拼接占位符（?）
        insertSQL.append("?,".repeat(columns.size()));
        insertSQL.setLength(insertSQL.length() - 1);
        insertSQL.append(")");

        try (Connection conn = DriverManager.getConnection(JDBC_URL);
             PreparedStatement pstmt = conn.prepareStatement(insertSQL.toString())) {
            // 批量处理每一行数据
            for (RowData row : tableData.getRowDataList()) {
                int index = 1;
                for (ColumnData column : row.getColumnDataList()) {
                    pstmt.setObject(index++, column.getColumnValue());
                }
                pstmt.addBatch(); // 添加到批处理
            }
            pstmt.executeBatch(); // 执行批处理插入
            return "数据成功插入" + tableData.getTableName() + "表。";
        } catch (SQLException e) {
            e.printStackTrace();
            return "插入数据时出错：" + e.getMessage();
        }
    }

    // 从数据表查询数据的操作
    @Action(description = "Retrieve data from table")
    public List<Map<String, Object>> retrieveData(String sqlSelectQuery) {
        List<Map<String, Object>> result = new ArrayList<>();
        try (Connection conn = DriverManager.getConnection(JDBC_URL);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sqlSelectQuery)) {

            // 获取查询结果的元数据（列信息）
            ResultSetMetaData metaData = rs.getMetaData();
            int columnCount = metaData.getColumnCount();

            // 遍历查询结果，将每行数据转换为Map（键：列名，值：列值）
            while (rs.next()) {
                Map<String, Object> row = new HashMap<>();
                for (int i = 1; i <= columnCount; i++) {
                    row.put(metaData.getColumnName(i), rs.getObject(i));
                }
                result.add(row);
            }
            return result;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }
}
```


## ✅ 代码解析：核心逻辑是什么？
1. 用`@Agent`注解标记Java类，用`@Action`注解标记类中的方法；  
2. 这些被注解的类和方法会自动注册为可调用的**MCP工具**或**A2A任务**；  
3. LLM可通过MCP运行时调用这些工具，无需编写HTTP模板代码或手动配置JSON-RPC连接；  
4. ✨ 额外优势：可轻松将其接入Spring生态系统，非常适合由AI智能体驱动的生产级应用。  


## 控制器（Controller）
```java
package io.github.vishalmysore;

import io.github.vishalmysore.mcp.domain.*;
import io.github.vishalmysore.mcp.server.MCPToolsController;
import lombok.extern.java.Log;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@Log
@RestController
@RequestMapping("/mcp") // 基础请求路径：/mcp
public class MCPController extends MCPToolsController {

    // 列出所有可用工具的接口（GET请求）
    @GetMapping("/list-tools")
    public ResponseEntity<Map<String, List<Tool>>> listTools() {
        Map<String, List<Tool>> response = new HashMap<>();
        // 从父类MCPToolsController中获取工具列表
        response.put("tools", super.getToolsResult().getTools());
        return ResponseEntity.ok(response);
    }

    // 调用工具的接口（POST请求）
    @PostMapping("/call-tool")
    public ResponseEntity<CallToolResult> callTool(@RequestBody ToolCallRequest request) {
        // 调用父类MCPToolsController的工具调用方法
        return super.callTool(request);
    }
}
```

只需用`@Action`注解标记方法、用`@Agent`注解标记类，即可轻松将Java类或Spring Bean暴露为MCP工具或A2A任务。这些工具会被自动发现，并通过上述轻量级控制器访问。通过继承`MCPToolsController`，自定义的`MCPController`可提供如下端点：
- `/mcp/list-tools`：列出所有可用工具；
- `/mcp/call-tool`：动态调用任意工具。

这一设计能让后端服务无缝集成到任何AI驱动的工作流或前端界面中。


## 工具配置示例（JSON）
```json
{
  "tools": [
    {
      "parameters": {
        "type": "object",
        "properties": {
          "serverName": {
            "type": "string",
            "description": "参数：serverName（服务器名称）"
          }
        },
        "required": [
          "serverName"
        ],
        "additionalProperties": false
      },
      "inputSchema": {
        "type": "object",
        "properties": {
          "serverName": {
            "type": "string",
            "description": "参数：serverName（服务器名称）",
            "additionalProperties": null,
            "items": null
          }
        },
        "required": [
          "serverName"
        ]
      },
      "annotations": null,
      "description": "start database server（启动数据库服务器）",
      "name": "startServer",
      "type": null
    },
    {
      "parameters": {
        "type": "object",
        "properties": {
          "tableData": {
            "type": "object",
            "description": "参数：tableData（表数据，含表名、列信息等）"
          }
        },
        "required": [
          "tableData"
        ],
        "additionalProperties": false
      },
      "inputSchema": {
        "type": "object",
        "properties": {
          "tableData": {
            "type": "object",
            "description": "参数：tableData（表数据，含表名、列信息等）",
            "additionalProperties": null,
            "items": null
          }
        },
        "required": [
          "tableData"
        ]
      },
      "annotations": null,
      "description": "Create tables（创建数据表）",
      "name": "createTables",
      "type": null
    }
  ]
}
```

![](https://fastly.jsdelivr.net/gh/bucketio/img1@main/2025/10/20/1760939308400-643b8580-5928-4fba-9c5d-65c617ec88b7.png)


## NodeJS代理服务（对接Spring Boot）
```javascript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { 
  CallToolRequestSchema, 
  ListResourcesRequestSchema, 
  ListToolsRequestSchema, 
  ReadResourceRequestSchema 
} from "@modelcontextprotocol/sdk/types.js";
import puppeteer from "puppeteer-core";
import { Browserbase } from "@browserbasehq/sdk";

// 创建MCP服务器
const server = new Server(
  {
    name: "springboot-proxy", // 服务名称：Spring Boot代理
    version: "1.0.0" // 服务版本
  },
  {
    capabilities: {
      tools: {} // 工具将从Spring Boot动态加载
    }
  }
);

// 处理器：从Spring Boot获取工具列表
server.setRequestHandler(ListToolsRequestSchema, async () => {
  const response = await fetch("http://localhost:7860/mcp/list-tools", {
    method: "GET",
    headers: { "Content-Type": "application/json" }
  });

  if (!response.ok) {
    throw new Error(`获取工具失败：${response.statusText}`);
  }

  const data = await response.json();
  // 控制台打印Spring Boot提供的可用工具（可选）
  // console.log("Available tools from Spring Boot:", JSON.stringify(data, null, 2));
  return {
    tools: data.tools // 返回工具列表
  };
});

// 处理器：通过代理调用Spring Boot中的工具
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const response = await fetch("http://localhost:7860/mcp/call-tool", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: request.params.name, // 工具名称
      arguments: request.params.arguments ?? {} // 工具参数（若无则为空对象）
    })
  });

  if (!response.ok) {
    throw new Error(`工具调用失败：${response.statusText}`);
  }

  const data = await response.json();
  return data; // 返回工具调用结果（需符合CallToolResponseSchema格式）
});

// 通过标准输入输出（stdio）启动服务器
async function runServer() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // 控制台打印服务器启动信息（可选）
  // console.log("Proxy server is running on stdio...");
}

// 启动服务器并捕获错误
runServer().catch(console.error);
```


## 额外操作指南
1. 将上述NodeJS服务配置到Claude桌面客户端中；  
2. 或通过以下命令启动调试器：  
   ```bash
   npx @modelcontextprotocol/inspector node dist\testserver.js
   ```

完整代码：https://github.com/vishalmysore/SqlAIAgent