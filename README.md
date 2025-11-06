# Errol's Blog

个人技术博客网站，专注于分享数据结构、算法、AI技术和智能体(Agent)开发教程，支持Markdown文章渲染。

## 功能特性

- 响应式设计，适配各种设备
- 支持Markdown文章渲染
- 文章列表展示
- 文章详情页面
- 简洁美观的UI设计
- 搜索功能
- 文章分类和标签系统
- 多教程系列支持
- GitHub Pages自动部署

## 项目结构

```
├── articles/                     # Markdown文章文件目录
├── DataStructure_Tutorial/       # 数据结构与算法教程
├── LLM_Agent_mcp_skills_Tutorial/ # LLM智能体与MCP技能教程
├── AI_Programming_Tutorial/      # AI编程教程
├── AboutMe/                      # 关于博主信息
├── data/                         # 数据文件目录
├── books/                        # 电子书目录
├── src/
│   ├── scripts/                  # JavaScript文件
│   │   ├── utils.js              # 工具函数模块
│   │   ├── search.js             # 搜索功能模块
│   │   ├── markdown-parser.js    # Markdown解析器模块
│   │   ├── main.js               # 主入口文件
│   │   └── main-new.js           # 新版主入口文件
│   └── styles/                   # CSS样式文件
│       ├── fixed-styles.css      # 固定样式文件
│       ├── main.css              # 主样式文件
│       ├── markdown.css          # Markdown渲染样式
│       └── optimized-styles.css  # 优化后的样式文件
├── index.html                    # 主页
├── articles.html                 # 文章列表页
├── article.html                  # 文章详情页
├── ai-programming.html           # AI编程专栏
├── data-structure.html           # 数据结构专栏
├── about.html                    # 关于页面
├── package.json                  # 项目配置文件
├── OPTIMIZATION_REPORT.md        # 优化报告
├── robots.txt                    # 搜索引擎爬虫规则
├── sitemap.xml                   # 网站地图
└── README.md                     # 项目说明文件
```

## 本地开发

1. 克隆仓库

```bash
git clone https://github.com/errol/errol.github.io.git
cd errol.github.io
```

2. 安装依赖（可选，如需扩展功能）

```bash
npm install
```

3. 启动本地服务器

使用任何静态文件服务器，例如：

```bash
# 使用Python内置服务器
python -m http.server

# 或使用Node.js http-server
npx http-server
```

然后在浏览器中访问 `http://localhost:8000`

## 部署到GitHub Pages

1. 将项目推送到GitHub仓库（仓库名必须为 `yourusername.github.io`）

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/errol/errol.github.io.git
git push -u origin main
```

2. 在GitHub仓库设置中启用GitHub Pages
   - 进入仓库 → Settings → Pages
   - Source 选择 `Deploy from a branch`
   - Branch 选择 `main` 或 `master`，然后点击 Save

3. 等待几分钟，GitHub Pages会自动部署您的网站

## 添加新文章

1. 在 `articles/` 目录下创建新的Markdown文件
2. 在 `src/scripts/markdown-parser.js` 中的 `loadArticlesFromFiles()` 方法中添加新文章信息
3. 推送更改到GitHub仓库

## 技术栈

- HTML5
- CSS3
- JavaScript
- Markdown

## 优化内容

本项目经过全面优化，包括：

1. **性能优化**
   - CSS文件合并和压缩
   - 减少HTTP请求次数
   - 优化资源加载

2. **用户体验改进**
   - 响应式设计增强
   - 交互效果优化
   - 移动端适配

3. **SEO优化**
   - Meta标签完善
   - 语义化结构改进
   - 内容优化

4. **功能增强**
   - 搜索功能实现
   - 文章分类和标签系统
   - 分页功能

5. **代码结构优化**
   - 模块化重构
   - 功能分离
   - 可维护性提升

详细优化内容请参见 [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) 文件。

## 教程系列

### 1. 数据结构与算法教程
包含从入门到进阶的完整数据结构与算法讲解，涵盖数组、链表、树、图、排序、搜索等经典算法。

### 2. LLM智能体与MCP技能教程
专注于大语言模型(LLM)智能体开发、MCP协议应用、提示工程、RAG技术和模型优化等AI前沿技术。

### 3. AI编程教程
提供AI应用开发的实用指南和最佳实践。

## 部署状态

✅ 已部署到GitHub Pages
✅ 自动化构建流程已配置
✅ 响应式设计已优化
✅ 搜索功能已实现

## 贡献指南

欢迎提交Issue或Pull Request来改进这个博客网站。如果您有任何建议或问题，请随时联系作者。