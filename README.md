# Errol's Blog

个人博客网站，支持Markdown文章渲染。

## 功能特性

- 响应式设计，适配各种设备
- 支持Markdown文章渲染
- 文章列表展示
- 文章详情页面
- 简洁美观的UI设计

## 项目结构

```
├── articles/         # Markdown文章文件目录
├── src/
│   ├── scripts/      # JavaScript文件
│   └── styles/       # CSS样式文件
├── index.html        # 主页
├── articles.html     # 文章列表页
├── article.html      # 文章详情页
├── about.html        # 关于页面
├── package.json      # 项目配置文件
└── README.md         # 项目说明文件
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
2. 在 `src/scripts/markdown-parser.js` 中的 `loadSampleArticles()` 方法中添加新文章信息
3. 推送更改到GitHub仓库

## 技术栈

- HTML5
- CSS3
- JavaScript
- Markdown

## 许可证

MIT