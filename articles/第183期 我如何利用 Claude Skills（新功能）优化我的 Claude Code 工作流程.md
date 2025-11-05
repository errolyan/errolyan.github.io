# 第183期 我如何利用 Claude Skills（新功能）优化我的 Claude Code 工作流程

![](https://fastly.jsdelivr.net/gh/bucketio/img17@main/2025/11/05/1762313916619-6e25becb-1951-48c6-bd69-49f3ee0bfb60.png)

Claude Code 因全新的 Claude Skills 功能变得更加强大，借助该功能，你只需向 Claude 传授一次工作流程，它就能每次都按此流程执行任务。

听起来似乎难以置信，但读完本文并观看我的演示后，你会明白 Claude Skills 能为 Claude Code 工作流程带来多大的助力。

如果你对 Claude Skills 一无所知，通过我对其解决的问题以及当前使用方法的深入解析，你将掌握所有必备知识，甚至能动手实践。

温馨提示：若你并非高级会员，可点击此处阅读本文。欢迎在 Medium 上关注我，并订阅我的 YouTube 频道，以了解更多关于 Claude Code 的内容，支持我的创作。

我使用 Claude Code 已有数月，也一直在 Medium 上分享如何充分利用其每一项新功能的技巧。但自动化需求的核心始终存在一个棘手问题——重复性操作。

每次开启新项目时，你总会发现自己在反复向 Claude 解释相同的内容：

“添加 TypeScript，并使用这些特定的 tsconfig 配置。”
“按照我的偏好设置 ESLint 配置。”
“遵循团队的命名规范创建 React 组件。”
这是我们所有人都面临的问题，我们也曾尝试利用之前讨论过的各种 Claude Code 功能来解决，比如我撰写的关于 Claude Code 斜杠命令、Claude Code 钩子的教程，以及其他相关教程。

大多数开发者在每个项目的前30分钟，都在重复向 Claude 传授自己的工作流程、偏好设置和标准规范。

如果你也有这样的困扰，Claude Skills 能帮你快速节省宝贵时间：它将知识传递过程自动化，把你重复性的解释转化为可复用、可执行的指令，Claude 会在需要时自动加载这些指令。

此时你可能会想：

“等等，这不就是模型上下文协议（MCP）能解决的问题吗？”

事实并非完全如此。

MCP 的优势在于让 Claude 能够访问外部数据源和工具，例如连接数据库、API 或文件系统——我已详细介绍过这一点，并在此处分享了最佳 MCP 服务器列表。

但 Skills 有所不同：它的核心是教 Claude“如何做”。

Skills 是可执行的知识包，它会告诉 Claude：“当用户需要完成 X 任务时，我们团队/组织是这样操作的。”

简单来说，MCP 帮 Claude 连接你的数据，而 Skills 教 Claude 掌握你的专业方法。

在测试并深入了解 Claude Skills 的功能后，我意识到自己之前使用 Claude 的方式太过繁琐。接下来，我将向你展示如何利用这一功能优化工作流程。

## 什么是 Claude Skills？
![Ec5gkO](https://raw.githubusercontent.com/errolyan/tuchuang/master/uPic/Ec5gkO.png)

Claude Skills 是可移植的指令包，能教会 Claude 按照你的方式执行特定任务。

我用通俗易懂的语言来解释一下：一个 Skill 本质上就是一个文件夹，其中包含以下内容：

- **SKILL.md**：Markdown 格式的文件，记录你的指令、模式和最佳实践
- **支持资源**：脚本、模板、配置文件或参考代码
- **执行逻辑**：可选代码，用于确保任务每次都以相同方式执行

我喜欢把 Skills 比作给 Claude 的“食谱卡”。就像你给别人一份食谱，让他们能烤出和你祖母做的一样完美的巧克力蛋糕，Skills 也会为 Claude 提供一步步的指令，确保它能完全按照你的要求完成特定任务。

以下是文件系统中一个典型 Skill 的结构：

```
~/.claude/skills/
└── darkmode-toggle/
    ├── SKILL.md
    ├── templates/
    │   ├── toggle-button.html
    │   └── theme-styles.css
    └── scripts/
        └── theme-handler.js
```

在这个示例中，`darkmode-toggle` Skill 包含了 Claude 为任何网站添加专业深色模式切换功能所需的全部内容——既有 SKILL.md 中的指令，也有可供参考或直接使用的实际模板文件。

Skills 的一大亮点是跨平台可用性：它能在 Claude.ai、Claude Code 和 API 中正常工作。你只需创建一次 Skill，它就能在你使用的所有 Claude 产品中发挥作用。

此外，Skills 还具备可组合性：Claude 能自动识别完成某项任务所需的技能，并无缝地将多个技能结合使用。但 Skills 真正强大之处在于：它只会在相关时才被加载。

你无需手动选择技能或告知 Claude 该使用哪一个——Claude 会扫描你已有的技能，判断当前任务需要哪些技能，然后只加载最低限度的必要信息。这样既能保证 Claude 的运行速度，又能让它在需要时获取你的专业经验。

接下来，我将通过实际案例展示它的工作原理。

## Claude Skills 的工作原理：真实案例
要理解 Skills 的工作方式，不妨看看我在实际项目中是如何使用这一功能的。

目前，我在 VS Code 中打开了一个基础网页项目，包含 index.html、styles.css 和 app.js 三个文件——相当于一张“白纸”。


![](https://fastly.jsdelivr.net/gh/bucketio/img0@main/2025/11/05/1762313966339-e0ef47c2-4c7e-4323-a476-b9d3b60fbf4d.png)


### 我要解决的问题
如果我想为这个网站添加一个专业的深色模式切换功能，该怎么做呢？

我需要的不是普通的深色模式，而是符合我特定实现标准的版本：平滑过渡效果、基于 localStorage 的偏好保存、语义化 HTML 结构，以及符合无障碍规范。

在有 Claude Skills 之前，我会打开 Claude Code，输入详细的提示词，逐一说明所有要求：

“在右上角添加一个深色模式切换按钮，用月亮/太阳图标表示。
使用 CSS 自定义属性实现主题功能。
通过 localStorage 保存用户偏好。
添加平滑过渡效果。
确保按钮带有合适的 ARIA 标签，符合无障碍要求……”

即便如此，Claude 仍可能遗漏某些细节，或者采用与我期望不同的实现方式。

现在，有了 Claude Skills 功能，我可以引导 Claude Code 按照我特定的方式创建这个功能。那么，具体该从何入手呢？

### 创建我的第一个 Skill
下面我将展示如何把上述需求转化为可复用的 Claude Skill，从此不再需要反复解释。

#### 步骤1：搭建 Skill 结构
我打开终端，创建了 Skill 文件夹，执行的命令如下：
```bash
mkdir -p ~/.claude/skills/darkmode-toggle
cd ~/.claude/skills/darkmode-toggle
```
按回车键或点击即可查看图片完整尺寸

#### 步骤2：编写 SKILL.md 文件
这一步是关键。我在名为 SKILL.md 的文件中，详细记录了我的具体实现模式。

我会用记事本打开这个新文件，并粘贴以下 Skill 代码：
```bash
notepad SKILL.md
```
执行该命令后会发生以下操作：
1. 打开记事本
2. 提示是否创建新文件（点击“是”）
3. 粘贴以下内容：

```markdown
---
name: Dark Mode Toggle
description: 为 HTML 网站添加专业的深色模式切换功能，支持平滑过渡和基于 localStorage 的偏好保存
version: 1.0.0
---

# 深色模式切换 Skill

## 概述
本 Skill 提供了为 HTML 网站添加深色模式功能的标准化实现。当用户需要添加深色模式、主题切换或明暗切换功能时，可使用本 Skill。

## 适用场景
- 用户要求“添加深色模式”
- 用户需要“主题切换”或“明暗切换”功能
- 用户提及让网站“支持深色模式”

## 实现模式

### HTML 结构
在 body 元素中添加以下切换按钮：
```html
<button id="theme-toggle" aria-label="切换深色模式">🌙</button>
```

### CSS 变量与主题
使用 CSS 自定义属性实现主题功能：
```css
:root {
  --bg-color: #ffffff; /* 浅色模式背景色 */
  --text-color: #000000; /* 浅色模式文本色 */
  --card-bg: #f5f5f5; /* 浅色模式卡片背景色 */
}

[data-theme="dark"] {
  --bg-color: #1a1a1a; /* 深色模式背景色 */
  --text-color: #ffffff; /* 深色模式文本色 */
  --card-bg: #2d2d2d; /* 深色模式卡片背景色 */
}

body {
  background-color: var(--bg-color);
  color: var(--text-color);
  /* 平滑过渡效果，时长 0.3 秒 */
  transition: background-color 0.3s ease, color 0.3s ease;
}

#theme-toggle {
  position: fixed; /* 固定定位 */
  top: 20px; /* 距离顶部 20px */
  right: 20px; /* 距离右侧 20px */
  padding: 10px 15px; /* 内边距 */
  border: none; /* 无边框 */
  border-radius: 50%; /* 圆形按钮 */
  background: var(--card-bg); /* 背景色继承卡片背景变量 */
  cursor: pointer; /* 鼠标悬停时显示指针 */
  font-size: 20px; /* 图标大小 */
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); /* 轻微阴影 */
  transition: transform 0.2s ease; /* 缩放过渡效果 */
}

#theme-toggle:hover {
  transform: scale(1.1); /* 鼠标悬停时放大 1.1 倍 */
}
```

### JavaScript 逻辑
结合 localStorage 实现偏好保存功能：
```javascript
const themeToggle = document.getElementById('theme-toggle');
// 获取当前主题，若无则默认浅色模式
const currentTheme = localStorage.getItem('theme') || 'light';

// 设置初始主题
document.documentElement.setAttribute('data-theme', currentTheme);
// 根据初始主题设置按钮图标
themeToggle.textContent = currentTheme === 'light' ? '🌙' : '☀️';

// 切换主题功能
themeToggle.addEventListener('click', () => {
  // 获取当前主题
  const theme = document.documentElement.getAttribute('data-theme');
  // 切换主题（浅色变深色，深色变浅色）
  const newTheme = theme === 'light' ? 'dark' : 'light';
  
  // 更新文档主题属性
  document.documentElement.setAttribute('data-theme', newTheme);
  // 保存主题到 localStorage
  localStorage.setItem('theme', newTheme);
  // 更新按钮图标
  themeToggle.textContent = newTheme === 'light' ? '🌙' : '☀️';
});
```

## 质量检查清单
- 切换按钮在右上角可见
- 主题切换时具有 0.3 秒的平滑过渡效果
- 页面刷新后仍能保留主题偏好
- 按钮图标随当前主题更新
- 所有现有元素能适配主题变量
- 支持键盘导航操作（符合无障碍要求）
```

粘贴完成后，按 Ctrl+S 保存并关闭文件。我将这个文件保存在 `~/.claude/skills/darkmode-toggle/` 目录下。

随后，通过以下命令验证文件是否正确保存：
```bash
Get-Content SKILL.md
```


正如你在上面的演示中看到的，我已成功添加 Skill 并完成验证，现在它已可以投入使用。

#### 步骤3：使用 Skill
接下来就是自动化的核心环节。

我回到 VS Code 中的项目（包含 index.html、styles.css、app.js 三个文件），打开 Claude Code，只需输入一句话：

“为这个网站添加深色模式”


背后的工作流程如下：

1. **Skill 发现**：Claude 扫描我的 `~/.claude/skills/` 文件夹，识别出“darkmode-toggle”Skill 与当前请求相关
2. **Skill 加载**：Claude 加载 SKILL.md 文件，读取我的实现模式
3. **智能应用**：Claude 分析现有文件，并应用 Skill 中的模式：
   - 向 HTML 文件中添加切换按钮
   - 将 CSS 变量整合到现有样式中，确保不破坏原有功能
   - 在 app.js 文件中实现 JavaScript 逻辑
4. **执行**：几秒钟内，我的网站就拥有了完全符合我要求的深色模式切换功能

### Claude Code 如何检测所需 Skill


当我输入“为这个网站添加深色模式”时，并没有告诉 Claude Code 要使用“darkmode-toggle”Skill。

Claude 会先扫描项目文件、分析我的请求，然后检查所有可用 Skill 的 YAML 前置元数据中的“description”（描述）字段。

我在元数据中写的描述——“为 HTML 网站添加专业的深色模式切换功能，支持平滑过渡和基于 localStorage 的偏好保存”——与我的查询完全匹配。



之后，Claude 会提示我确认：“是否使用 ‘darkmode-toggle’ Skill？”

这个权限验证步骤对安全性至关重要，尤其是当 Skill 包含可执行代码时。我可以选择“仅本次批准”“对整个项目批准”或“拒绝并提供其他指令”。

### 实际应用：按标准执行的 Claude Skill


我点击“是”批准使用该 Skill，接下来发生的事情令人惊喜：Claude Code 立即开始实现深色模式功能，但这一次，它不再临时做决策，而是严格按照 Skill 中的具体规范执行，就像遵循设计蓝图一样。

Claude 对三个文件进行了更新，具体如下：

#### HTML（index.html）—— 按规范添加切换按钮：
```html
<button id="theme-toggle" aria-label="切换深色模式">🌙</button>
```
注意按钮的 ID（theme-toggle）、用于无障碍的 ARIA 标签，以及月亮图标——所有细节都与我在 Skill 中定义的模式完全一致。

#### CSS（styles.css）—— 实现我的配色方案和定位要求：
```css
:root {
  --bg-color: #ffffff;
  --text-color: #000000;
  --card-bg: #f5f5f5;
}

[data-theme="dark"] {
  --bg-color: #1a1a1a;
  --text-color: #ffffff;
  --card-bg: #2d2d2d;
}
#theme-toggle {
  position: fixed;
  top: 20px;
  right: 20px;
  /* ... 其余样式与我在 Skill 中定义的完全一致 */
}
```
![](https://fastly.jsdelivr.net/gh/bucketio/img15@main/2025/11/05/1762314310263-3c7b82a5-c4be-4538-afb0-7fbdd054467b.png)

配色完全符合我的规范：浅色模式使用 #ffffff（白色）和 #000000（黑色），深色模式背景使用 #1a1a1a（深灰色）；按钮采用固定定位，距离顶部 20px、右侧 20px，同时包含我定义的悬停效果和过渡动画。

#### JavaScript（app.js）—— 采用我定义的 localStorage 逻辑：
```javascript
const themeToggle = document.getElementById('theme-toggle');
const currentTheme = localStorage.getItem('theme') || 'light';

document.documentElement.setAttribute('data-theme', currentTheme);
themeToggle.textContent = currentTheme === 'light' ? '🌙' : '☀️';
```
注意，代码使用的是我在 Skill 中指定的 `document.documentElement.setAttribute('data-theme', ...)` 方法，而非 `body.classList.add('dark-mode')` 或其他方式。

这就是我的标准，而 Claude 严格遵循了这一标准——这正是 Claude Skills 与 Claude Code 配合，实现工作流程标准化的核心方式。



最终的成果不只是“一个深色模式实现”，而是“我的深色模式实现”——它遵循了我在 Skill 中记录的每一个细节：

- 符合我的样式指南的精确颜色值
- 特定的定位要求
- 我偏好的 DOM 操作方法
- 我定义的无障碍标准
- localStorage 键名命名规范
- 0.3 秒的平滑过渡效果

借助 Claude Skills，你只需在 Claude Code 工作流程中记录一次自己的标准，此后无论你或团队中的任何人向 Claude Code 请求相关功能，它都会按照你的方式实现。

## Skill 的可组合性
这一功能还有更强大的用法。

假设我的 Skill 库中还有以下三个 Skill：
- `responsive-navbar`——我的标准化导航栏实现
- `form-validation`——带有我偏好的错误样式的客户端验证功能
- `api-integration`——团队专用的、包含错误处理的 fetch 封装工具

此时，我只需对 Claude Code 说：

“搭建一个包含导航栏、联系表单和深色模式的着陆页”

Claude 会自动识别出需要用到三个 Skill，加载它们并将其无缝组合使用。

## 总结思考
如今，每当我优化某个流程或发现更好的实现方法时，都会创建对应的 Claude Skill，以便在 Claude Code 中随时使用。

如果你仍在每次开启新项目时，手动向 Claude 重复解释自己的实现模式，那你的工作方式就太过低效了。相反，你应该开始构建自己的 Skill 库，消除 Claude Code 工作流程中的重复性任务。

如果你已准备好创建第一个 Skill，可按以下步骤操作：
1. **明确高频重复任务**——你经常需要向 Claude 重复解释的内容是什么？
2. **创建 Skill 文件夹**——执行命令 `mkdir -p ~/.claude/skills/your-skill-name`
3. **记录实现模式**——编写 SKILL.md 文件，包含清晰的指令
4. **测试验证**——通过 Claude Code 验证功能是否正常
5. **优化与扩展**——逐步完善并扩充你的 Skill 库

Claude Skills 和 Claude Code 的潜力才刚刚开始显现。如果你想了解更多 Claude Code 的使用技巧，并及时获取此类新功能的更新。
