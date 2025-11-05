// Markdown解析器模块
class MarkdownParser {
    constructor() {
        // 文章数据存储
        this.articles = [];
        this.articleDetails = {};
    }

    // 初始化解析器
    async init() {
        // 加载实际文章文件
        await this.loadArticlesFromFiles();
        
        // 根据当前页面执行相应操作
        if (window.location.pathname.includes('article.html')) {
            await this.renderArticleDetails();
        } else {
            this.renderArticleList();
        }
    }

    // 从文件系统加载文章
    async loadArticlesFromFiles() {
        // 由于浏览器安全限制，我们不能直接列出目录内容
        // 这里我们预定义一个文章文件列表（在实际项目中，这个列表可以由后端生成）
        const articleFiles = [
            '第183期 我如何利用 Claude Skills（新功能）优化我的 Claude Code 工作流程.md',
            '第182期 Claude Skills：深入探究Anthropic的智能体框架.md',
            '第181期 何时使用 Claude Skills（LLM的工具箱） 与 MCPs（LLM的USB接口）？.md',
            '第180期 智能体人工智能（Agentic AI）：单智能体系统 vs 多智能体系统《基于LangGraph与结构化数据源构建智能体系统》.md',
            '第179期 在生产环境运行AI智能体花了4.7万元：关于A2A和MCP，那些没人会告诉你的事.md',
            '第178期 如何使用AI做代码开发-claude code让我翻倍效率.md',
            '第177期 深度解析：OpenAI推出GPT-5驱动的Aardvark，重新定义智能体安全研究.md',
            '第176期 Claude Code 性能强劲——6个月重构100w行代码（经验与教训贴）.md',
            '第175期 超越提示工程：面向稳健多目标人工智能智能体的神经-符号-因果架构.md',
            '第174期 TIMM：让迁移学习变得异常简单的PyTorch"隐藏"库.md',
            '第173期 [实战项目]基于Python和LSTM的人工智能股市预测.md',
            '第172期 解读苹Apple 核心安全模型.md',
            '第171期 到底神经网络究竟是如何"学习"的？.md',
            '第170期 如何在1个月内学习人工智能与大语言模型.md',
            '第169期 如何从零开始学习大语言模型（LLMs）.md',
            '第168期 人工智能工程师：初学者的务实学习路线图.md',
            '第167期 Python 人工智能教程  Python 人工智能编程.md',
            '第166期 赶快使用AI编程效率提升3倍.md',
            '第165期 无需提示词的微调：Bonepoke 与系统姿态的隐藏调控旋钮.md',
            '第164期 SFT、DFO、PEFT、GRPO对比：为大语言模型选择合适的微调策略.md',
            '第163期 微调简化指南：让你的AI实现专业化.md',
            '第162期 自定义目标检测的 YOLO 微调完整指南.md',
            '第161期 知识困境：人工智能为何越来越笨.md',
            '第160期 如何训练你的大语言模型：使用 Unsloth 进行低秩适配微调！.md',
            '第159期 如何将 TFRecord 数据集转换为 ArrayRecord（并使用 Grain 构建快速数据管道）.md',
            '第158期 二分类任务中不平衡数据集的重采样：真的值得吗？.md',
            '第157期 构建交易分析数据集：从原始股票数据到可落地的洞察.md',
            '第156期 适用于RAG的最佳开源嵌入模型 多语言自然语言处理及阿拉伯语文本的高性能开源嵌入模型.md',
            '第155期 未来AI工程师必看的10篇论文-解析塑造该领域的10篇顶尖论文（未读就是不合格的算法工程师）.md',
            '第154期 学生必看：12款高效学习的最佳AI工具.md',
            '第153期 这3款AI工具，没它们我真的不行——即刻提升工作效率.md',
            '第152期 AI正在入侵和理解你，你还有秘密吗？.md',
            '第151期 强化学习：实现99%最优策略的秘诀？利用强化学习与先进机器学习优化序贯决策.md',
            '第150期 我如何用Python开发出一款AI工具，赚到了第一笔1000美元.md',
            '第149期 企业用人工智能取代团队后，发生了这些事.md',
            '第148期 帮你打造这个系统——智能SCADA安全系统（我的经验能帮到你）.md',
            '第147期 如何在AI Agent中构建长期记忆（最新技术研究）.md',
            '第146期《2025年AI现状报告》解读（四）：调研篇.md',
            '第145期《2025年AI现状报告》解读（三）：安全篇.md',
            '第144期《2025年AI现状报告》解读（二）：产业篇.md',
            '第143期 《2025年AI现状报告》解读（一）：研究篇.md',
            '第142期 mini-swe-agent：极简AI编程代理的崛起.md',
            '第141期 在 AI 世界中什么是 MCP？一个简单的解释.md',
            '第140期 工作场所中AI的影响：数据会说话.md',
            '第139期 掌握这 5 项 AI 技能，解锁 2025 年 80% 的人工智能.md',
            '第138期 人工智能泡沫即将破裂吗？.md',
            '第137期 快速学习人工智能的5本最佳书籍.md',
            '第136期 谷歌Jules Tools反击Copilot的主导地位：重新定义工作流自动化.md',
            '第135期 人工智能在10个方面超越了我，而我却毫不在意（再不学习就落伍了）.md',
            '第134期 2025年人工智能副业如何发展为全职事业.md',
            '第133期 人工智能正在悄悄帮人赚钱的10种方式（鲜有人提及）.md',
            '第132期 人工智能如何洞悉你的恐惧：个性化背后的阴暗面.md',
            '第131期 因果推断：规避偏差与无根据的关联假设.md',
            '第130期 为何对中小企业而言，AI智能体比数据团队更高效.md',
            '第129期 构建SQL-of-Thought：具备引导式错误修正功能的多智能体文本转SQL系统.md',
            '第128期 到2026年每位资深IT专业人员都必须掌握的顶级AI工具与框架.md',
            '第127期 10个你可能不知道自己需要的Python自动化项目.md',
            '第126期 向量数据库高级教程（检索以及相似度计算）.md',
            '第125期借助 n8n 构建 AI 智能体：10个自动化实践方案.md',
            '第124期适合你实践的 10 个 LLM 与 RAG 项目.md',
            '第123期自学出身且经验不足的机器学习工程师的5个典型特征.md',
            '第122期构建能解决复杂任务的AI智能体，不止需要简单的编排.md',
            '第121期借助AI快速试错-AI辅助设计开发原型时代的核心法则.md',
            '第120期将网站转化为适用于大语言模型（LLM）的知识库.md',
            '第119期 Perplexity 重磅推出 10 款免费 AI 智能体：包揽你的全部工作（"Comet" 捷径指南）.md',
            '第118期 模型上下文协议（MCP）——使用Java构建SQL数据库代理（MCP代理教程）.md',
            '第117期 MCP服务无法正常运行？一招解决的方法｜Claude MCP智能体教程.md'
        ];

        const articles = [];
        
        // 由于我们不能在浏览器中直接读取文件内容，我们需要使用fetch API
        // 这里我们模拟异步加载文章元数据
        for (let i = 0; i < articleFiles.length; i++) {
            const filename = articleFiles[i];
            // 从文件名提取期数作为ID
            const match = filename.match(/第(\d+)期/);
            const id = match ? match[1] : `${i + 1}`;
            
            // 提取标题（去掉期数和.md扩展名）
            const title = filename.replace(/^第\d+期[\s-]+/, '').replace(/\.md$/, '').trim();
            
            // 生成一个基于期数的模拟日期（期数越大，日期越新）
            const date = this.generateDateFromId(parseInt(id));
            
            // 生成摘要
            const excerpt = `这是《${title}》的文章摘要，点击阅读全文了解更多详情。`;
            
            articles.push({
                id,
                title,
                excerpt,
                date,
                filename
            });
        }
        
        // 按日期倒序排序（最新的在前）
        this.articles = articles.sort((a, b) => new Date(b.date) - new Date(a.date));
    }
    
    // 根据文章ID生成模拟日期
    generateDateFromId(id) {
        // 假设第117期对应2024年1月1日，每增加一期，日期增加一天
        const baseDate = new Date('2024-01-01');
        const daysToAdd = id - 117;
        baseDate.setDate(baseDate.getDate() + daysToAdd);
        
        const year = baseDate.getFullYear();
        const month = String(baseDate.getMonth() + 1).padStart(2, '0');
        const day = String(baseDate.getDate()).padStart(2, '0');
        
        return `${year}-${month}-${day}`;
    }

    // 渲染文章列表（已按日期倒序排列）
    renderArticleList() {
        const container = document.getElementById('article-container');
        if (!container) return;

        container.innerHTML = '';

        if (this.articles.length === 0) {
            container.innerHTML = '<p>暂无文章</p>';
            return;
        }

        // 文章已在loadArticlesFromFiles方法中按日期倒序排序
        this.articles.forEach(article => {
            const articleCard = document.createElement('article');
            articleCard.className = 'article-card';
            articleCard.innerHTML = `
                <h3><a href="article.html?id=${article.id}">${article.title}</a></h3>
                <p class="article-date">${article.date}</p>
                <p class="article-excerpt">${article.excerpt}</p>
                <a href="article.html?id=${article.id}" class="read-more">阅读更多</a>
            `;
            container.appendChild(articleCard);
        });
    }

    // 渲染文章详情
    async renderArticleDetails() {
        const urlParams = new URLSearchParams(window.location.search);
        const articleId = urlParams.get('id');

        const container = document.getElementById('article-content');
        if (!container || !articleId) {
            container.innerHTML = '<p>文章不存在</p>';
            return;
        }

        // 查找当前文章信息
        const article = this.articles.find(a => a.id === articleId);
        if (!article) {
            container.innerHTML = '<p>文章不存在</p>';
            return;
        }

        // 设置文章标题和日期
        const titleElement = document.querySelector('.article-title');
        const dateElement = document.querySelector('.article-date');
        if (titleElement) titleElement.textContent = article.title;
        if (dateElement) dateElement.textContent = article.date;

        try {
            // 尝试从articles目录加载文章内容
            const response = await fetch(`./articles/${article.filename}`);
            if (!response.ok) {
                throw new Error('Failed to fetch article');
            }
            const markdownContent = await response.text();
            
            // 解析Markdown
            const htmlContent = this.parseMarkdown(markdownContent);
            container.innerHTML = htmlContent;
        } catch (error) {
            console.error('Error loading article:', error);
            // 如果无法加载实际文章内容，显示文章信息和提示
            container.innerHTML = `
                <p>文章ID: ${article.id}</p>
                <p>文件名: ${article.filename}</p>
                <p>提示: 由于浏览器安全限制，在本地开发环境中可能无法直接读取Markdown文件内容。</p>
                <p>部署到GitHub Pages后，此功能将正常工作。</p>
            `;
        }
    }

    // 简单的Markdown解析（在没有marked.js的情况下使用）
    parseMarkdown(markdown) {
        let html = markdown;
        
        // 标题
        html = html.replace(/^(#{1,6})\s+([^\n]+)/gm, (match, hashes, content) => {
            const level = hashes.length;
            return `<h${level}>${content}</h${level}>`;
        });
        
        // 粗体
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        
        // 斜体
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        
        // 链接
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
        
        // 代码块
        html = html.replace(/```([^`]+)```/g, (match, code) => {
            const [language, codeContent] = code.split('\n', 2);
            return `<pre><code class="language-${language || 'plaintext'}">${codeContent || code}</code></pre>`;
        });
        
        // 行内代码
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // 段落
        html = html.replace(/^(?!<h|<pre|<ul|<ol|<blockquote|<p)([\s\S]*?)(?=\n\n|$)/gm, '<p>$1</p>');
        
        // 无序列表
        html = html.replace(/^\*\s+([^\n]+)/gm, '<li>$1</li>');
        html = html.replace(/(<li>[\s\S]*?)(?=\n\n|<li|$)/g, '<ul>$1</ul>');
        
        // 有序列表
        html = html.replace(/^\d+\.\s+([^\n]+)/gm, '<li>$1</li>');
        html = html.replace(/(<li>[\s\S]*?)(?=\n\n|<li|$)/g, (match) => {
            if (!match.includes('<ul>') && !match.includes('</ul>')) {
                return '<ol>' + match + '</ol>';
            }
            return match;
        });
        
        // 引用
        html = html.replace(/^>\s+([^\n]+)/gm, '<blockquote><p>$1</p></blockquote>');
        
        return html;
    }
}

// 创建实例并导出
const markdownParser = new MarkdownParser();

// 导出供其他模块使用
window.markdownParser = markdownParser;