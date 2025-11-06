// Markdown解析器模块
class MarkdownParser {
    constructor() {
        // 文章数据存储
        this.articles = [];
        this.articleDetails = {};
        this.currentPage = 1;
        this.articlesPerPage = 10; // 每页显示10篇文章
    }

    // 加载所有文章
    async loadAllArticles() {
        try {
            // 清空当前文章列表
            this.articles = [];

            // 获取所有文章文件
            const allArticles = [
                // 这里应该从articles目录动态获取所有文件
                // 由于是静态网站，我们手动列出一些示例文章
                '第117期 MCP服务无法正常运行？一招解决的方法｜Claude MCP智能体教程.md',
                '第118期 模型上下文协议（MCP）——使用Java构建SQL数据库代理（MCP代理教程）.md',
                '第119期 Perplexity 重磅推出 10 款免费 AI 智能体：包揽你的全部工作（“Comet” 捷径指南）.md',
                '第120期将网站转化为适用于大语言模型（LLM）的知识库.md',
                '第121期借助AI快速试错-AI辅助设计开发原型时代的核心法则.md',
                '第122期构建能解决复杂任务的AI智能体，不止需要简单的编排.md',
                '第123期自学出身且经验不足的机器学习工程师的5个典型特征.md',
                '第124期适合你实践的 10 个 LLM 与 RAG 项目.md',
                '第125期借助 n8n 构建 AI 智能体：10个自动化实践方案.md',
                '第126期 向量数据库高级教程（检索以及相似度计算）.md'
            ];

            // 模拟文章数据
                for (const file of allArticles) {
                    const id = file.split('.')[0];
                    const title = file.replace('.md', '');
                    const date = this.generateDateFromId(id);
                    const excerpt = `这是${title}的摘要内容，包含了文章的主要观点和关键信息。`;
                    const category = this.getCategoryFromTitle(title);
                    const tags = this.getTagsFromTitle(title);

                    this.articles.push({
                        id,
                        title,
                        date,
                        excerpt,
                        category,
                        tags,
                        folder: 'articles' // 设置folder属性
                    });
            }

            // 按日期倒序排序
            this.articles.sort((a, b) => new Date(b.date) - new Date(a.date));

        } catch (error) {
            console.error('加载文章失败:', error);
        }
    }

    // 从标题获取分类
    getCategoryFromTitle(title) {
        if (title.includes('AI') || title.includes('智能体') || title.includes('LLM') || title.includes('大语言模型')) {
            return 'AI技术';
        } else if (title.includes('数据结构') || title.includes('算法')) {
            return '数据结构';
        } else if (title.includes('编程') || title.includes('开发')) {
            return '编程技术';
        }
        return '技术文章';
    }

    // 从标题获取标签
    getTagsFromTitle(title) {
        const tags = [];
        if (title.includes('AI') || title.includes('人工智能')) tags.push('AI');
        if (title.includes('智能体')) tags.push('智能体');
        if (title.includes('LLM') || title.includes('大语言模型')) tags.push('大语言模型');
        if (title.includes('教程')) tags.push('教程');
        if (title.includes('数据结构')) tags.push('数据结构');
        if (title.includes('算法')) tags.push('算法');
        if (tags.length === 0) tags.push('技术');
        return tags;
    }
    
    // 加载并显示AI编程相关的文章
    async loadAIProgrammingArticles() {
        const container = document.getElementById('article-container');
        if (!container) return;

        container.innerHTML = '<p>正在加载文章...</p>';

        // 只加载AI编程专栏的文章
        await this.loadArticlesFromFolder('AI_Programming_Tutorial');

        // 筛选AI编程专栏文章，使用folder属性确保与其他专栏一致
        const aiProgrammingArticles = this.articles.filter(article => 
            article.folder === 'AI_Programming_Tutorial'
        );
        
        if (aiProgrammingArticles.length === 0) {
            container.innerHTML = '<p>暂无AI编程相关文章</p>';
            return;
        }

        // 渲染文章列表
        this.renderArticles(aiProgrammingArticles, container);
    }

    // 加载并显示数据结构相关的文章
    async loadDataStructureArticles() {
        const container = document.getElementById('article-container');
        if (!container) return;

        container.innerHTML = '<p>正在加载文章...</p>';

        // 只加载数据结构专栏的文章
        await this.loadArticlesFromFolder('DataStructure_Tutorial');

        // 筛选数据结构专栏文章
        const dataStructureArticles = this.articles.filter(article => 
            article.folder === 'DataStructure_Tutorial'
        );
        
        if (dataStructureArticles.length === 0) {
            container.innerHTML = '<p>暂无数据结构相关文章</p>';
            return;
        }

        // 渲染文章列表
        this.renderArticles(dataStructureArticles, container);
    }

    // 从指定文件夹加载文章
    async loadArticlesFromFolder(folder) {
        try {
            // 清空当前文章列表
            this.articles = [];
            
            let files;
            // 根据文件夹确定文件列表
            if (folder === 'AI_Programming_Tutorial') {
                files = [
                    '第0期 AI编程教程大纲.md',
                    '第1期 从代码到对话：编程的范式革命.md',
                    '第2期 什么是大型语言模型（LLM）？核心能力解析.md',
                    '第4期 AI编程的优势、局限性与伦理考量.md',
                    '第7期 提示工程的基本原则与模式.md',
                    '第8期 提示词优化与指令工程.md',
                    '第9期 上下文工程与多轮交互技巧.md',
                    '第10期 代码生成的应用场景.md',
                    '第10期 提示词实验与效果评估.md',
                    '第11期 代码生成提示词的设计原则.md',
                    '第12期 常见编程语言的提示技巧.md',
                    '第13期 代码生成的高级技巧.md',
                    '第14期 调试与问题解决.md',
                    '第15期 代码质量优化.md',
                    '第16期 算法与性能优化.md',
                    '第17期 代码调试与问题诊断.md',
                    '第18期 开发环境集成：IDE插件与扩展.md',
                    '第19期 版本控制与AI：智能提交与代码审查.md',
                    '第20期 文档生成：从代码到知识库.md',
                    '第21期 测试用例生成：自动化测试流程.md',
                    '第22期 安全编码实践与AI辅助安全审计.md',
                    '第23期 敏捷开发与DevOps实践中的AI应用.md',
                    '第24期 构建个人AI编程系统：提示库与工作流定制.md'
                ];
            } else if (folder === 'DataStructure_Tutorial') {
                files = [
                    '第0期 数据结构与算法教程大纲.md',
                    '第1期 入门篇.md',
                    '第2期 数组.md',
                    '第3期 链表.md',
                    '第4期 栈.md',
                    '第5期 队列.md',
                    '第6期 递归.md',
                    '第7期 排序.md',
                    '第8期 二分查找.md',
                    '第9期 跳表.md',
                    '第10期 散列表.md',
                    '第11期 哈希算法.md',
                    '第12期 二叉树基础.md',
                    '第13期 红黑树.md',
                    '第14期 递归树.md',
                    '第15期 堆.md',
                    '第16期 图的表示.md',
                    '第17期 深度和广度优先搜索.md',
                    '第18期 字符串匹配基础.md',
                    '第19期 Trie树.md',
                    '第20期 AC自动机.md',
                    '第21期 贪心算法.md',
                    '第22期 分治算法.md',
                    '第23期 回溯算法.md',
                    '第24期 动态规划.md',
                    '第25期 拓扑排序.md',
                    '第26期 有权图的应用：最短路径.md',
                    '第27期 位图&布隆过滤器.md',
                    '第28期 B+树.md',
                    '第29期 索引.md',
                    '第30期 并行算法.md',
                    '第31期 Redis用到的数据结构.md',
                    '第32期 搜索引擎的理论基础.md',
                    '第33期 高性能队列Disruptor.md',
                    '第34期 微服务的鉴权限流接口.md',
                    '第35期 短网址服务系统.md',
                    '第36期 权衡选择数据结构和算法.md',
                    '第37期 leetcode练习题.md'
                ];
            } else if (folder === 'LLM_Agent_mcp_skills_Tutorial') {
                files = [
                    '第0期 大纲介绍.md',
                    '第1期 MCP协议概述与基础概念.md',
                    '第4期 MCP服务器高级应用与最佳实践.md',
                    '第16期 RAG系统部署与生产环境优化.md',
                    '第5期 监督微调(SFT)：LLM微调基础技术详解.md',
                    '第6期 直接偏好优化(DPO)：轻量级人类偏好对齐技术.md',
                    '第7期 基于人类反馈的强化学习(RLHF)：高级偏好对齐方法.md',
                    '第8期 参数高效微调：LoRA与QLoRA技术详解.md',
                    '第9期 多智能体系统设计：协作式智能体架构与实现.md',
                    '第10期 规划型智能体：基于推理与规划的任务解决方法.md',
                    '第11期 工具使用型智能体：API与外部工具集成指南.md',
                    '第12期 记忆增强型智能体：长期记忆与学习能力构建.md',
                    '第13期 RAG技术基础：检索增强生成的原理与架构.md',
                    '第14期 RAG高级优化：提升检索质量与生成效果的策略.md',
                    '第15期 RAG高级技术：混合检索与重排序策略.md',
                    '第16期 RAG系统部署与生产环境优化.md',
                    '第17期 提示工程基础：设计有效提示词的核心原则.md',
                    '第18期 高级提示工程技术：推理、多模态与动态提示构建.md',
                    '第19期 提示工程实践：行业应用案例与实施指南.md',
                    '第20期 提示工程未来展望：趋势、研究方向与企业级能力建设.md',
                    '第21期 AI系统部署基础：架构设计与最佳实践.md',
                    '第22期 模型优化技术：量化、剪枝与知识蒸馏.md',
                    '第23期 AI系统监控与维护：性能监控、漂移检测与可靠性保障.md'
                ];
            } else {
                return;
            }
            
            const baseFolder = 'articles';
            const articleFolder = `${baseFolder}/${folder}`;
            
            // 为每个文件创建文章数据
            for (const file of files) {
                const id = file.split('.')[0];
                const title = file.replace('.md', '');
                const date = this.generateDateFromId(id);
                const excerpt = `这是${title}的摘要内容，包含了文章的核心概念和实践方法。`;
                const category = this.getCategorizeFromTitle(title, folder);
                const tags = this.getTagsFromTitle(title, folder);

                // 添加到文章列表，同时设置folder和column属性，确保兼容性
                this.articles.push({
                    id,
                    title,
                    date,
                    excerpt,
                    category,
                    tags,
                    folder: folder,
                    column: folder
                });
            }
            
            // 按日期倒序排序
            this.articles.sort((a, b) => new Date(b.date) - new Date(a.date));
            
        } catch (error) {
            console.error('加载文章失败:', error);
        }
    }
    
    // 从ID生成日期（用于排序）
    generateDateFromId(id) {
        const now = new Date();
        // 从ID中提取数字部分作为天数
        const match = id.match(/\d+/);
        const daysToSubtract = match ? parseInt(match[0]) : 0;
        const date = new Date(now);
        date.setDate(date.getDate() - (daysToSubtract % 30)); // 限制在30天内循环
        return date.toISOString().split('T')[0];
    }
    
    // 从标题获取分类
    getCategorizeFromTitle(title, folder) {
        if (folder === 'AI_Programming_Tutorial') {
            return 'AI编程';
        } else if (folder === 'DataStructure_Tutorial') {
            return '数据结构';
        } else if (folder === 'LLM_Agent_mcp_skills_Tutorial') {
            return 'LLM智能体';
        }
        return '编程';
    }
    
    // 从标题获取标签
    getTagsFromTitle(title, folder) {
        if (folder === 'AI_Programming_Tutorial') {
            return ['AI', '编程', '教程'];
        } else if (folder === 'DataStructure_Tutorial') {
            return ['数据结构', '算法', '教程'];
        } else if (folder === 'LLM_Agent_mcp_skills_Tutorial') {
            return ['LLM', '智能体', 'MCP', '教程'];
        }
        return ['编程'];
    }
    
    // 渲染文章列表
    renderArticles(articles, container) {
        if (!container) return;

        container.innerHTML = '';
        const articleGrid = document.createElement('div');
        articleGrid.className = 'article-grid';

        articles.forEach(article => {
            const articleCard = document.createElement('div');
            articleCard.className = 'article-card'; // 使用统一的CSS类名

            // 处理可能不存在的folder属性
            const folder = article.folder || '';
            const id = article.id || '';

            articleCard.innerHTML = `
                <div class="article-card-content">
                    <h3><a href="/article.html?id=${encodeURIComponent(id)}&folder=${encodeURIComponent(folder)}">${article.title}</a></h3>
                    <div class="article-date">${article.date}</div>
                    <div class="article-excerpt">${article.excerpt}</div>
                    <div class="article-tags">
                        <span class="category-tag">${article.category}</span>
                        ${article.tags ? article.tags.map(tag => `<span class="tag">${tag}</span>`).join('') : ''}
                    </div>
                </div>
            `;
            articleGrid.appendChild(articleCard);
        });

        container.appendChild(articleGrid);
    }
    
    // 加载文章详情
    async loadArticleDetails(id, folder) {
        try {
            // 首先确保文章数据已加载
            if (this.articles.length === 0 || folder) {
                if (folder === 'LLM_Agent_mcp_skills_Tutorial') {
                    await this.loadArticlesFromFolder('LLM_Agent_mcp_skills_Tutorial');
                } else if (folder === 'AI_Programming_Tutorial') {
                    await this.loadArticlesFromFolder('AI_Programming_Tutorial');
                } else if (folder === 'DataStructure_Tutorial') {
                    await this.loadArticlesFromFolder('DataStructure_Tutorial');
                } else {
                    await this.loadAllArticles();
                }
            }

            // 根据ID和folder查找文章
            const articleInfo = this.articles.find(a => a.id === decodeURIComponent(id) && (!folder || a.folder === folder));

            if (!articleInfo) {
                throw new Error('文章不存在');
            }

            // 生成更详细的文章内容
            let content = `# ${articleInfo.title}\n\n`;
            content += `${articleInfo.excerpt}\n\n`;
            content += `这篇文章详细介绍了**${articleInfo.title}**的相关内容。`;
            content += `\n\n## 核心要点\n\n`;
            content += `- **技术要点1**：详细解释了相关技术的核心概念\n`;
            content += `- **技术要点2**：提供了实际应用的示例和最佳实践\n`;
            content += `- **技术要点3**：分析了当前技术的发展趋势和未来展望\n\n`;
            content += `## 实践指南\n\n`;
            content += `在实际开发中，建议按照以下步骤进行：\n\n`;
            content += `1. 理解基础概念\n`;
            content += `2. 参考官方文档\n`;
            content += `3. 动手实践\n`;
            content += `4. 总结经验\n\n`;
            content += `## 总结\n\n`;
            content += `通过本文的学习，您应该对${articleInfo.title}有了深入的理解。`;
            content += `建议继续关注相关技术的发展，不断提升自己的技能水平。`;

            return {
                title: articleInfo.title,
                date: articleInfo.date,
                category: articleInfo.category,
                tags: articleInfo.tags,
                content: content
            };
        } catch (error) {
            console.error('加载文章详情失败:', error);
            throw error;
        }
    }
    
    // 渲染文章详情
    async renderArticleDetails() {
        try {
            const urlParams = new URLSearchParams(window.location.search);
            const id = urlParams.get('id');
            const folder = urlParams.get('folder');

            // ID是必需的参数
            if (!id) {
                const container = document.getElementById('article-content');
                if (container) {
                    container.innerHTML = '<p>缺少文章ID参数</p>';
                }
                return;
            }

            // folder参数是可选的，因为我们现在在所有文章中查找
            const articleDetails = await this.loadArticleDetails(id, folder || '');

            if (!articleDetails) {
                const container = document.getElementById('article-content');
                if (container) {
                    container.innerHTML = '<p>文章不存在</p>';
                }
                return;
            }

            // 设置标题和日期
            const titleElement = document.getElementById('article-title');
            const dateElement = document.getElementById('article-date');
            const contentElement = document.getElementById('article-content');

            if (titleElement) titleElement.textContent = articleDetails.title;
            if (dateElement) dateElement.textContent = articleDetails.date;

            // 渲染分类和标签
            const categoriesContainer = document.getElementById('article-categories');
            if (categoriesContainer) {
                categoriesContainer.innerHTML = '';
                const categoryTag = document.createElement('span');
                categoryTag.className = 'category-tag';
                categoryTag.textContent = articleDetails.category;
                categoriesContainer.appendChild(categoryTag);
            }

            const tagsContainer = document.getElementById('article-tags');
            if (tagsContainer) {
                tagsContainer.innerHTML = '';
                articleDetails.tags.forEach(tag => {
                    const tagElement = document.createElement('span');
                    tagElement.className = 'tag';
                    tagElement.textContent = tag;
                    tagsContainer.appendChild(tagElement);
                });
            }

            // 解析并渲染Markdown内容
            if (contentElement) {
                contentElement.innerHTML = this.parseMarkdown(articleDetails.content);
            }

        } catch (error) {
            console.error('渲染文章详情失败:', error);
            const contentElement = document.getElementById('article-content');
            if (contentElement) {
                contentElement.innerHTML = '<p>加载文章失败，请稍后重试</p>';
            }
        }
    }
    
    // 简单的Markdown解析
    parseMarkdown(text) {
        return text
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2">$1</a>')
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/^- (.*$)/gm, '<li>$1</li>')
            .replace(/<\/li>\s*<li>/g, '</li><li>')
            .replace(/^<li>(.*$)/gm, '<ul><li>$1</li></ul>')
            .replace(/<\/ul>\s*<ul>/g, '');
    }
}

// 初始化并暴露方法
const markdownParser = new MarkdownParser();
window.MarkdownParser = MarkdownParser;
window.markdownParser = markdownParser;

// 全局方法
window.loadAllArticles = async function() {
    // 加载所有文章
    await markdownParser.loadAllArticles();

    // 渲染文章列表
    const container = document.getElementById('article-container');
    if (container && markdownParser.articles.length > 0) {
        markdownParser.renderArticles(markdownParser.articles, container);
    } else if (container) {
        container.innerHTML = '<p>暂无文章</p>';
    }
};

window.loadAIProgrammingArticles = async function() {
    await markdownParser.loadAIProgrammingArticles();
};

window.loadDataStructureArticles = async function() {
    await markdownParser.loadDataStructureArticles();
};

window.renderArticleDetails = async function() {
    await markdownParser.renderArticleDetails();
};

// 页面加载时不需要自动初始化，由各页面单独调用相应方法