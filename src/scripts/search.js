// 搜索功能模块
class SearchEngine {
    constructor(articles) {
        this.articles = articles;
        this.searchIndex = this.buildSearchIndex();
    }

    // 构建搜索索引
    buildSearchIndex() {
        const index = {};

        this.articles.forEach(article => {
            // 合并标题和摘要作为搜索内容
            const content = `${article.title} ${article.excerpt}`.toLowerCase();
            const words = content.split(/\s+/);

            words.forEach(word => {
                // 简单的词干提取（移除常见后缀）
                const stem = word.replace(/(ing|ed|ly|es|s)$/, '');

                if (!index[stem]) {
                    index[stem] = new Set();
                }
                index[stem].add(article.id);
            });
        });

        return index;
    }

    // 搜索文章
    search(query) {
        if (!query || query.trim() === '') {
            return this.articles;
        }

        const terms = query.toLowerCase().split(/\s+/);
        const matchedArticleIds = new Set();

        terms.forEach(term => {
            // 简单的词干提取
            const stem = term.replace(/(ing|ed|ly|es|s)$/, '');

            if (this.searchIndex[stem]) {
                this.searchIndex[stem].forEach(id => matchedArticleIds.add(id));
            }
        });

        // 如果没有匹配的词干，进行模糊匹配
        if (matchedArticleIds.size === 0) {
            this.articles.forEach(article => {
                const content = `${article.title} ${article.excerpt}`.toLowerCase();
                if (content.includes(query.toLowerCase())) {
                    matchedArticleIds.add(article.id);
                }
            });
        }

        // 返回匹配的文章，按日期倒序排列
        return this.articles
            .filter(article => matchedArticleIds.has(article.id))
            .sort((a, b) => new Date(b.date) - new Date(a.date));
    }
}

// 导出搜索引擎
window.SearchEngine = SearchEngine;