# 第四部分：RAG技术实战与优化
## 第15期 RAG高级技术：混合检索与重排序策略

## 4.3 RAG高级技术：混合检索与重排序策略

在前两篇文章中，我们介绍了RAG技术的基础架构和优化策略。随着RAG技术的发展，各种高级检索技术不断涌现，旨在解决传统语义检索的局限性。本文将深入探讨RAG系统中的混合检索与重排序策略，这些技术能够显著提升检索质量和回答准确性。

### 1. 混合检索技术原理

传统的RAG系统主要依赖于向量相似度检索，但这种方法在处理某些类型的查询时存在局限性。混合检索通过结合多种检索方法的优势，能够更全面地捕捉文档与查询之间的相关性。

#### 1.1 混合检索的核心思想

混合检索的核心思想是：不同的检索方法在不同场景下各有优势，将它们结合起来可以获得更好的整体性能。主要包括以下几种检索方法的组合：

- **语义检索**：捕捉深层语义相关性
- **关键词检索**：精确匹配关键术语
- **混合语义-关键词检索**：结合两者优势

#### 1.2 常见的混合检索策略

```python
class HybridRetrievalSystem:
    def __init__(self, vector_store, keyword_store, fusion_method="rrf"):
        self.vector_store = vector_store  # 向量数据库
        self.keyword_store = keyword_store  # 关键词搜索引擎（如Elasticsearch）
        self.fusion_method = fusion_method  # 结果融合方法
    
    def vector_search(self, query, top_k=10):
        """执行向量相似度检索"""
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return [(doc, score) for doc, score in results]
    
    def keyword_search(self, query, top_k=10):
        """执行关键词检索"""
        # 这里假设keyword_store提供了搜索方法
        results = self.keyword_store.search(query, size=top_k)
        return [(doc, score) for doc, score in results]
    
    def reciprocal_rank_fusion(self, vector_results, keyword_results, top_k=5):
        """使用倒数排序融合(RRF)方法合并结果
        
        RRF公式: score = sum(1/(k + rank))，其中k通常为60
        """
        k = 60  # RRF参数
        fused_scores = {}
        
        # 处理向量检索结果
        for rank, (doc, _) in enumerate(vector_results, 1):
            doc_id = doc.metadata.get("id", str(id(doc)))
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0}
            fused_scores[doc_id]["score"] += 1 / (k + rank)
        
        # 处理关键词检索结果
        for rank, (doc, _) in enumerate(keyword_results, 1):
            doc_id = doc.metadata.get("id", str(id(doc)))
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0}
            fused_scores[doc_id]["score"] += 1 / (k + rank)
        
        # 按融合分数排序并返回
        sorted_results = sorted(
            fused_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        return [(item["doc"], item["score"]) for item in sorted_results[:top_k]]
    
    def linear_fusion(self, vector_results, keyword_results, weights=(0.6, 0.4), top_k=5):
        """线性加权融合结果"""
        vector_weight, keyword_weight = weights
        doc_scores = {}
        
        # 归一化向量分数
        if vector_results:
            max_vector_score = max(score for _, score in vector_results)
        else:
            max_vector_score = 1
        
        # 归一化关键词分数
        if keyword_results:
            max_keyword_score = max(score for _, score in keyword_results)
        else:
            max_keyword_score = 1
        
        # 计算向量结果的加权分数
        for doc, score in vector_results:
            doc_id = doc.metadata.get("id", str(id(doc)))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "vector_score": 0, "keyword_score": 0}
            doc_scores[doc_id]["vector_score"] = (score / max_vector_score) * vector_weight
        
        # 计算关键词结果的加权分数
        for doc, score in keyword_results:
            doc_id = doc.metadata.get("id", str(id(doc)))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "vector_score": 0, "keyword_score": 0}
            doc_scores[doc_id]["keyword_score"] = (score / max_keyword_score) * keyword_weight
        
        # 计算最终分数并排序
        results = []
        for doc_id, scores in doc_scores.items():
            final_score = scores["vector_score"] + scores["keyword_score"]
            results.append((scores["doc"], final_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search(self, query, top_k=5, **kwargs):
        """执行混合检索"""
        # 并行执行两种检索
        vector_results = self.vector_search(query, top_k=top_k * 3)  # 获取更多结果用于融合
        keyword_results = self.keyword_search(query, top_k=top_k * 3)
        
        # 根据指定的融合方法合并结果
        if self.fusion_method == "rrf":
            return self.reciprocal_rank_fusion(vector_results, keyword_results, top_k=top_k)
        else:  # linear
            weights = kwargs.get("weights", (0.6, 0.4))
            return self.linear_fusion(vector_results, keyword_results, weights=weights, top_k=top_k)
```

### 2. BM25与语义检索的结合

BM25作为经典的关键词检索算法，与语义检索的结合是混合检索中最常用的策略之一。

#### 2.1 BM25原理回顾

BM25（Best Matching 25）是一种基于概率检索模型的算法，它考虑了：
- 词项在文档中的频率
- 文档长度的归一化
- 词项在整个集合中的逆文档频率

#### 2.2 BM25与语义检索的集成实现

```python
class BM25SemanticHybrid:
    def __init__(self, documents, embedding_model):
        # 初始化BM25
        from rank_bm25 import BM25Okapi
        self.tokenized_corpus = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.documents = documents
        
        # 初始化嵌入模型
        self.embedding_model = embedding_model
        # 预计算文档嵌入
        self.document_embeddings = self.embedding_model.encode(documents)
    
    def bm25_search(self, query, top_k=10):
        """执行BM25检索"""
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 获取排序后的文档索引
        sorted_indices = bm25_scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            results.append((self.documents[idx], float(bm25_scores[idx]), idx))
        
        return results
    
    def semantic_search(self, query, top_k=10):
        """执行语义检索"""
        import numpy as np
        
        # 计算查询嵌入
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 计算相似度
        similarities = np.dot(self.document_embeddings, query_embedding) / (
            np.linalg.norm(self.document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 获取排序后的文档索引
        sorted_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            results.append((self.documents[idx], float(similarities[idx]), idx))
        
        return results
    
    def hybrid_search(self, query, top_k=5, alpha=0.5):
        """执行混合检索
        
        alpha: 控制BM25和语义检索的权重，范围[0,1]
              alpha=0: 纯BM25
              alpha=1: 纯语义检索
        """
        # 获取两种检索结果
        bm25_results = self.bm25_search(query, top_k=top_k * 3)
        semantic_results = self.semantic_search(query, top_k=top_k * 3)
        
        # 构建文档索引到分数的映射
        doc_scores = {}
        
        # 处理BM25结果
        if bm25_results:
            max_bm25_score = max(score for _, score, _ in bm25_results)
            for doc, score, idx in bm25_results:
                normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0
                doc_scores[idx] = {
                    "doc": doc,
                    "bm25_score": normalized_score,
                    "semantic_score": 0,
                    "final_score": 0
                }
        
        # 处理语义结果
        if semantic_results:
            max_semantic_score = max(score for _, score, _ in semantic_results)
            for doc, score, idx in semantic_results:
                normalized_score = score / max_semantic_score if max_semantic_score > 0 else 0
                if idx not in doc_scores:
                    doc_scores[idx] = {
                        "doc": doc,
                        "bm25_score": 0,
                        "semantic_score": normalized_score,
                        "final_score": 0
                    }
                else:
                    doc_scores[idx]["semantic_score"] = normalized_score
        
        # 计算混合分数
        for idx, scores in doc_scores.items():
            scores["final_score"] = (
                scores["bm25_score"] * (1 - alpha) + 
                scores["semantic_score"] * alpha
            )
        
        # 按混合分数排序
        sorted_results = sorted(
            doc_scores.values(), 
            key=lambda x: x["final_score"], 
            reverse=True
        )
        
        return [(item["doc"], item["final_score"]) for item in sorted_results[:top_k]]
```

### 3. 重排序技术详解

重排序是提升检索质量的另一个重要技术，它通常在初步检索之后进行，用于对检索结果进行更精细的排序。

#### 3.1 交叉编码器重排序

交叉编码器（Cross-Encoder）是一种强大的重排序工具，它能够同时处理查询和文档，生成更准确的相关性评分。

```python
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, docs, top_k=3):
        """使用交叉编码器对检索结果进行重排序
        
        Args:
            query: 查询文本
            docs: 文档列表，格式为 [(doc1, score1), (doc2, score2), ...]
            top_k: 返回的文档数量
            
        Returns:
            重排序后的文档列表
        """
        # 准备查询-文档对
        if isinstance(docs[0], tuple):
            # 如果docs包含分数，提取文档内容
            document_texts = [doc for doc, _ in docs]
        else:
            document_texts = docs
        
        # 创建查询-文档对
        query_doc_pairs = [[query, doc] for doc in document_texts]
        
        # 使用交叉编码器计算相关性分数
        relevance_scores = self.model.predict(query_doc_pairs)
        
        # 创建包含原始文档和新分数的列表
        reranked_docs = []
        for i, score in enumerate(relevance_scores):
            if isinstance(docs[0], tuple):
                doc, _ = docs[i]
            else:
                doc = docs[i]
            reranked_docs.append((doc, float(score)))
        
        # 按新分数排序
        reranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_docs[:top_k]
```

#### 3.2 多级重排序策略

在实际应用中，我们常常采用多级重排序策略，结合不同的模型和方法，以达到最佳效果。

```python
class MultiStageReranker:
    def __init__(self, config=None):
        """多级重排序器
        
        Args:
            config: 配置字典，指定各阶段使用的重排序器
        """
        if config is None:
            config = {
                "stage1": {
                    "type": "cross_encoder",
                    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "top_k": 20  # 第一阶段保留的文档数
                },
                "stage2": {
                    "type": "cross_encoder",
                    "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # 更大的模型
                    "top_k": 5
                }
            }
        
        self.config = config
        self.rerankers = {}
        
        # 初始化各阶段的重排序器
        for stage, stage_config in config.items():
            if stage_config["type"] == "cross_encoder":
                from sentence_transformers import CrossEncoder
                self.rerankers[stage] = CrossEncoder(stage_config["model"])
            # 可以添加其他类型的重排序器
    
    def rerank(self, query, docs, top_k=5):
        """执行多级重排序
        
        Args:
            query: 查询文本
            docs: 初始检索结果，格式为 [(doc1, score1), (doc2, score2), ...]
            top_k: 最终返回的文档数量
            
        Returns:
            重排序后的文档列表
        """
        current_docs = docs
        
        # 执行每一级重排序
        for stage, stage_config in self.config.items():
            stage_top_k = stage_config.get("top_k", top_k)
            
            if stage_config["type"] == "cross_encoder":
                reranker = self.rerankers[stage]
                
                # 准备查询-文档对
                if isinstance(current_docs[0], tuple):
                    document_texts = [doc for doc, _ in current_docs]
                else:
                    document_texts = current_docs
                
                query_doc_pairs = [[query, doc] for doc in document_texts]
                
                # 计算相关性分数
                scores = reranker.predict(query_doc_pairs)
                
                # 创建新的文档列表
                new_docs = []
                for i, score in enumerate(scores):
                    if isinstance(current_docs[0], tuple):
                        doc, _ = current_docs[i]
                    else:
                        doc = current_docs[i]
                    new_docs.append((doc, float(score)))
                
                # 排序并保留指定数量的文档
                new_docs.sort(key=lambda x: x[1], reverse=True)
                current_docs = new_docs[:stage_top_k]
        
        # 返回最终结果
        return current_docs[:top_k]
```

### 4. 高级混合检索技术

除了基础的BM25和语义检索结合外，还有一些更高级的混合检索技术。

#### 4.1 ColBERT检索

ColBERT（Contextualized Late Interaction over BERT）是一种高效的语义检索方法，它能够捕捉查询和文档之间的细粒度交互。

```python
class ColBERTRetrieval:
    def __init__(self, index_path=None):
        """初始化ColBERT检索器
        
        Args:
            index_path: 预构建的索引路径，如果为None，则需要后续构建
        """
        from ragatouille import RAGPretrainedModel
        self.rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        self.index_path = index_path
        
        if index_path:
            self.rag.load_index(index_path)
    
    def index_documents(self, documents, index_path, max_document_length=512):
        """索引文档
        
        Args:
            documents: 文档列表
            index_path: 索引保存路径
            max_document_length: 文档最大长度（token数）
        """
        self.rag.index(
            collection=documents,
            index_name=index_path,
            max_document_length=max_document_length,
            split_documents=True
        )
        self.index_path = index_path
    
    def search(self, query, top_k=5):
        """执行ColBERT检索
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            检索结果列表，格式为 [(文档内容, 分数)]
        """
        if not self.index_path:
            raise ValueError("索引尚未构建或加载")
        
        results = self.rag.search(query=query, k=top_k)
        
        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_results.append((result["content"], result["score"]))
        
        return formatted_results
```

#### 4.2 基于知识图谱的检索增强

知识图谱可以为检索提供结构化的语义信息，增强检索的准确性。

```python
class GraphEnhancedRetrieval:
    def __init__(self, vector_store, graph_store):
        """初始化基于知识图谱的增强检索器
        
        Args:
            vector_store: 向量存储
            graph_store: 图数据库接口
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
    
    def extract_entities(self, query):
        """从查询中提取实体"""
        # 这里可以使用命名实体识别模型或规则方法
        # 简化示例：使用关键词匹配
        import re
        # 假设我们有一个实体列表
        entities = []
        # 实际应用中应该使用更复杂的实体提取方法
        return entities
    
    def expand_query_with_graph(self, query):
        """使用知识图谱扩展查询"""
        # 提取实体
        entities = self.extract_entities(query)
        
        # 从知识图谱获取相关概念
        related_concepts = []
        for entity in entities:
            # 查询实体的邻居节点
            neighbors = self.graph_store.get_neighbors(entity, top_k=3)
            related_concepts.extend(neighbors)
        
        # 去重
        related_concepts = list(set(related_concepts))
        
        # 构建扩展查询
        if related_concepts:
            expanded_query = query + " " + " ".join(related_concepts)
        else:
            expanded_query = query
        
        return expanded_query, related_concepts
    
    def search(self, query, top_k=5, use_graph_expansion=True):
        """执行基于知识图谱的增强检索
        
        Args:
            query: 原始查询
            top_k: 返回的文档数量
            use_graph_expansion: 是否使用图扩展
            
        Returns:
            检索结果
        """
        if use_graph_expansion:
            # 扩展查询
            expanded_query, related_concepts = self.expand_query_with_graph(query)
            # 使用扩展查询进行检索
            results = self.vector_store.similarity_search_with_score(expanded_query, k=top_k)
        else:
            # 直接使用原始查询
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        return results
```

### 5. 混合检索与重排序的集成应用

将混合检索和重排序技术集成到完整的RAG系统中，可以显著提升系统性能。

```python
class AdvancedRAGSystem:
    def __init__(self, config):
        """初始化高级RAG系统
        
        Args:
            config: 配置字典
        """
        # 初始化混合检索器
        vector_store = self._init_vector_store(config["vector_store"])
        keyword_store = self._init_keyword_store(config["keyword_store"])
        
        self.retriever = HybridRetrievalSystem(
            vector_store=vector_store,
            keyword_store=keyword_store,
            fusion_method=config.get("fusion_method", "rrf")
        )
        
        # 初始化重排序器
        if config.get("use_reranking", True):
            if config.get("reranking_type") == "multi_stage":
                self.reranker = MultiStageReranker(config.get("reranking_config"))
            else:
                self.reranker = CrossEncoderReranker(
                    model_name=config.get("reranking_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                )
        else:
            self.reranker = None
        
        # 初始化生成器
        self.generator = self._init_generator(config["generator"])
    
    def _init_vector_store(self, config):
        """初始化向量存储"""
        # 这里应该根据配置初始化具体的向量数据库
        # 例如Chroma, FAISS, Pinecone等
        pass
    
    def _init_keyword_store(self, config):
        """初始化关键词存储"""
        # 这里应该根据配置初始化关键词搜索引擎
        # 例如Elasticsearch, Whoosh等
        pass
    
    def _init_generator(self, config):
        """初始化文本生成器"""
        # 这里应该根据配置初始化LLM
        # 例如OpenAI API, HuggingFace模型等
        pass
    
    def answer_query(self, query, top_k=5, **kwargs):
        """回答用户查询
        
        Args:
            query: 用户查询
            top_k: 检索和重排序的文档数量
            **kwargs: 其他参数
            
        Returns:
            包含回答和源文档的字典
        """
        # 1. 执行混合检索
        retrieved_docs = self.retriever.search(query, top_k=top_k * 3, **kwargs)
        
        # 2. 如果启用了重排序，执行重排序
        if self.reranker:
            reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=top_k)
        else:
            reranked_docs = retrieved_docs[:top_k]
        
        # 3. 构建上下文
        context = "\n\n---\n\n".join([doc for doc, _ in reranked_docs])
        
        # 4. 生成回答
        prompt = self._build_prompt(query, context)
        answer = self.generator.generate(prompt)
        
        return {
            "answer": answer,
            "sources": reranked_docs
        }
    
    def _build_prompt(self, query, context):
        """构建提示词"""
        return f"""
        基于以下上下文信息回答用户问题：
        
        上下文：
        {context}
        
        问题：{query}
        
        回答：
        """
```

### 6. 评估与优化

评估混合检索和重排序系统的性能是确保其有效性的关键。

#### 6.1 评估指标

常用的评估指标包括：

- **准确率@k (Precision@k)**：前k个结果中相关文档的比例
- **召回率@k (Recall@k)**：前k个结果中覆盖的相关文档比例
- **平均精度均值 (MAP)**：所有查询的平均精度的平均值
- **归一化折损累计增益 (NDCG)**：考虑了相关度等级的排序质量指标

#### 6.2 自动调优策略

```python
class RetrievalOptimizer:
    def __init__(self, retrieval_system, eval_dataset):
        """初始化检索优化器
        
        Args:
            retrieval_system: 要优化的检索系统
            eval_dataset: 评估数据集，格式为 [(查询, 相关文档列表)]
        """
        self.retrieval_system = retrieval_system
        self.eval_dataset = eval_dataset
    
    def evaluate(self, params):
        """评估给定参数下的检索性能
        
        Args:
            params: 参数字典
            
        Returns:
            评估指标字典
        """
        # 根据参数配置检索系统
        self._configure_system(params)
        
        # 执行评估
        precision_at_1 = 0
        precision_at_3 = 0
        precision_at_5 = 0
        
        for query, relevant_docs in self.eval_dataset:
            # 执行检索
            results = self.retrieval_system.search(query, top_k=5)
            retrieved_docs = [doc for doc, _ in results]
            
            # 计算精度
            relevant_ids = set([doc.metadata.get("id", "") for doc in relevant_docs])
            retrieved_ids = set([doc.metadata.get("id", "") for doc in retrieved_docs])
            
            # Precision@1
            if len(retrieved_ids) >= 1:
                first_doc_id = list(retrieved_ids)[0]
                precision_at_1 += 1 if first_doc_id in relevant_ids else 0
            
            # Precision@3
            if len(retrieved_ids) >= 3:
                top3_ids = set(list(retrieved_ids)[:3])
                precision_at_3 += len(top3_ids.intersection(relevant_ids)) / 3
            
            # Precision@5
            top5_ids = set(list(retrieved_ids)[:5])
            precision_at_5 += len(top5_ids.intersection(relevant_ids)) / 5
        
        # 计算平均值
        num_queries = len(self.eval_dataset)
        return {
            "precision@1": precision_at_1 / num_queries,
            "precision@3": precision_at_3 / num_queries,
            "precision@5": precision_at_5 / num_queries
        }
    
    def _configure_system(self, params):
        """根据参数配置检索系统"""
        # 这里应该根据具体的检索系统实现配置逻辑
        # 例如设置融合权重、重排序参数等
        pass
    
    def grid_search(self, param_grid, metric="precision@3"):
        """网格搜索最优参数
        
        Args:
            param_grid: 参数网格，格式为 {"param1": [val1, val2, ...], ...}
            metric: 优化目标指标
            
        Returns:
            最优参数和对应的性能
        """
        import itertools
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        best_score = -1
        best_params = None
        
        # 评估每个参数组合
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            scores = self.evaluate(params)
            
            if scores[metric] > best_score:
                best_score = scores[metric]
                best_params = params
        
        return best_params, best_score
```

### 总结

本文详细介绍了RAG系统中的混合检索与重排序技术，这些技术是提升RAG系统性能的关键。通过结合多种检索方法的优势，并使用先进的重排序技术，我们可以显著提高检索结果的相关性和准确性。

在实际应用中，选择合适的混合检索策略和重排序方法需要考虑多种因素，包括数据特点、查询类型、计算资源限制等。建议通过实验和评估找到最适合特定应用场景的组合方案。

在下一篇文章中，我们将探讨RAG系统的部署和生产环境优化，包括如何构建可扩展、高可用的RAG服务，以及如何处理大规模知识库的管理和更新。