# 第四部分：RAG技术实战与优化
## 第14期 RAG高级优化：提升检索质量与生成效果的策略

## 4.2 RAG高级优化：提升检索质量与生成效果的策略

在前一篇文章中，我们介绍了RAG技术的基本原理和架构设计。在实际应用中，构建一个高性能的RAG系统需要解决许多挑战，包括检索质量、上下文管理、生成质量等方面的问题。本文将深入探讨RAG系统的高级优化策略，帮助你构建更加高效、准确和用户友好的RAG应用。

### 检索质量优化

#### 1. 语义检索增强

**密集检索增强技术**：

```python
class EnhancedDenseRetriever:
    def __init__(self, vector_db, query_expansion=True, cross_encoder_rerank=True):
        self.vector_db = vector_db
        self.query_expansion = query_expansion
        self.cross_encoder_rerank = cross_encoder_rerank
        
        if cross_encoder_rerank:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def expand_query(self, query, num_variations=3):
        """使用LLM进行查询扩展"""
        import openai
        
        prompt = f"""
        将以下查询扩展为{num_variations}个不同的表述，保持核心查询意图不变。
        每个变体应从不同角度或使用不同的关键词来表达相同的意思。
        
        查询: {query}
        
        变体列表:
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        expanded_queries = response.choices[0].message.content.strip().split('\n')
        # 清理和过滤结果
        expanded_queries = [q.strip() for q in expanded_queries if q.strip()]
        # 确保不包含原始查询本身
        expanded_queries = [q for q in expanded_queries if q.lower() != query.lower()[:len(q)]]
        
        return expanded_queries[:num_variations]
    
    def retrieve_with_expansion(self, query, top_k=5):
        """使用查询扩展进行检索"""
        # 获取扩展查询
        expanded_queries = self.expand_query(query)
        all_queries = [query] + expanded_queries
        
        # 对每个查询进行检索
        all_results = {}
        for q_idx, q in enumerate(all_queries):
            results = self.vector_db.query(q, n_results=top_k)
            
            # 合并结果，为来自不同查询的相同文档分配权重
            for doc_id, doc, metadata, distance in zip(
                results["ids"], results["documents"], 
                results["metadatas"], results["distances"]
            ):
                relevance_score = 1 - distance
                # 原始查询权重更高
                if q_idx == 0:
                    relevance_score *= 1.5
                
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "id": doc_id,
                        "content": doc,
                        "metadata": metadata,
                        "scores": [],
                        "final_score": 0
                    }
                all_results[doc_id]["scores"].append(relevance_score)
        
        # 计算最终分数（例如取平均值或最大值）
        for doc_id in all_results:
            # 可以使用不同的策略，这里使用最大值
            all_results[doc_id]["final_score"] = max(all_results[doc_id]["scores"])
        
        # 按最终分数排序
        sorted_results = sorted(
            all_results.values(), 
            key=lambda x: x["final_score"], 
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def rerank_with_cross_encoder(self, query, retrieved_docs, top_k=3):
        """使用交叉编码器重新排序检索结果"""
        # 准备用于交叉编码器的文本对
        pairs = [[query, doc["content"]] for doc in retrieved_docs]
        
        # 计算相关性分数
        scores = self.cross_encoder.predict(pairs)
        
        # 更新文档分数
        for i, doc in enumerate(retrieved_docs):
            doc["cross_score"] = float(scores[i])
        
        # 按交叉编码器分数排序
        reranked_docs = sorted(
            retrieved_docs, 
            key=lambda x: x["cross_score"], 
            reverse=True
        )
        
        return reranked_docs[:top_k]
    
    def retrieve(self, query, top_k=5):
        """综合检索方法"""
        if self.query_expansion:
            # 使用查询扩展检索
            retrieved_docs = self.retrieve_with_expansion(query, top_k=top_k*2)  # 多获取一些结果用于重排序
        else:
            # 基础检索
            results = self.vector_db.query(query, n_results=top_k*2)
            retrieved_docs = [
                {"id": doc_id, "content": doc, "metadata": metadata, "score": 1 - distance}
                for doc_id, doc, metadata, distance in zip(
                    results["ids"], results["documents"], 
                    results["metadatas"], results["distances"]
                )
            ]
        
        # 如果启用了交叉编码器重排序
        if self.cross_encoder_rerank and retrieved_docs:
            retrieved_docs = self.rerank_with_cross_encoder(query, retrieved_docs, top_k=top_k)
        else:
            retrieved_docs = retrieved_docs[:top_k]
        
        return retrieved_docs
```

#### 2. 混合检索策略深度优化

```python
class AdvancedHybridRetriever:
    def __init__(self, vector_db, keyword_engine, fusion_method="rrf"):
        self.vector_db = vector_db
        self.keyword_engine = keyword_engine  # 可以是Elasticsearch或其他关键词搜索引擎
        self.fusion_method = fusion_method  # rrf: Reciprocal Rank Fusion, linear: 线性加权
    
    def vector_search(self, query, top_k=10):
        """向量检索"""
        results = self.vector_db.query(query, n_results=top_k)
        vector_docs = []
        for doc_id, doc, metadata, distance in zip(
            results["ids"], results["documents"], 
            results["metadatas"], results["distances"]
        ):
            vector_docs.append({
                "id": doc_id,
                "content": doc,
                "metadata": metadata,
                "vector_score": 1 - distance,
                "vector_rank": len(vector_docs) + 1
            })
        return vector_docs
    
    def keyword_search(self, query, top_k=10):
        """关键词检索"""
        # 这里是示例，实际实现需要根据具体的关键词引擎调整
        # 假设keyword_engine提供了search方法
        results = self.keyword_engine.search(query, size=top_k)
        keyword_docs = []
        for i, result in enumerate(results):
            keyword_docs.append({
                "id": result["id"],
                "content": result["content"],
                "metadata": result.get("metadata", {}),
                "keyword_score": result["score"],
                "keyword_rank": i + 1
            })
        return keyword_docs
    
    def reciprocal_rank_fusion(self, vector_results, keyword_results, top_k=5):
        """使用倒数排序融合(RRF)合并结果"""
        # RRF公式: score = sum(1/(k + rank)) 其中k是常数，通常为60
        k = 60
        fused_scores = {}
        
        # 处理向量检索结果
        for doc in vector_results:
            doc_id = doc["id"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0}
            fused_scores[doc_id]["score"] += 1 / (k + doc["vector_rank"])
        
        # 处理关键词检索结果
        for doc in keyword_results:
            doc_id = doc["id"]
            if doc_id not in fused_scores:
                # 如果文档只在关键词结果中，需要合并内容和元数据
                fused_scores[doc_id] = {"doc": doc, "score": 0}
            else:
                # 如果已经存在，合并信息
                fused_scores[doc_id]["doc"]["keyword_score"] = doc.get("keyword_score", 0)
                fused_scores[doc_id]["doc"]["keyword_rank"] = doc.get("keyword_rank", 0)
            
            fused_scores[doc_id]["score"] += 1 / (k + doc["keyword_rank"])
        
        # 按融合分数排序
        sorted_results = sorted(
            fused_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        # 返回前top_k个结果
        return [item["doc"] for item in sorted_results[:top_k]]
    
    def linear_fusion(self, vector_results, keyword_results, weights=(0.6, 0.4), top_k=5):
        """线性加权融合结果"""
        vector_weight, keyword_weight = weights
        fused_scores = {}
        
        # 归一化向量分数
        max_vector_score = max([doc["vector_score"] for doc in vector_results], default=1)
        
        # 归一化关键词分数（如果有）
        max_keyword_score = 1
        if keyword_results:
            max_keyword_score = max([doc["keyword_score"] for doc in keyword_results], default=1)
        
        # 处理向量检索结果
        for doc in vector_results:
            doc_id = doc["id"]
            normalized_vector_score = doc["vector_score"] / max_vector_score
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    "doc": doc,
                    "vector_score": normalized_vector_score,
                    "keyword_score": 0,
                    "final_score": 0
                }
            else:
                fused_scores[doc_id]["vector_score"] = normalized_vector_score
        
        # 处理关键词检索结果
        for doc in keyword_results:
            doc_id = doc["id"]
            normalized_keyword_score = doc["keyword_score"] / max_keyword_score
            
            if doc_id not in fused_scores:
                # 如果文档只在关键词结果中，需要初始化
                fused_scores[doc_id] = {
                    "doc": doc,
                    "vector_score": 0,
                    "keyword_score": normalized_keyword_score,
                    "final_score": 0
                }
            else:
                # 如果已经存在，更新关键词分数
                fused_scores[doc_id]["keyword_score"] = normalized_keyword_score
                # 合并文档信息
                fused_scores[doc_id]["doc"]["keyword_score"] = doc["keyword_score"]
        
        # 计算最终分数
        for doc_id in fused_scores:
            fs = fused_scores[doc_id]
            fs["final_score"] = (
                fs["vector_score"] * vector_weight +
                fs["keyword_score"] * keyword_weight
            )
            # 将最终分数添加到文档中
            fs["doc"]["final_score"] = fs["final_score"]
        
        # 按最终分数排序
        sorted_results = sorted(
            fused_scores.values(), 
            key=lambda x: x["final_score"], 
            reverse=True
        )
        
        # 返回前top_k个结果
        return [item["doc"] for item in sorted_results[:top_k]]
    
    def retrieve(self, query, top_k=5, vector_weight=0.6, keyword_weight=0.4):
        """执行混合检索"""
        # 执行向量检索
        vector_results = self.vector_search(query, top_k=top_k*3)  # 获取更多结果用于融合
        
        # 执行关键词检索
        keyword_results = self.keyword_search(query, top_k=top_k*3)
        
        # 融合结果
        if self.fusion_method == "rrf":
            fused_results = self.reciprocal_rank_fusion(vector_results, keyword_results, top_k=top_k)
        else:  # linear
            fused_results = self.linear_fusion(vector_results, keyword_results, (vector_weight, keyword_weight), top_k=top_k)
        
        return fused_results
```

#### 3. 文档分块优化

```python
class SmartChunking:
    def __init__(self, min_chunk_size=100, max_chunk_size=1000, overlap=100):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def semantic_chunking(self, text, sentences_per_chunk=5):
        """基于语义的分块方法"""
        import spacy
        
        # 加载spaCy模型
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        # 将文本分割为句子
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # 按句子组进行分块
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # 如果添加当前句子会超过最大块大小
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                # 添加当前块
                chunks.append(' '.join(current_chunk))
                
                # 计算重叠部分（回溯句子）
                overlap_sentences = []
                overlap_length = 0
                
                # 从当前块末尾开始，添加句子直到达到重叠大小
                for i in range(len(current_chunk)-1, -1, -1):
                    overlap_sentences.insert(0, current_chunk[i])
                    overlap_length += len(current_chunk[i].split())
                    if overlap_length >= self.overlap:
                        break
                
                # 开始新块，包含重叠部分
                current_chunk = overlap_sentences + [sentence]
                current_length = overlap_length + sentence_length
            else:
                # 添加到当前块
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def hierarchical_chunking(self, text):
        """层次化分块方法"""
        import re
        
        # 首先按段落分割
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # 层次1：段落块
        level1_chunks = []
        level2_chunks = []
        
        # 处理每个段落
        for paragraph in paragraphs:
            # 检查段落长度
            para_length = len(paragraph.split())
            
            if para_length <= self.max_chunk_size:
                # 如果段落小于最大块大小，直接作为块
                level1_chunks.append(paragraph)
            else:
                # 如果段落过大，需要进一步分割
                # 按句子分割段落
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                # 分块句子
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sent_length = len(sentence.split())
                    
                    if current_length + sent_length > self.max_chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        level1_chunks.append(chunk_text)
                        level2_chunks.append(chunk_text)  # 子块也添加到level2
                        
                        # 添加重叠
                        overlap_sentences = []
                        overlap_length = 0
                        for i in range(len(current_chunk)-1, -1, -1):
                            overlap_sentences.insert(0, current_chunk[i])
                            overlap_length += len(current_chunk[i].split())
                            if overlap_length >= self.overlap:
                                break
                        
                        current_chunk = overlap_sentences + [sentence]
                        current_length = overlap_length + sent_length
                    else:
                        current_chunk.append(sentence)
                        current_length += sent_length
                
                # 处理最后一个块
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    level1_chunks.append(chunk_text)
                    level2_chunks.append(chunk_text)
        
        # 返回不同层次的块
        return {
            "level1": level1_chunks,  # 主要块
            "level2": level2_chunks   # 子块（用于精细检索）
        }
    
    def chunk_with_metadata(self, text, title=None, source=None, chunk_type="semantic"):
        """分块并添加元数据"""
        if chunk_type == "semantic":
            chunks = self.semantic_chunking(text)
        elif chunk_type == "hierarchical":
            # 对于层次化分块，我们使用level1作为主要块
            chunks = self.hierarchical_chunking(text)["level1"]
        else:
            # 默认使用基于句子的简单分块
            chunks = self.semantic_chunking(text)
        
        # 为每个块添加元数据
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "chunk_id": f"{source}_{i}" if source else f"chunk_{i}",
                "chunk_number": i + 1,
                "total_chunks": len(chunks),
                "chunk_type": chunk_type
            }
            
            if title:
                metadata["title"] = title
            if source:
                metadata["source"] = source
            
            # 添加前一个和后一个块的引用，用于上下文重建
            if i > 0:
                metadata["prev_chunk"] = f"{source}_{i-1}" if source else f"chunk_{i-1}"
            if i < len(chunks) - 1:
                metadata["next_chunk"] = f"{source}_{i+1}" if source else f"chunk_{i+1}"
            
            chunks_with_metadata.append({
                "text": chunk,
                "metadata": metadata
            })
        
        return chunks_with_metadata
```

### 上下文管理优化

#### 1. 上下文压缩技术

```python
class ContextCompressor:
    def __init__(self, max_context_length=2000, compression_method="llm_based"):
        self.max_context_length = max_context_length
        self.compression_method = compression_method
    
    def estimate_token_count(self, text):
        """估算文本的token数量"""
        # 这是一个简化的估算方法，实际应用中应该使用更准确的tokenizer
        # 例如使用tiktoken或其他模型特定的tokenizer
        return len(text.split()) * 1.3  # 粗略估计，英文文本通常一个词约1.3个token
    
    def compress_with_llm(self, text, query, max_length):
        """使用LLM压缩文本，保留与查询相关的信息"""
        import openai
        
        prompt = f"""
        请压缩以下文本，保留与查询"{query}"最相关的信息。
        压缩后的文本应尽量简洁，但必须保留所有与查询相关的关键信息和细节。
        不要添加任何原文中不存在的信息。
        目标长度应控制在约{max_length}个token以内。
        
        原始文本：
        {text}
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=int(max_length * 0.8)  # 控制输出长度
        )
        
        return response.choices[0].message.content.strip()
    
    def extractive_summarization(self, text, query, max_length):
        """使用抽取式摘要方法"""
        import re
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        
        # 按句子分割
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return ""
        
        # 计算句子与查询的相似度
        vectorizer = TfidfVectorizer().fit([query] + sentences)
        query_vec = vectorizer.transform([query]).toarray()[0]
        sentence_vecs = vectorizer.transform(sentences).toarray()
        
        # 计算余弦相似度
        similarity_scores = []
        for vec in sentence_vecs:
            if np.linalg.norm(vec) > 0 and np.linalg.norm(query_vec) > 0:
                sim = np.dot(vec, query_vec) / (np.linalg.norm(vec) * np.linalg.norm(query_vec))
            else:
                sim = 0
            similarity_scores.append(sim)
        
        # 按相似度排序句子
        sorted_sentences = sorted(
            zip(sentences, similarity_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 贪心选择句子，直到达到最大长度
        selected_sentences = []
        current_length = 0
        
        for sentence, score in sorted_sentences:
            sentence_length = self.estimate_token_count(sentence)
            
            if current_length + sentence_length <= max_length:
                selected_sentences.append(sentence)
                current_length += sentence_length
            else:
                # 如果句子很重要但太长，可以考虑截断
                if score > 0.5:  # 只有相似度高的句子才截断
                    remaining_length = max_length - current_length
                    # 粗略截断句子
                    words = sentence.split()
                    estimated_words_needed = int(remaining_length / 1.3)  # 转回单词数
                    if estimated_words_needed > 10:  # 确保截断后的句子有意义
                        truncated = ' '.join(words[:estimated_words_needed]) + "..."
                        selected_sentences.append(truncated)
                        current_length += remaining_length
                break
        
        # 恢复句子的原始顺序
        selected_indices = [sentences.index(s) for s in [s for s, _ in selected_sentences]]
        ordered_sentences = [s for i, s in sorted(zip(selected_indices, [s for s, _ in selected_sentences]))]
        
        return ' '.join(ordered_sentences)
    
    def compress_context(self, retrieved_docs, query, max_context_length=None):
        """压缩检索到的上下文，确保在最大长度限制内"""
        if max_context_length is None:
            max_context_length = self.max_context_length
        
        # 估算每个文档的长度
        docs_with_length = []
        total_length = 0
        
        for doc in retrieved_docs:
            length = self.estimate_token_count(doc["content"])
            docs_with_length.append({
                "doc": doc,
                "length": length
            })
            total_length += length
        
        # 如果总长度已经在限制内，直接返回
        if total_length <= max_context_length:
            return [d["doc"] for d in docs_with_length]
        
        # 否则需要压缩
        # 1. 先按相关性排序（假设有score字段）
        sorted_docs = sorted(docs_with_length, key=lambda x: x["doc"].get("final_score", 0), reverse=True)
        
        # 2. 计算每个文档可分配的最大长度
        # 基于相关性分配权重
        scores = [d["doc"].get("final_score", 1) for d in sorted_docs]
        total_score = sum(scores)
        
        if total_score > 0:
            # 按相关性比例分配长度
            allocated_lengths = [int((score / total_score) * max_context_length) for score in scores]
        else:
            # 如果没有分数，平均分配
            allocated_lengths = [max_context_length // len(sorted_docs) for _ in sorted_docs]
        
        # 3. 压缩每个文档
        compressed_docs = []
        actual_total = 0
        
        for i, (doc_with_length, allocated) in enumerate(zip(sorted_docs, allocated_lengths)):
            doc = doc_with_length["doc"]
            current_length = doc_with_length["length"]
            
            if current_length <= allocated:
                # 如果文档已经小于分配长度，不需要压缩
                compressed_docs.append(doc)
                actual_total += current_length
            else:
                # 压缩文档
                if self.compression_method == "llm_based":
                    compressed_content = self.compress_with_llm(doc["content"], query, allocated)
                else:  # extractive
                    compressed_content = self.extractive_summarization(doc["content"], query, allocated)
                
                # 估算压缩后的长度
                compressed_length = self.estimate_token_count(compressed_content)
                
                # 创建压缩后的文档
                compressed_doc = doc.copy()
                compressed_doc["content"] = compressed_content
                compressed_doc["compressed"] = True
                compressed_doc["original_length"] = current_length
                compressed_doc["compressed_length"] = compressed_length
                
                compressed_docs.append(compressed_doc)
                actual_total += compressed_length
        
        # 4. 如果仍然超过限制，进一步压缩最重要的文档
        while actual_total > max_context_length and compressed_docs:
            # 找出最长的文档
            longest_idx = max(range(len(compressed_docs)), key=lambda i: self.estimate_token_count(compressed_docs[i]["content"]))
            longest_doc = compressed_docs[longest_idx]
            
            # 计算需要减少的长度
            excess = actual_total - max_context_length
            current_length = self.estimate_token_count(longest_doc["content"])
            target_length = max(50, current_length - excess)  # 确保不小于50个token
            
            # 进一步压缩
            if self.compression_method == "llm_based":
                more_compressed = self.compress_with_llm(longest_doc["content"], query, target_length)
            else:
                more_compressed = self.extractive_summarization(longest_doc["content"], query, target_length)
            
            # 更新文档
            old_length = current_length
            compressed_docs[longest_idx]["content"] = more_compressed
            new_length = self.estimate_token_count(more_compressed)
            actual_total = actual_total - old_length + new_length
            
            # 如果已经无法进一步压缩，移除最不重要的文档
            if new_length <= target_length and actual_total > max_context_length:
                # 移除相关性最低的文档
                min_score_idx = min(range(len(compressed_docs)), key=lambda i: compressed_docs[i].get("final_score", 0))
                removed_length = self.estimate_token_count(compressed_docs[min_score_idx]["content"])
                actual_total -= removed_length
                compressed_docs.pop(min_score_idx)
        
        return compressed_docs
```

#### 2. 上下文排序与重组织

```python
class ContextOrganizer:
    def __init__(self, max_context_length=2000):
        self.max_context_length = max_context_length
    
    def estimate_token_count(self, text):
        """估算文本的token数量"""
        return len(text.split()) * 1.3  # 粗略估计
    
    def calculate_semantic_overlap(self, text1, text2):
        """计算两个文本之间的语义重叠度"""
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # 加载模型
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 生成嵌入
        embeddings = model.encode([text1, text2])
        
        # 计算余弦相似度
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return similarity
    
    def organize_by_relevance(self, retrieved_docs, query):
        """按与查询的相关性排序文档"""
        # 假设文档已经有相关性分数
        # 如果没有，可以使用向量相似度或其他方法计算
        sorted_docs = sorted(
            retrieved_docs, 
            key=lambda x: x.get("final_score", x.get("score", 0)), 
            reverse=True
        )
        return sorted_docs
    
    def organize_by_recency(self, retrieved_docs):
        """按时间排序文档"""
        # 假设元数据中有时间信息
        def get_date(doc):
            metadata = doc.get("metadata", {})
            # 尝试不同的日期字段名
            for date_field in ["date", "created_at", "updated_at", "publication_date"]:
                if date_field in metadata:
                    return metadata[date_field]
            # 如果没有日期信息，返回一个默认的旧日期
            return "1900-01-01"
        
        sorted_docs = sorted(
            retrieved_docs, 
            key=get_date, 
            reverse=True  # 最新的在前
        )
        return sorted_docs
    
    def organize_by_topic(self, retrieved_docs):
        """按主题组织文档"""
        # 这是一个简化实现，实际应用中可能需要更复杂的主题建模
        # 这里我们假设元数据中已经有主题信息
        topics = {}
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            topic = metadata.get("topic", "general")
            
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(doc)
        
        # 按每个主题中的文档数量排序主题
        sorted_topics = sorted(
            topics.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        # 合并排序后的文档
        organized_docs = []
        for topic, docs in sorted_topics:
            # 对每个主题内的文档按相关性排序
            sorted_docs = sorted(
                docs, 
                key=lambda x: x.get("final_score", x.get("score", 0)), 
                reverse=True
            )
            organized_docs.extend(sorted_docs)
        
        return organized_docs
    
    def organize_with_flow(self, retrieved_docs, query):
        """按逻辑流组织文档，确保上下文连贯性"""
        # 1. 首先选择最相关的文档作为起点
        sorted_docs = sorted(
            retrieved_docs, 
            key=lambda x: x.get("final_score", x.get("score", 0)), 
            reverse=True
        )
        
        if not sorted_docs:
            return []
        
        # 2. 构建文档图，基于语义重叠
        doc_graph = {}
        for i, doc1 in enumerate(sorted_docs):
            doc_graph[i] = []
            for j, doc2 in enumerate(sorted_docs):
                if i != j:
                    overlap = self.calculate_semantic_overlap(doc1["content"], doc2["content"])
                    if overlap > 0.5:  # 只考虑重叠度高的文档
                        doc_graph[i].append((j, overlap))
        
        # 3. 使用贪心算法构建连贯性强的文档序列
        used = set()
        organized_indices = []
        
        # 从最相关的文档开始
        current_idx = 0
        used.add(current_idx)
        organized_indices.append(current_idx)
        
        # 迭代添加最连贯的下一个文档
        while len(used) < len(sorted_docs):
            best_next = None
            best_overlap = 0
            
            # 查找与当前文档最连贯的未使用文档
            for idx, overlap in doc_graph[current_idx]:
                if idx not in used and overlap > best_overlap:
                    best_next = idx
                    best_overlap = overlap
            
            # 如果找不到连贯的文档，选择下一个最相关的
            if best_next is None:
                for i in range(len(sorted_docs)):
                    if i not in used:
                        best_next = i
                        break
            
            # 添加到序列
            if best_next is not None:
                used.add(best_next)
                organized_indices.append(best_next)
                current_idx = best_next
            else:
                break
        
        # 4. 重新排序文档
        organized_docs = [sorted_docs[i] for i in organized_indices]
        return organized_docs
    
    def format_context(self, organized_docs, query, include_metadata=True):
        """将组织好的文档格式化为上下文文本"""
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(organized_docs):
            # 格式化文档内容
            content = doc["content"]
            
            # 添加元数据信息
            if include_metadata:
                metadata = doc.get("metadata", {})
                metadata_parts = []
                
                if "title" in metadata:
                    metadata_parts.append(f"标题: {metadata['title']}")
                if "source" in metadata:
                    metadata_parts.append(f"来源: {metadata['source']}")
                if "date" in metadata:
                    metadata_parts.append(f"日期: {metadata['date']}")
                
                if metadata_parts:
                    content = f"[文档 {i+1}: {', '.join(metadata_parts)}]\n{content}"
                else:
                    content = f"[文档 {i+1}]\n{content}"
            
            # 估算长度
            content_length = self.estimate_token_count(content)
            
            # 检查是否超过最大长度
            if total_length + content_length > self.max_context_length:
                # 可以选择截断或跳过
                # 这里我们选择跳过剩余文档
                break
            
            context_parts.append(content)
            total_length += content_length
        
        # 合并为单个上下文文本
        context = "\n\n---\n\n".join(context_parts)
        
        return context
```

### 生成质量优化

#### 1. 提示词工程优化

```python
class AdvancedPromptBuilder:
    def __init__(self):
        # 不同任务类型的提示词模板
        self.templates = {
            "qa": """
            你是一个专业的问答助手，基于提供的上下文信息回答用户问题。
            
            要求：
            1. 严格基于提供的上下文信息回答，不要添加外部知识
            2. 如果上下文没有相关信息，请明确表示不知道，不要猜测
            3. 回答要准确、简洁、结构清晰
            4. 对于每个结论，请引用相关的文档来源
            5. 如果有多个相关文档，请综合所有信息提供全面的回答
            
            上下文信息：
            {context}
            
            用户问题：{query}
            
            回答：
            """,
            
            "summarization": """
            请基于以下上下文信息，为用户的查询生成一个全面、准确的摘要。
            
            要求：
            1. 摘要必须完全基于提供的上下文信息
            2. 重点突出与用户查询直接相关的内容
            3. 保持逻辑连贯，结构清晰
            4. 避免冗余，简明扼要地传达关键信息
            5. 对于重要的数据和结论，请保留精确的表述
            
            上下文信息：
            {context}
            
            用户查询：{query}
            
            摘要：
            """,
            
            "creative": """
            基于提供的上下文信息，为用户的请求生成创意内容。
            
            你可以：
            - 扩展上下文信息中的概念
            - 基于事实创建引人入胜的叙述
            - 将技术内容转化为更易于理解的形式
            - 提出基于上下文的创新想法
            
            但请注意：
            1. 所有创意内容必须以事实为基础
            2. 明确区分事实信息和创意扩展
            3. 确保内容准确反映原始上下文
            
            上下文信息：
            {context}
            
            用户请求：{query}
            
            创意回答：
            """
        }
    
    def detect_task_type(self, query):
        """检测查询的任务类型"""
        # 这是一个简化的实现，实际应用中可以使用更复杂的方法
        query_lower = query.lower()
        
        # 问答类型
        qa_keywords = ["what is", "who is", "when", "where", "why", "how", "explain", "define", "list", "examples"]
        if any(keyword in query_lower for keyword in qa_keywords):
            return "qa"
        
        # 摘要类型
        summary_keywords = ["summary", "summarize", "overview", "main points", "key takeaways"]
        if any(keyword in query_lower for keyword in summary_keywords):
            return "summarization"
        
        # 默认返回问答类型
        return "qa"
    
    def build_prompt_with_history(self, query, context, history=None, task_type=None):
        """构建包含对话历史的提示词"""
        if task_type is None:
            task_type = self.detect_task_type(query)
        
        # 获取适合任务类型的模板
        template = self.templates.get(task_type, self.templates["qa"])
        
        # 基础提示词
        base_prompt = template.format(context=context, query=query)
        
        # 如果有对话历史，添加到提示词中
        if history and len(history) > 0:
            history_text = "对话历史：\n"
            
            for turn in history:
                role = "用户" if turn["role"] == "user" else "助手"
                history_text += f"{role}：{turn['content']}\n"
            
            # 在原始模板基础上添加历史
            enhanced_template = """
            {history}
            
            请基于对话历史和以下上下文信息，继续回答用户的最新问题。
            
            上下文信息：
            {context}
            
            用户最新问题：{query}
            
            回答：
            """
            
            base_prompt = enhanced_template.format(
                history=history_text.strip(),
                context=context,
                query=query
            )
        
        return base_prompt
    
    def build_multi_step_prompt(self, query, context):
        """构建多步骤思考的提示词"""
        multi_step_template = """
        请通过以下步骤回答用户问题：
        
        步骤1: 仔细分析用户的问题，确定需要回答的核心内容
        步骤2: 从提供的上下文中识别与问题相关的所有信息
        步骤3: 评估信息的可靠性和相关性
        步骤4: 基于收集的信息，构建全面、准确的回答
        步骤5: 检查回答是否完全基于上下文，没有引入外部信息
        步骤6: 确保回答逻辑清晰，结构良好
        
        上下文信息：
        {context}
        
        用户问题：{query}
        
        请按照上述步骤，提供你的思考过程和最终回答。
        """
        
        return multi_step_template.format(context=context, query=query)
    
    def build_citation_prompt(self, query, context):
        """构建带引用要求的提示词"""
        citation_template = """
        请基于提供的上下文信息回答用户问题，并在回答中为每个事实性陈述提供引用。
        
        具体要求：
        1. 严格基于上下文回答问题
        2. 在每个事实性陈述后立即添加引用标记，格式为[文档X]
        3. 如果来自多个文档，请分别引用
        4. 回答结束后，列出每个引用对应的完整文档信息
        5. 如果信息冲突，请说明并提供所有相关引用
        
        上下文信息：
        {context}
        
        用户问题：{query}
        
        回答：
        """
        
        return citation_template.format(context=context, query=query)
```

#### 2. 回答生成优化

```python
class AdvancedAnswerGenerator:
    def __init__(self, api_key, default_model="gpt-3.5-turbo"):
        import openai
        openai.api_key = api_key
        self.default_model = default_model
    
    def generate_with_retry(self, prompt, model=None, max_retries=3, **kwargs):
        """带重试机制的回答生成"""
        import openai
        import time
        
        if model is None:
            model = self.default_model
        
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                return response.choices[0].message.content
            except openai.error.RateLimitError:
                # 速率限制错误，等待后重试
                wait_time = (2 ** attempt) * 0.5  # 指数退避
                print(f"速率限制，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            except openai.error.APIError as e:
                # API错误，打印错误信息并重试
                print(f"API错误 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
            except Exception as e:
                # 其他错误，打印错误信息并抛出
                print(f"生成回答时出错: {e}")
                raise
    
    def generate_with_temperature_scaling(self, prompt, query, temperature_range=(0.1, 0.7, 0.3)):
        """基于查询复杂度调整生成温度"""
        # 简单的复杂度检测（实际应用中可以使用更复杂的方法）
        query_length = len(query.split())
        
        # 根据查询长度简单分类
        if query_length < 10:
            # 简单查询，使用较低温度
            temperature = temperature_range[0]
        elif query_length > 30:
            # 复杂查询，可能需要更多创造性，使用中等温度
            temperature = temperature_range[1]
        else:
            # 中等复杂度查询
            temperature = temperature_range[2]
        
        # 生成回答
        answer = self.generate_with_retry(prompt, temperature=temperature)
        return answer, temperature
    
    def generate_with_self_critique(self, prompt, model=None):
        """使用自我批评机制改进回答质量"""
        # 生成初步回答
        initial_answer = self.generate_with_retry(prompt, model=model)
        
        # 创建自我批评提示词
        critique_prompt = f"""
        请分析以下回答是否存在以下问题：
        1. 是否完全基于提供的上下文信息，没有添加外部知识？
        2. 是否准确回答了用户的所有问题？
        3. 是否存在逻辑不一致或矛盾的地方？
        4. 是否有足够的细节和解释？
        5. 是否有需要改进的地方？
        
        初步回答：
        {initial_answer}
        
        请提供具体的批评意见和改进建议。
        """
        
        # 获取批评意见
        critique = self.generate_with_retry(critique_prompt, model=model)
        
        # 创建改进提示词
        improve_prompt = f"""
        请基于以下批评意见，改进初步回答：
        
        批评意见：
        {critique}
        
        原始提示词：
        {prompt}
        
        初步回答：
        {initial_answer}
        
        改进后的回答：
        """
        
        # 生成改进后的回答
        improved_answer = self.generate_with_retry(improve_prompt, model=model)
        
        return improved_answer
    
    def generate_structured_answer(self, prompt, output_format="markdown"):
        """生成结构化的回答"""
        # 添加格式要求到提示词
        format_instructions = ""
        
        if output_format == "markdown":
            format_instructions = "请使用Markdown格式组织你的回答，包括适当的标题、列表和强调。"
        elif output_format == "html":
            format_instructions = "请使用HTML格式组织你的回答，包括适当的标题、列表和样式。"
        elif output_format == "json":
            format_instructions = "请使用JSON格式输出，包括'answer'（主要回答）和'key_points'（关键点列表）两个字段。"
        
        if format_instructions:
            enhanced_prompt = f"{prompt}\n\n{format_instructions}"
        else:
            enhanced_prompt = prompt
        
        # 生成结构化回答
        structured_answer = self.generate_with_retry(enhanced_prompt)
        
        return structured_answer
```

### 端到端RAG优化系统

```python
class OptimizedRAGSystem:
    def __init__(self, config):
        # 初始化各个组件
        self.config = config
        
        # 1. 初始化向量数据库
        from rag_components import VectorDatabase
        self.vector_db = VectorDatabase(
            collection_name=config.get("collection_name", "documents"),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        
        # 2. 初始化高级检索器
        if config.get("use_hybrid_retrieval", True):
            from rag_components import AdvancedHybridRetriever
            # 这里需要初始化关键词引擎，例如Elasticsearch
            # 为简化，这里假设有一个简单的关键词引擎
            keyword_engine = SimpleKeywordEngine()
            self.retriever = AdvancedHybridRetriever(
                vector_db=self.vector_db,
                keyword_engine=keyword_engine,
                fusion_method=config.get("fusion_method", "rrf")
            )
        else:
            from rag_components import EnhancedDenseRetriever
            self.retriever = EnhancedDenseRetriever(
                vector_db=self.vector_db,
                query_expansion=config.get("query_expansion", True),
                cross_encoder_rerank=config.get("cross_encoder_rerank", True)
            )
        
        # 3. 初始化上下文压缩器
        from rag_components import ContextCompressor
        self.compressor = ContextCompressor(
            max_context_length=config.get("max_context_length", 2000),
            compression_method=config.get("compression_method", "llm_based")
        )
        
        # 4. 初始化上下文组织器
        from rag_components import ContextOrganizer
        self.organizer = ContextOrganizer(
            max_context_length=config.get("max_context_length", 2000)
        )
        
        # 5. 初始化高级提示词构建器
        from rag_components import AdvancedPromptBuilder
        self.prompt_builder = AdvancedPromptBuilder()
        
        # 6. 初始化高级回答生成器
        from rag_components import AdvancedAnswerGenerator
        self.answer_generator = AdvancedAnswerGenerator(
            api_key=config["openai_api_key"],
            default_model=config.get("default_model", "gpt-3.5-turbo")
        )
    
    def answer_query(self, query, history=None, task_type=None, return_diagnostics=False):
        """优化的查询回答流程"""
        diagnostics = {
            "timing": {},
            "retrieval_stats": {},
            "context_stats": {},
            "generation_stats": {}
        }
        import time
        
        # 1. 检索相关文档
        start_time = time.time()
        retrieved_docs = self.retriever.retrieve(query, top_k=self.config.get("top_k", 5))
        diagnostics["timing"]["retrieval"] = time.time() - start_time
        diagnostics["retrieval_stats"]["docs_retrieved"] = len(retrieved_docs)
        
        # 2. 压缩上下文
        start_time = time.time()
        compressed_docs = self.compressor.compress_context(
            retrieved_docs, 
            query, 
            max_context_length=self.config.get("max_context_length", 2000)
        )
        diagnostics["timing"]["compression"] = time.time() - start_time
        diagnostics["context_stats"]["docs_after_compression"] = len(compressed_docs)
        
        # 3. 组织上下文
        start_time = time.time()
        # 根据配置选择组织方法
        organization_method = self.config.get("organization_method", "relevance")
        
        if organization_method == "relevance":
            organized_docs = self.organizer.organize_by_relevance(compressed_docs, query)
        elif organization_method == "recency":
            organized_docs = self.organizer.organize_by_recency(compressed_docs)
        elif organization_method == "topic":
            organized_docs = self.organizer.organize_by_topic(compressed_docs)
        elif organization_method == "flow":
            organized_docs = self.organizer.organize_with_flow(compressed_docs, query)
        else:
            organized_docs = compressed_docs
        
        # 格式化为上下文文本
        context = self.organizer.format_context(
            organized_docs, 
            query, 
            include_metadata=self.config.get("include_metadata", True)
        )
        diagnostics["timing"]["organization"] = time.time() - start_time
        diagnostics["context_stats"]["context_length"] = len(context.split())
        
        # 4. 构建提示词
        start_time = time.time()
        prompt_type = self.config.get("prompt_type", "standard")
        
        if prompt_type == "with_history" and history:
            prompt = self.prompt_builder.build_prompt_with_history(
                query, context, history, task_type
            )
        elif prompt_type == "multi_step":
            prompt = self.prompt_builder.build_multi_step_prompt(query, context)
        elif prompt_type == "citation":
            prompt = self.prompt_builder.build_citation_prompt(query, context)
        else:
            # 检测任务类型
            if task_type is None:
                task_type = self.prompt_builder.detect_task_type(query)
            
            # 使用标准提示词模板
            prompt = self.prompt_builder.build_prompt_with_history(
                query, context, history, task_type
            )
        
        diagnostics["timing"]["prompt_building"] = time.time() - start_time
        
        # 5. 生成回答
        start_time = time.time()
        generation_strategy = self.config.get("generation_strategy", "standard")
        
        if generation_strategy == "temperature_scaling":
            answer, temperature = self.answer_generator.generate_with_temperature_scaling(prompt, query)
            diagnostics["generation_stats"]["temperature"] = temperature
        elif generation_strategy == "self_critique":
            answer = self.answer_generator.generate_with_self_critique(prompt)
            diagnostics["generation_stats"]["strategy"] = "self_critique"
        elif generation_strategy == "structured":
            output_format = self.config.get("output_format", "markdown")
            answer = self.answer_generator.generate_structured_answer(prompt, output_format)
            diagnostics["generation_stats"]["output_format"] = output_format
        else:
            # 标准生成
            temperature = self.config.get("temperature", 0.1)
            answer = self.answer_generator.generate_with_retry(prompt, temperature=temperature)
            diagnostics["generation_stats"]["temperature"] = temperature
        
        diagnostics["timing"]["generation"] = time.time() - start_time
        
        # 6. 计算总时间
        diagnostics["timing"]["total"] = sum(diagnostics["timing"].values())
        
        if return_diagnostics:
            return {
                "answer": answer,
                "sources": organized_docs,
                "diagnostics": diagnostics
            }
        else:
            return {
                "answer": answer,
                "sources": organized_docs
            }
    
    def add_documents(self, documents, metadatas=None):
        """添加文档到知识库"""
        # 预处理文档
        processed_docs = []
        processed_metadatas = []
        
        for i, doc in enumerate(documents):
            # 智能分块
            if self.config.get("use_smart_chunking", True):
                from rag_components import SmartChunking
                chunker = SmartChunking(
                    min_chunk_size=self.config.get("min_chunk_size", 100),
                    max_chunk_size=self.config.get("max_chunk_size", 1000),
                    overlap=self.config.get("chunk_overlap", 100)
                )
                
                chunk_type = self.config.get("chunk_type", "semantic")
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                
                chunks_with_meta = chunker.chunk_with_metadata(
                    doc, 
                    title=metadata.get("title"),
                    source=metadata.get("source"),
                    chunk_type=chunk_type
                )
                
                # 添加分块后的文档和元数据
                for chunk_info in chunks_with_meta:
                    processed_docs.append(chunk_info["text"])
                    # 合并原始元数据和分块元数据
                    merged_metadata = metadata.copy()
                    merged_metadata.update(chunk_info["metadata"])
                    processed_metadatas.append(merged_metadata)
            else:
                # 不使用分块
                processed_docs.append(doc)
                if metadatas and i < len(metadatas):
                    processed_metadatas.append(metadatas[i])
        
        # 添加到向量数据库
        self.vector_db.add_documents(processed_docs, processed_metadatas)
        
        return {
            "docs_added": len(processed_docs),
            "total_docs": self.vector_db.get_stats()
        }
```

### 性能监控与持续优化

#### 1. RAG系统监控

```python
class RAGMonitor:
    def __init__(self):
        # 初始化监控指标
        self.metrics = {
            "queries": 0,
            "retrieval": {
                "avg_time": 0,
                "total_time": 0,
                "docs_per_query": 0
            },
            "generation": {
                "avg_time": 0,
                "total_time": 0,
                "avg_tokens": 0
            },
            "total": {
                "avg_time": 0,
                "total_time": 0
            }
        }
        
        # 用户反馈存储
        self.user_feedback = []
    
    def log_query(self, diagnostics):
        """记录查询的诊断信息"""
        # 更新查询计数
        self.metrics["queries"] += 1
        
        # 更新时间指标
        timing = diagnostics.get("timing", {})
        self.metrics["retrieval"]["total_time"] += timing.get("retrieval", 0)
        self.metrics["generation"]["total_time"] += timing.get("generation", 0)
        self.metrics["total"]["total_time"] += timing.get("total", 0)
        
        # 更新文档计数
        retrieval_stats = diagnostics.get("retrieval_stats", {})
        self.metrics["retrieval"]["docs_per_query"] += retrieval_stats.get("docs_retrieved", 0)
        
        # 重新计算平均值
        q = self.metrics["queries"]
        self.metrics["retrieval"]["avg_time"] = self.metrics["retrieval"]["total_time"] / q
        self.metrics["generation"]["avg_time"] = self.metrics["generation"]["total_time"] / q
        self.metrics["total"]["avg_time"] = self.metrics["total"]["total_time"] / q
        self.metrics["retrieval"]["docs_per_query"] = self.metrics["retrieval"]["docs_per_query"] / q
    
    def log_feedback(self, query, answer, rating, comment=None):
        """记录用户反馈"""
        from datetime import datetime
        feedback = {
            "query": query,
            "answer": answer,
            "rating": rating,  # 1-5星
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        self.user_feedback.append(feedback)
    
    def get_performance_report(self):
        """生成性能报告"""
        # 计算用户满意度
        if self.user_feedback:
            avg_rating = sum(f["rating"] for f in self.user_feedback) / len(self.user_feedback)
            positive_feedback = sum(1 for f in self.user_feedback if f["rating"] >= 4)
            satisfaction_rate = positive_feedback / len(self.user_feedback) * 100
        else:
            avg_rating = 0
            satisfaction_rate = 0
        
        report = {
            "query_count": self.metrics["queries"],
            "response_time": {
                "avg_total": self.metrics["total"]["avg_time"],
                "avg_retrieval": self.metrics["retrieval"]["avg_time"],
                "avg_generation": self.metrics["generation"]["avg_time"]
            },
            "retrieval_stats": {
                "avg_docs_per_query": self.metrics["retrieval"]["docs_per_query"]
            },
            "user_satisfaction": {
                "avg_rating": avg_rating,
                "satisfaction_rate": satisfaction_rate,
                "feedback_count": len(self.user_feedback)
            }
        }
        
        return report
    
    def detect_problems(self):
        """检测潜在问题"""
        problems = []
        report = self.get_performance_report()
        
        # 检测响应时间问题
        if report["response_time"]["avg_total"] > 5:  # 超过5秒
            problems.append({
                "type": "slow_response",
                "severity": "high",
                "description": f"平均响应时间过长: {report['response_time']['avg_total']:.2f}秒",
                "recommendation": "检查检索和生成阶段的性能，考虑优化向量数据库查询或更换更快的模型"
            })
        
        # 检测检索问题
        if report["retrieval_stats"]["avg_docs_per_query"] < 1:  # 平均检索不到1个文档
            problems.append({
                "type": "poor_retrieval",
                "severity": "medium",
                "description": "检索结果数量过少，可能是知识库太小或查询扩展不足",
                "recommendation": "增加知识库文档数量，启用查询扩展或调整检索参数"
            })
        
        # 检测用户满意度问题
        if report["user_satisfaction"]["satisfaction_rate"] < 70:  # 满意度低于70%
            problems.append({
                "type": "low_satisfaction",
                "severity": "high",
                "description": f"用户满意度低: {report['user_satisfaction']['satisfaction_rate']:.1f}%",
                "recommendation": "分析用户反馈，优化提示词或改进检索策略"
            })
        
        return problems
```

#### 2. A/B测试框架

```python
class RAGABTester:
    def __init__(self, base_config, test_configs):
        self.base_config = base_config
        self.test_configs = test_configs  # 测试配置列表
        self.experiments = {}
        
        # 初始化每个配置的监控器
        self.monitors = {
            "base": RAGMonitor()
        }
        
        for i, config in enumerate(test_configs):
            config_name = config.get("name", f"test_{i}")
            self.monitors[config_name] = RAGMonitor()
    
    def assign_experiment(self, user_id=None):
        """为用户分配实验配置"""
        import hashlib
        import random
        
        # 如果提供了用户ID，使用一致性哈希确保用户始终分配到相同的实验
        if user_id:
            # 使用用户ID的哈希值来确定实验
            hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 100
            
            # 假设20%流量到基础配置，80%均匀分配到测试配置
            if hash_val < 20:
                return "base"
            else:
                test_idx = (hash_val - 20) // (80 // len(self.test_configs))
                test_idx = min(test_idx, len(self.test_configs) - 1)
                return self.test_configs[test_idx].get("name", f"test_{test_idx}")
        else:
            # 没有用户ID时随机分配
            all_experiments = ["base"] + [config.get("name", f"test_{i}") for i, config in enumerate(self.test_configs)]
            return random.choice(all_experiments)
    
    def get_config(self, experiment_name):
        """获取指定实验的配置"""
        if experiment_name == "base":
            return self.base_config
        
        for config in self.test_configs:
            if config.get("name") == experiment_name:
                # 合并基础配置和测试配置
                merged_config = self.base_config.copy()
                merged_config.update(config)
                return merged_config
        
        return self.base_config  # 默认返回基础配置
    
    def log_experiment_result(self, experiment_name, diagnostics):
        """记录实验结果"""
        if experiment_name in self.monitors:
            self.monitors[experiment_name].log_query(diagnostics)
    
    def log_experiment_feedback(self, experiment_name, query, answer, rating, comment=None):
        """记录实验的用户反馈"""
        if experiment_name in self.monitors:
            self.monitors[experiment_name].log_feedback(query, answer, rating, comment)
    
    def get_experiment_comparison(self):
        """比较不同实验的性能"""
        comparison = {}
        
        for name, monitor in self.monitors.items():
            report = monitor.get_performance_report()
            comparison[name] = report
        
        return comparison
    
    def determine_winner(self):
        """确定表现最好的实验"""
        comparison = self.get_experiment_comparison()
        
        # 简单的获胜标准：用户满意度最高，且响应时间合理
        candidates = []
        
        for name, report in comparison.items():
            # 只有有足够反馈的实验才参与比较
            if report["user_satisfaction"]["feedback_count"] >= 10:
                candidates.append((name, report))
        
        if not candidates:
            return None, "没有足够的反馈数据来确定获胜者"
        
        # 首先按用户满意度排序
        candidates.sort(key=lambda x: x[1]["user_satisfaction"]["satisfaction_rate"], reverse=True)
        
        # 然后在满意度相近的情况下，选择响应时间更短的
        winner = candidates[0]
        
        return winner[0], {
            "satisfaction_rate": winner[1]["user_satisfaction"]["satisfaction_rate"],
            "response_time": winner[1]["response_time"]["avg_total"],
            "feedback_count": winner[1]["user_satisfaction"]["feedback_count"]
        }


### 实施建议与最佳实践

#### 1. 优化策略选择指南

在实际应用中，我们需要根据具体场景选择合适的优化策略。以下是一些决策建议：

- **对于小规模知识库**：优先使用语义分块和混合检索，可以不启用复杂的上下文压缩
- **对于大规模知识库**：必须实现高效的检索过滤和上下文压缩机制
- **对于实时性要求高的应用**：减少重排序和压缩步骤，优先保证响应速度
- **对于准确性要求高的应用**：启用交叉编码器重排序和自我批评生成

#### 2. 常见问题的排查流程

当RAG系统出现问题时，可以按照以下流程进行排查：

1. **检索质量问题**：
   - 检查嵌入模型是否合适
   - 验证分块策略是否合理
   - 尝试使用混合检索或查询扩展

2. **生成质量问题**：
   - 优化提示词模板
   - 检查上下文是否包含足够的相关信息
   - 调整生成参数（如温度）

3. **性能问题**：
   - 监控各阶段的耗时
   - 优化向量数据库查询
   - 考虑使用更轻量级的模型

#### 3. 持续优化的关键指标

监控以下关键指标可以帮助持续改进RAG系统：

- **检索相关度**：检索结果与查询的相关性评分
- **上下文利用率**：生成回答中引用上下文的比例
- **用户满意度**：通过评分和反馈收集
- **响应时间**：各阶段的处理时间
- **错误率**：无法回答或回答不准确的比例

### 总结

本文详细介绍了RAG系统的高级优化策略，涵盖了检索质量优化、上下文管理优化和生成质量优化等关键方面。通过实施这些优化策略，我们可以显著提升RAG系统的性能和用户体验。

值得注意的是，RAG系统的优化是一个持续的过程，需要结合具体应用场景、数据特点和用户需求进行调整。建议从小规模测试开始，逐步实施和验证各种优化策略，找到最适合特定应用的组合方案。

在下一篇文章中，我们将探讨RAG系统的实际部署和运维策略，包括如何构建可扩展、高可用的RAG服务，以及如何处理大规模知识库的更新和维护。