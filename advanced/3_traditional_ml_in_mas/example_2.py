# Реализация кластеризации для RAPTOR из статьи
# Источник: адаптировано из реализации RAPTOR

from typing import List, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAPTORClustering:
    """
    Кластеризация эмбеддингов документов для RAPTOR.
    Использует UMAP (снижение размерности) + Gaussian Mixture (кластеризация).
    """
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def cluster_embeddings(self, embeddings: np.ndarray, target_dim: int = 2) -> np.ndarray:
        """
        Кластеризация эмбеддингов с использованием UMAP + GMM
        Источник: алгоритм из RAPTOR paper[citation:3]
        """
        # Шаг 1: Снижение размерности (UMAP)
        n_neighbors = int((len(embeddings) - 1) ** 0.5) if len(embeddings) > 10 else 5
        reduced = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=target_dim,
            metric='cosine',
            random_state=42
        ).fit_transform(embeddings)
        
        # Шаг 2: Оптимальное число кластеров (BIC)
        max_clusters = min(50, len(embeddings))
        bics = []
        for n in range(2, max_clusters):
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(reduced)
            bics.append(gmm.bic(reduced))
        
        best_n = range(2, max_clusters)[np.argmin(bics)]
        
        # Шаг 3: Финальная кластеризация
        gmm = GaussianMixture(n_components=best_n, random_state=42)
        labels = gmm.fit_predict(reduced)
        
        return labels, reduced
    
    def build_hierarchy(self, documents: List[str]) -> dict:
        """
        Построение иерархической структуры (дерева) для RAPTOR
        Возвращает структуру {level: {cluster_id: {"summary": str, "children": list}}}
        """
        # Шаг 1: Разбиваем документы на чанки
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc)
            all_chunks.extend(chunks)
        
        # Шаг 2: Получаем эмбеддинги
        embeddings = self.embedding_model.embed_documents(all_chunks)
        embeddings = np.array(embeddings)
        
        # Шаг 3: Кластеризуем
        labels, reduced = self.cluster_embeddings(embeddings)
        
        # Шаг 4: Группируем чанки по кластерам
        clusters = {}
        for i, (chunk, label) in enumerate(zip(all_chunks, labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunk)
        
        return {
            "leaf_chunks": all_chunks,
            "clusters": clusters,
            "embeddings": embeddings,
            "reduced_embeddings": reduced
        }

# Использование
documents = ["Полный текст книги 1", "Полный текст книги 2", ...]
raptor = RAPTORClustering()
hierarchy = raptor.build_hierarchy(documents)

print(f"Сформировано {len(hierarchy['clusters'])} смысловых кластеров")
for cluster_id, chunks in hierarchy['clusters'].items():
    print(f"Кластер {cluster_id}: {len(chunks)} чанков")
