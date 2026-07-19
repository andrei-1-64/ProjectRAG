from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class HybridRetriever:
    def __init__(self, config):
        self.top_k = config.get('top_k', 5)
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.documents = ["București este capitala României.", "Politehnica București a fost fondată în 1818, de Gheroge Lazar."] # Exemplu de date
        self.bm25 = None
        self.index = None
        self.fit(self.documents) # Inițializăm automat la pornire

    def fit(self, documents):
        self.documents = documents
        tokenized_corpus = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Creare index dense
        embeddings = self.model.encode(documents)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query):
        """Combinație hibridă: recuperare + reranking simplificat"""
        # 1. Sparse search
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. Dense search
        query_embedding = self.model.encode([query])
        _, dense_indices = self.index.search(np.array(query_embedding).astype('float32'), self.top_k)
        
        # 3. Intersecția/Fuziunea rezultatelor
        # (Aici poți implementa un scor ponderat între BM25 și FAISS)
        return self.documents[:self.top_k] # Returnează top K rezultate