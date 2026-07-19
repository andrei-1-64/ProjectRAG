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
        self.fit(self.documents) 

    def fit(self, documents):
        self.documents = documents
        tokenized_corpus = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
       
        embeddings = self.model.encode(documents)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query):
     
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        
        query_embedding = self.model.encode([query])
        _, dense_indices = self.index.search(np.array(query_embedding).astype('float32'), self.top_k)
        
       
        return self.documents[:self.top_k] 