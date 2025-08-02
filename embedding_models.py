import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingModel:
    def __init__(self, short_name: str, name: str):
        self.short_name = short_name
        self._model = SentenceTransformer(name)
    
    def create_embedding(self, sentence: str):
        return self._model.encode([sentence])


gte_base = EmbeddingModel("gte_base", "thenlper/gte-base")
gte_large = EmbeddingModel("gte_large", "thenlper/gte-large")
labse = EmbeddingModel("labse", "sentence-transformers/LaBSE")
distilbert = EmbeddingModel("distilbert", "sentence-transformers/distiluse-base-multilingual-cased-v2")
me5_base = EmbeddingModel("me5_base", "intfloat/multilingual-e5-base")
me5_large = EmbeddingModel("me5_large", "intfloat/multilingual-e5-large")
qwen3_06B = EmbeddingModel("qwen3_06B", "Qwen/Qwen3-Embedding-0.6B")

def calculate_similarity(embedding1: np.array, embedding2: np.array):
    return cosine_similarity(embedding1, embedding2)[0][0]