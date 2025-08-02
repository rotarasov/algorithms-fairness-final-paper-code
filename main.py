import json
import warnings

from embedding_models import labse, distilbert, me5_base, me5_large, gte_base, gte_large, qwen3_06B, calculate_similarity
from llm_providers import openai_, deepseek

warnings.simplefilter("ignore", category=FutureWarning)

llm_provider = openai_
embedding_model = me5_large

responses = llm_provider.load_responses()
similarity_output = ""
for i in range(18):
    en = responses["responses"]["en"][i]
    ru = responses["responses"]["ru"][i]
    uk = responses["responses"]["uk"][i]
    
    en_emb = embedding_model.create_embedding(f"query: {en}")
    ru_emb = embedding_model.create_embedding(f"query: {ru}")
    uk_emb = embedding_model.create_embedding(f"query: {uk}")
    
    message = f"Responses #{i+1}:\nEN: {en}\nRU: {ru}\nUK: {uk}\n"
    message += f"Similarity: EN-RU: {calculate_similarity(en_emb, ru_emb)}; EN-UK: {calculate_similarity(en_emb, uk_emb)}; RU-UK: {calculate_similarity(ru_emb, uk_emb)}\n"

    print(message)
    
    similarity_output += message
    
llm_provider.write_similarity_output(embedding_model.short_name, similarity_output)
