from openai import OpenAI
from typing import List
import time

from backend.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    COST_PER_1K_TOKENS_CHAT
)

client = OpenAI(api_key=OPENAI_API_KEY)
 
def get_embedding(text: str) -> List[float]:
    if not text or text.strip() == "":
        raise ValueError ("El texto no puede estar vacio")
    
    #Limpiar el texto antes de enviar a OpenAI
    text = text.replace('\n', ' ').strip()
    
    try:
        response = client.embeddings.create(
            #Texto a convertir
            input=text,
            model=EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error al generar embedding: {str(e)}")
        raise
    
def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    if not texts or len(texts) == 0:
        raise ValueError("La lista de texto no puede estar vacia")
    
    #Limpiar textos antes de enviar
    cleaned_texts = [text.replace('\n', ' ').strip() for text in texts]
    #Filtrar textos vacios despues de limpiar
    cleaned_texts = [t for t in cleaned_texts if t]
    if len(cleaned_texts) == 0:
        raise ValueError("Todos los textos estan vacios despues de limpiar")
    print(f"Generando embeddings para {len(cleaned_texts)} textos...")
    start_time = time.time()
    
    try:
        response = client.embeddings.create(
            input=cleaned_texts,
            model=EMBEDDING_MODEL
        )
        embeddings = [item.embedding for item in response.data]
        
        #Calcular tiempo transcurrido
        elapse_time = time.time() - start_time
        total_tokens = response.usage.total_tokens
        estimated_cost = (total_tokens / 1000) * COST_PER_1K_TOKENS_CHAT
        
        print(f"Embeddings generados en {elapse_time:.2f}s")
        print(f"Tokens usados: {total_tokens:,}")
        print(f"Costo estimado: ${estimated_cost:.6f} USD")
        
        return embeddings
    except Exception as e:
        print(f"Error al generar embeddings batch: {str(e)}")
        raise

def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    if len(embedding1) != len(embedding2):
        raise ValueError(
            f"Embedding no tienen la misma dimension: "
            f"{len(embedding1)} vs {len(embedding2)}"
        )
        
    #Calcular producto punto, el zip empareja elementos
    dot_product = sum(a * b for a, b  in zip(embedding1,embedding2))
    
    #Calcular la magnitude
    magnitude1 = sum(a * a for a in embedding1) ** 0.5
    magnitude2 = sum(b * b for b in embedding2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    similitary = dot_product / (magnitude1 * magnitude2)
    
    return similitary

if __name__ == "__main__":
    print("Ejecutando...")