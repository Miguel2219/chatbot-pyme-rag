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
    print("Probando embeddings_manual.py...\n")
    
    print("=" * 50)
    print("Prueba 1: Embedding de un texto")
    print("=" * 50)
    
    texto_prueba = "Ajiaco Santafereño cuesta $12.000"
    print(f"Texto: '{texto_prueba}'")
    
    embedding = get_embedding(texto_prueba)
    print(f"Embedding generado")
    print(f"Dimesiones: {len(embedding)}")
    print(f"Primeros 5 valores: {embedding[:5]}")
    print(f"Ultimos 5 valores: {embedding[-5:]}")
    
    print("\n" + "=" * 50)
    print("PRUEBA 2: Embeddings batch")
    print("=" * 50)
    
    textos_prueba = [
        "Ajiaco Santafereño cuesta $12.000",
        "Bandeja Paisa vale $15.000",
        "Horario de lunes a viernes 8AM a 8PM"
    ]
    
    embeddings = get_embeddings_batch(textos_prueba)
    print(f"{len(embeddings)} embeddings generados")
    print("\n" + "=" *50)
    print("Prueba 3: Similitud coseno")
    print("=" * 50)
    
    texto1 = "Ajiaco Santafereño cuesta $12.000"
    texto2 = "¿Cuál es el precio del ajiaco?"
    texto3 = "Horario de atención del restaurante"
    
    emb1 = get_embedding(texto1)
    emb2 = get_embedding(texto2)
    emb3 = get_embedding(texto3)
    
    sim_relacionados = cosine_similarity(emb1, emb2)
    sim_diferentes = cosine_similarity(emb1, emb3)
    
    print(f"\nTexto 1: '{texto1}'")
    print(f"Texto 2: '{texto2}'")
    print(f"Similitud (relacionados): {sim_relacionados:.4f}")
    
    print(f"\nTexto 1: '{texto1}'")
    print(f"Texto 3: '{texto3}'")
    print(f"Similitud (diferentes): {sim_diferentes:.4f}")
    
    print("\nembeddings_manual.py funciona correctamente!")
    print("\nNota: Textos relacionados tienen similitud > 0.7")
    print("Textos diferentes tienen similitud < 0.5")