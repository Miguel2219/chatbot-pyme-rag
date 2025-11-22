from openai import OpenAI
from typing import List, Dict, Optional
import time
from vector_store_manager import VectorStoreManager
from config import (
    OPENAI_API_KEY,
    CHAT_MODEL,
    TEMPETURE,
    MAX_TOKENS,
    TOP_K_RESULTS,
    SYSTEM_PROMPT,
    COST_PER_1K_TOKENS_CHAT
)

client = OpenAI(api_key=OPENAI_API_KEY)

class RAGEngine:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        print(f" RAGEngine inicializado")
    
    def retrieve(
        self,
        query,
        n_results: int = TOP_K_RESULTS
    ) -> List[Dict]:
        results = self.vector_store.search_similar(query, n_results=n_results)
        return results
    
    def _build_context(self, documents: List[Dict]) -> str:
        if not documents or len(documents) == 0:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents,1):
            doc_text = f"Documento {i} ({doc['source']}):\n{doc['content']}"
            context_parts.append(doc_text)
        context = "\n\n".join(context_parts)
        return context
    
    def _build_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        if context:
            user_content = f"""Usa la siguiente información para responder la pregunta del usuario.

INFORMACIÓN DISPONIBLE:
{context}

PREGUNTA DEL USUARIO:
{query}

INSTRUCCIONES:
- Responde basándote SOLO en la información proporcionada
- Si la respuesta no está en la información, di "No tengo esa información disponible"
- Sé conciso y directo
- Usa formato amigable (sin listas numeradas innecesarias)
            """
        else:
            user_content = f"""No encontré información específica para responder tu pregunta.

PREGUNTA:
{query}

Por favor, intenta reformular tu pregunta o contacta directamente al restaurante:
WhatsApp: 300-123-4567"""
        messages = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': user_content
            }
        ]
        return messages
    
    def generate(
        self,
        query: str,
        context: str
    ) -> Dict:
        print(f'\n Generando respuesta con {CHAT_MODEL}')
        start_time = time.time()
        
        try:
            messages = self._build_prompt(query, context)
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=TEMPETURE,
                max_tokens=MAX_TOKENS
            )
            answer = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            
            result = {
                'answer': answer,
                'model': CHAT_MODEL,
                'time': elapsed_time
            }
            
            return result
        except Exception as e:
            return {
                'answer': 'Lo siento, hubo un error al procesar tu pregunta. Por favor intenta de nuevo',
                'tokens_used': 0,
                'cost': 0.0,
                'model':CHAT_MODEL,
                'time': time.time() - start_time,
                'error': str(e)
            }
    
    def query(
        self,
        query: str,
        n_results: int = TOP_K_RESULTS,
        verbose: bool = True
    ) -> Dict:
        if verbose:
            print("\n" + "=" *60)
            print(f"RAG Query: '{query}'")
            print("=" * 60)
        start_time_total = time.time()
        start_time_retrieval = time.time()
        documents = self.retrieve(query, n_results=n_results)
        start_time_retrieval = time.time() - start_time_retrieval
        if verbose:
            print(f" Encontrados : {len(documents)} documentos")
        context = self._build_context(documents)
        generation_result = self.generate(query, context)
        time_total = time.time() - start_time_total
        result = {
            'query': query,
            'answer': generation_result['answer'],
            'sources': [
                {
                    'source': doc['source'],
                    'chunk_index': doc['chunk_index'],
                    'distance': doc['distance']
                    
                }
                for doc in documents
            ],
            'model': generation_result['model'],
            'time_total': time_total
        }
        return result

if __name__ == "__main__":
    
    print("🧪 Probando RAGEngine...\n")
    print("⚠️  NOTA: Necesitas tener documentos en ChromaDB para que funcione.")
    print("   Si no has ejecutado el script de indexación, hazlo primero.\n")
    
    engine = RAGEngine()
    
    print("\n" + "=" * 60)
    print("PRUEBA: Consulta al RAG Engine")
    print("=" * 60)
    
    test_query = "¿Cuánto cuesta el ajiaco?"
    
    result = engine.query(test_query, verbose=True)
    
    print("\n✅ RAGEngine funciona correctamente!")
    print("\n💡 Prueba con tus propias preguntas modificando test_query")
    print(f"La respuesta es: '{result['answer']}'")