import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import time
from embeddings_manual import get_embeddings_batch
from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME
)

class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path = persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True #En produccion false
            )
        )
        self.collection = self._get_or_create_collection()
        
        print(f"VectorStoreManager Inicializado")
        print(f"Directorio: {persist_directory}")
        print(f"Coleccion: {collection_name}")
        
    def _get_or_create_collection(self):
            try:
                collection = self.client.get_collection(name=self.collection_name)
                print(f"Coleccion {self.collection_name} encontrada")
            except Exception:
                collection = self.client.create_collection(name=self.collection_name)    
                print(f"Coleccion {self.collection_name} creada")
            return collection

    def add_documents(
        self,
        documents: List[Dict[str, str]],
        batch_size: int = 100,
    ) -> int:
        if not documents or len(documents) == 0:
            print("No hay documentos para agregar")
            return 0
        print(f"\n Agregando {len(documents)} documentos a ChromaDB...")
        start_time = time.time()
        total_added = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            print(f" Procesando lote {batch_num}/{total_batches}...")
            
            try:
                texts = [doc['content'] for doc in batch]
                embeddings = get_embeddings_batch(texts)
                ids = [
                    f"{doc['source']}_chunk{doc['chunk_index']}"
                    for doc in batch
                ]
                metadatas = [
                    {
                        'source': doc['source'],
                        'chunk_index': doc['chunk_index']
                    }
                    for doc in batch
                ]
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas
                )
                total_added += len(batch)
            except Exception as e:
                continue
        elapsed_time = time.time() - start_time
        print(f"\nDocumentos agregados: {total_added}/{len(documents)}")
        print(f"   Tiempo total: {elapsed_time:.2f}s")
        print(f"   Promedio: {elapsed_time/len(documents):.3f}s por documento")
        
        return total_added   
    
    def search_similar(
        self,
        query: str,
        n_results: int = 3,
    ) -> List[Dict]:
        if not query or query.strip() == "":
            print("Query vacía")
            return []

        print(f"Buscando documentos similares a: {query}")
        start_time = time.time()
        
        try:
            from embeddings_manual import get_embedding
            query_embedding = get_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            processed_results = []
            for i in range(len(results['documents'][0])):
                doc_result = {
                    'content': results['documents'][0][i],
                    'source': results['metadatas'][0][i]['source'],
                    'chunk_index': results['metadatas'][0][i]['chunk_index'],
                    'distance': results['distances'][0][i]
                }
                processed_results.append(doc_result)
            elapsed_time = time.time() - start_time
            
            print(f"Encontrados {len(processed_results)} documentos")
            print(f"Tiempo: {elapsed_time:.3f}s")
            
            for i, doc in enumerate(processed_results, 1):
                print(f"{i}. [{doc['source']}] Distancia: {doc['distance']:.4f}")
                print(f"Preview: {doc['content'][:80]}...")
                return processed_results
        except Exception as e:
            print(f"Error en busqueda: {str(e)}")
            return []    
    
    def get_collection_stats(self) -> Dict:
        try:
            count = self.collection.count()
            
            stats = {
                'collection_name': self.collection_name,
                'total_documents': count,
                'persist_directory': self.persist_directory
            }
            
            return stats
        except Exception as e:
            print(f"Error al obtener estadisticas: {str(e)}")
            return {}
        
    def delete_collection(self) -> bool:
        try:
          self.client.delete_collection(name=self.collection_name)
          self.collection = self._get_or_create_collection()
          return True
        except Exception as e:
            return False    
        
    def reset_database(self) -> bool:
        try:
            self.client.reset()
            self.collection = self._get_or_create_collection()
            return True
        except Exception as e:
            return False

if __name__ == "__main__":
    
    print("Probando VectorStoreManager...\n")
    print("=" * 50)
    print("PRUEBA 1: Inicializar VectorStoreManager")
    print("=" * 50)
    
    manager = VectorStoreManager()
    
    print("\n" + "=" * 50)
    print("PRUEBA 2: Agregar documentos")
    print("=" * 50)
    
    test_documents = [
        {
            'content': 'Ajiaco Santafereño cuesta $12.000 pesos colombianos',
            'source': 'test.txt',
            'chunk_index': 0
        },
        {
            'content': 'Bandeja Paisa vale $15.000 pesos colombianos',
            'source': 'test.txt',
            'chunk_index': 1
        },
        {
            'content': 'Horario de lunes a viernes de 8AM a 8PM',
            'source': 'test.txt',
            'chunk_index': 2
        }
    ]
    
    added = manager.add_documents(test_documents)
    print(f"\nResultado: {added} documentos agregados")
    
    print("\n" + "=" * 50)
    print("PRUEBA 3: Estadísticas de la colección")
    print("=" * 50)
    
    stats = manager.get_collection_stats()
    print(f"Colección: {stats['collection_name']}")
    print(f"Total documentos: {stats['total_documents']}")
    
    print("\n" + "=" * 50)
    print("PRUEBA 4: Búsqueda de similitud")
    print("=" * 50)
    
    query1 = "¿Cuál es el precio del ajiaco?"
    results1 = manager.search_similar(query1, n_results=2)
    
    query2 = "¿Cuándo abren?"
    results2 = manager.search_similar(query2, n_results=2)
    
    print("\nVectorStoreManager funciona correctamente!")
    print("\nNota: Los documentos de prueba quedan en ChromaDB.")
    print("   Para limpiar, ejecuta: manager.reset_database()")