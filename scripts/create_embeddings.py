import sys
import os
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from backend.document_loader import load_and_split_documents
from backend.vector_store_manager import VectorStoreManager

def index_documents(
    knowlodge_base_path : str = "./knowledge_base",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    reset_db: bool = False
):
    #Leer todos los .txt, dividirlos en chunks y retornarlos con metada
    try:
        chunks = load_and_split_documents(
            directory_path=knowlodge_base_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    except Exception as e:
        return None
    #Inicializar el VectorStorageManager
    try:
        vector_store = VectorStoreManager()
        #Necesario cuando se modificaron archivos o se agregan nuevos archivos, mas no cuando se crea la BD desde 0
        if reset_db:
            vector_store.reset_database()
    except Exception as e:
        return None
    #Verificar si hay documentos
    stats_before = vector_store.get_collection_stats()
    docs_before = stats_before.get('total_documents', 0)
    print(f'\nDocumentos a indenixar {docs_before}')
    try:
        docs_added = vector_store.add_documents(chunks)
        if docs_added == 0:
            print('No se agregaron documentos')
            return None
    except Exception as e:
        return None
    
def verify_indexation():
    vector_storage = VectorStoreManager()
    stats = vector_storage.get_collection_stats()
    total_docs = stats.get('total_documents', 0)
    print(f'Documentos totales en chroma: {total_docs}')

if __name__ == "__main__":
    result = index_documents(
        knowlodge_base_path="./knowledge_base",
        chunk_size=500,
        chunk_overlap=50,
        reset_db=True
    )
    
    if result:
        verify_indexation()