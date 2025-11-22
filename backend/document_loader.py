import os
from typing import List, Dict
from config import KNOWLEDGE_BASE_PATH

def load_single_document(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content.strip()
    except UnicodeDecodeError:
        print(f"Advertencia: {file_path} no está en UTF-8, intentando latin-1")
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read().strip()
        
def load_all_documents(directory_path: str = KNOWLEDGE_BASE_PATH) -> List[Dict[str,str]]:
    documents = []
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Carpeta no encontrada: {directory_path}")
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                content = load_single_document(file_path)
                document = {
                    'content': content,
                    'source': filename,
                    'path': file_path
                }
                
                documents.append(document)
                print(f"Cargado: {filename} ({len(content)} caracteres)")
            except Exception as e:
                print(f"Error al cargar {filename}: {str(e)}")
                continue    
    if len(documents) == 0:
        raise ValueError(f"No se encontraron archivos .txt en {directory_path}")
    
    print(f"\n Total documentos cargados: {len(documents)}")
    return documents

def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    if not text or len(text) == 0:
        return []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor a 0")
    
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunk = chunk.strip()
        
        if chunk:
            chunks.append(chunk)
        
        start += chunk_size - chunk_overlap
    return chunks

def load_and_split_documents(
    directory_path: str = KNOWLEDGE_BASE_PATH,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str,str]]:
    documents = load_all_documents(directory_path)
    all_chunks = []
    
    for doc in documents:
        content = doc['content']
        source = doc['source']
        chunks = split_text_into_chunks(content, chunk_size, chunk_overlap)
        
        for index, chunk in enumerate(chunks):
            chunk_with_metadata = {
                'content': chunk,
                'source': source,
                'chunk_index': index
            }
            all_chunks.append(chunk_with_metadata)
    print(f"\n Total de chunks generados: {len(all_chunks)}")   
    print(f" Tamaño chunk: {chunk_size} caracteres")
    print(f" Overlap: {chunk_overlap} caracteres")
    
    return all_chunks     
    
if __name__ == "__main__":
    print("Probando document_loader.py...\n")
    
    print("=" * 50)
    print("Prueba 1: Cargar documentos completos")
    print("="*50)
    docs = load_all_documents()
    print(f"\nPrimer documento: {docs[0]['source']}")
    print(f"Primeros 200 caracteres:\n{docs[0]['content'][:200]}...")
    
    print("\n" + "=" * 50)
    print("Prueba 2: Dividir en chunks")
    print("="*50)
    chunks = load_and_split_documents(chunk_size=500, chunk_overlap=50)
    print(f"\nPrimer chunk: {chunks[0]['source']} [chunk {chunks[0]['chunk_index']}]")
    print(f"Contenido:\n{chunks[0]['content'][:200]}...")
    