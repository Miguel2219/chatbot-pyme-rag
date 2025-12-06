from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

def load_and_split_documents(
        directory_path: str = "./knowledge_base",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        glob_pattern: str = "**/*.txt"
) -> List[Document]:
    #DirectoryLoader carga todos losa rchiuvois
    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        loader_cls=TextLoader, #Loader especifico para leer txt
        loader_kwargs={
            'autodetect_encoding': True #Detectar encoding automaticamente
        },
        show_progress=True
    )
    documents = loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        lenght_function=len, #Medir longitud
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    chunks = text_splitter.split_documents(documents)
    #Agregar chux_index a metadata
    chunks_by_source = {}
    for chunk in chunks:
        source=chunk.metadata.get('source', 'unknown')
        if source not in chunks_by_source:
            chunks_by_source[source] = []
        chunks_by_source[source].append(chunk)
    for source, source_chunks in chunks_by_source.items: #Se intera en el diccionario obteniendo como clave source, y source_chunks como la lista de de valores para cada clave
        for i, chunk in enumerate(source_chunks):
            chunk.metadata['chunk_index'] = i
    return chunks
            

        

