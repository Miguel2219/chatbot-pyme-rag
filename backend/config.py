import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "ERROR: OPENAI_API_KEY no encontrada."
        "Verifica que existe en tu archivo .env"
    )

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small" )
TEMPETURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "500"))

CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "restaurante_knowledge")

KNOWLEDGE_BASE_PATH = "./knowledge_base"

TOP_K_RESULTS = 3

COST_PER_1K_TOKENS_CHAT = 0.00015 
COST_PER_1K_TOKENS_EMBEDDING = 0.00002

SYSTEM_PROMPT = """Eres un asistente virtual del Restaurante "El Buen Sabor" en Bogotá, Colombia.
    
    Tu trabajo es ayudar a los clientes respondiendo preguntas sobre:
    - Menú y precios
    - Horarios de atención
    - Servicio de domicilios
    - Reservas
    - Políticas del restaurante
    - Ubicación y contacto
    
    INSTRUCCIONES IMPORTANTES:
    - Sé amable, profesional y conciso
    - Usa el contexto proporcionado para responder
    - Si no tienes información, di "No tengo esa información, pero puedes contactarnos al 300-123-4567"
    - Usa pesos colombianos (COP) para precios
    - Si el cliente quiere hacer un pedido o reserva, dile quete contacte por WhatsApp: 300-123-4567
    - Nunca inventes información que no esté en el contexto
    
    Mantén un tono amigable y cercano, como un mesero experimentado."""
    
print("Configuración cargada:")
print(f"   - Chat Model: {CHAT_MODEL}")
print(f"   - Embedding Model: {EMBEDDING_MODEL}")
print(f"   - ChromaDB Path: {CHROMA_PERSIST_DIRECTORY}")
print(f"   - Collection: {CHROMA_COLLECTION_NAME}")
print(f"   - Top K Results: {TOP_K_RESULTS}")