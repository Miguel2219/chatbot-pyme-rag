# Chatbot PYME con RAG - WhatsApp Business

Sistema de chatbot inteligente con **RAG (Retrieval Augmented Generation)** integrado con WhatsApp Business para PYMEs colombianas. Incluye panel de administración web, sistema de versionado de documentos, y desplegado en AWS.
---

##  Descripción del Proyecto

Chatbot conversacional que responde preguntas de clientes usando información actualizada de la base de conocimiento del negocio. El sistema usa RAG para buscar información relevante en documentos internos antes de generar respuestas, eliminando alucinaciones y manteniendo información siempre actualizada.

**Caso de uso actual:** Restaurante

**Capacidades:**
- Responde preguntas sobre menú, precios, horarios
- Información sobre domicilios y reservas
- Políticas del negocio y preguntas frecuentes
- Integración WhatsApp Business (Semana 5)
- Panel de administración web (Semanas 8-9)
- Sistema de versionado de documentos (Semana 9)
- Deploy en AWS (Semana 11)

---

## Stack Tecnológico

### Backend & Core
- **Python 3.12+**
- **FastAPI** - API REST con documentación automática
- **Pydantic** - Validación de datos

### IA & RAG
- **OpenAI API**
  - `gpt-4o-mini` - Generación de respuestas
  - `text-embedding-3-small` - Embeddings
- **ChromaDB** - Base de datos vectorial
- **LangChain** - Framework RAG 

### Integraciones
- **Twilio API** - WhatsApp Business (Semana 5)
- **Ngrok** - Webhooks en desarrollo local

### Frontend Admin Panel
- **HTML5 + TailwindCSS + Alpine.js**
- **Jinja2 Templates**

### Testing & Deploy
- **Pytest** - Testing suite
- **AWS EC2** - Servidor producción
- **Nginx** - Reverse proxy
- **GitHub Actions** - CI/CD

---

## Instalación y Configuración

### Prerequisitos

- Python 3.12.7
- Cuenta OpenAI con API key
- Git

### 1. Clonar el repositorio
```bash
git clone https://github.com/Miguel2219/chatbot-pyme-rag.git
cd chatbot-pyme-rag
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate 
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar Script para bases vectorial (Embeddings manualmente)
```bash
python scripts/create_embeddings.py 
```

### 5. Configurar variables de entorno

Crear archivo `.env` en la raíz del proyecto:
```env
OPENAI_API_KEY=tu_api_key_aqui
```

---

## Objetivos de Aprendizaje

Este proyecto está diseñado para aprender:

1. **RAG desde cero** - Entender embeddings, similarity search, chunking
2. **LangChain** - Cuándo usar frameworks vs código manual
3. **Integraciones** - WhatsApp Business API con Twilio
4. **Testing** - Pytest para código confiable
5. **Deploy** - AWS EC2 en producción real

---

## Autor

**Miguel Paba**
- Tecnólogo en Análisis y Desarrollo de Software
- LinkedIn: [https://www.linkedin.com/in/miguel-paba-48580b339/]
- GitHub: [https://github.com/Miguel2219]
