# Chatbot PYME con RAG - WhatsApp Business

Sistema de chatbot inteligente con **RAG (Retrieval Augmented Generation)** integrado con WhatsApp Business para PYMEs colombianas. Incluye panel de administración web, sistema de versionado de documentos, y desplegado en AWS.

> **Proyecto en desarrollo** - Actualmente en Semana 3 de 12 semanas totales

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
- **Python 3.11+**
- **FastAPI** - API REST con documentación automática
- **Pydantic** - Validación de datos

### IA & RAG
- **OpenAI API**
  - `gpt-4o-mini` - Generación de respuestas
  - `text-embedding-3-small` - Embeddings
- **ChromaDB** - Base de datos vectorial
- **LangChain** - Framework RAG (desde Semana 4)

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

## Estructura del Proyecto
```
chatbot-pyme-rag/
├── backend/                    # Lógica del servidor
├── knowledge_base/             # Documentos del negocio
│   ├── menu.txt
│   ├── horarios.txt
│   ├── politicas.txt
│   └── faqs.txt
├── knowledge_base_versions/    # Sistema de versionado
├── admin_panel/                # Panel web de administración
├── tests/                      # Suite de testing
├── scripts/                    # Scripts de utilidad
├── deployment/                 # Archivos para AWS
├── requirements.txt            # Dependencias Python
└── README.md                   # Este archivo
```

---

## Instalación y Configuración

### Prerequisitos

- Python 3.11 o superior
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

### 4. Configurar variables de entorno

Crear archivo `.env` en la raíz del proyecto:
```env
OPENAI_API_KEY=tu_api_key_aqui
```

---

## Progreso del Proyecto

### Semana 3: RAG Manual (En progreso)
- [x] Estructura de carpetas
- [x] Base de conocimiento (4 archivos .txt)
- [x] Configuración inicial
- [ ] Implementación RAG manual (sin LangChain)
- [ ] Sistema de embeddings
- [ ] ChromaDB setup

### Semana 4: LangChain
- [ ] Refactorización con LangChain
- [ ] Document Loaders
- [ ] Text Splitters
- [ ] RetrievalQA chains

### Semana 5: WhatsApp Integration
- [ ] Twilio + WhatsApp setup
- [ ] Webhooks con FastAPI
- [ ] Historial de conversaciones

## Objetivos de Aprendizaje

Este proyecto está diseñado para aprender:

1. **RAG desde cero** - Entender embeddings, similarity search, chunking
2. **LangChain** - Cuándo usar frameworks vs código manual
3. **Integraciones** - WhatsApp Business API con Twilio
4. **Testing** - Pytest para código confiable
5. **Deploy** - AWS EC2 en producción real
6. **Buenas prácticas** - Código limpio, documentación, versionado

---

## Autor

**Miguel Paba**
- Tecnólogo en Análisis y Desarrollo de Software
- LinkedIn: [https://www.linkedin.com/in/miguel-paba-48580b339/]
- GitHub: [https://github.com/Miguel2219]

---
