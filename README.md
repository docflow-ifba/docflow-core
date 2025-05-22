/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate

# docflow-core

`docflow-core` é um microsserviço desenvolvido com **FastAPI** para realizar o **processamento automático**, **embedding vetorial** e **respostas automatizadas** (Q&A) sobre **editais acadêmicos** enviados em formato PDF.

Este sistema é parte de uma arquitetura maior voltada para auxiliar candidatos e gestores educacionais a entenderem, navegarem e consultarem informações complexas contidas em editais públicos, como os de ingresso em instituições federais de ensino.

---

## ✨ Funcionalidades

- 📨 **Consumo de PDFs via Kafka**: Recebe arquivos PDF através de um tópico Kafka.
- 🧾 **Conversão e limpeza de documentos**: Converte o conteúdo para Markdown, limpa ruídos (logos, rodapés, imagens, etc.) e extrai tabelas em arquivos separados.
- 🧠 **Embedding vetorial**: Gera embeddings com LangChain e FAISS para viabilizar buscas semânticas no conteúdo dos editais.
- 🤖 **Interface de Q&A**: Responde perguntas baseadas no conteúdo embedado utilizando modelos LLM externos via API REST.
- 📡 **API REST**: Permite consultas diretamente pelo endpoint `/query`.

---

## 📦 Estrutura do Projeto

docflow-core/
├── app/
│ ├── main.py
│ ├── api/
│ ├── kafka/
│ ├── services/
│ └── utils/
├── output/
├── pdfs/
├── requirements.txt
└── README.md

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.12
- Kafka em execução local (`localhost:9092`)
- API externa compatível com OpenAI para Q&A (ex: DeepSeek, OpenRouter)

### Instalação

```bash
git clone https://github.com/seu-usuario/docflow-core.git
cd docflow-core
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Iniciar a API
```
uvicorn app.main:app --reload
```

🧪 Exemplo de Consulta
```http
GET /query?prompt=como pagar a taxa de inscrição?
```

### 📤 Enviando PDF via Kafka
Envie uma mensagem para o tópico docflow-embed com um JSON como:

```json
{ "pdf_path": "./pdfs/edital_teste.pdf" }
```

## 🧠 Tecnologias

FastAPI
Kafka
LangChain + FAISS
HuggingFace Embeddings
docling (conversão de PDF para Markdown)
LLM externo via REST (OpenRouter / DeepSeek / Qwen)

📄 Licença
MIT License © 2025