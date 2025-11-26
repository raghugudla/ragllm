# RAGsmith
A simple Chat UI powered by **Gemma** and **Llama** models via **Ollama**, built with **LangChain**, instrumented with **LangSmith**, and served through **Chainlit**.

---

## 🚀 Open Source Stack

This project demonstrates an end-to-end LLM application using modern open-source tools:

- **LLMs via Ollama**
  - [Llama](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/)
  - [Gemma](https://ai.google.dev/gemma/docs/model_card)  
  Served locally using [Ollama](https://ollama.com/)

- **Framework:** [LangChain](https://www.langchain.com/) for LLM pipelines and retrieval workflows  
- **Observability:** [LangSmith](https://smith.langchain.com/) for tracing, testing, and monitoring  
- **UI:** [Chainlit](https://docs.chainlit.io/langchain) for an interactive chat interface  

---

## 🖥️ System Requirements

1. **Python 3.10+**  
   Earlier versions may not compile or run correctly.

2. **Ollama installed and running**
   - Download: https://ollama.com
   - Pull the required models:
     ```bash
     ollama pull llama3.1
     ollama pull gemma3:270m
     ```

---

## 📦 Installation

Follow these steps to set up the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/raghugudla/ragllm.git
cd ragllm
```

### 2. Execute below commands:
Create python virtual env and install project requirements.
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Directories: 
Create below directories and place your documents under `data/sources`
```
mkdir data/sources
mkdir data/chroma_db
```

### 4. Ingest documents: 
Below script will ingest documents under `data/sources` to chroma db `data/chroma_db`. This is one time setup. 

> **Note:** If you add new documents later, simply rerun this script.
```
python3 ingestion.py 
```

### 5. Environment Variables (Optional)

This project supports optional integration with **LangSmith** for tracing, debugging, performance monitoring, and dataset testing.

#### Create a `.env` file

Inside the project root, create a file named `.env` and add the following:

```env
# LangSmith (optional but highly recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=your_project_name
```
> **Note:** To run without tracing, set:
```env
LANGCHAIN_TRACING_V2=false
```

### 6. Launch UI: 
```
chainlit run main.py --no-cache
```
