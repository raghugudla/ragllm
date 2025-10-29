# langchain-gemma-ollama-chainlit
Simple Chat UI using Gemma model via Ollama, LangChain and Chainlit

### Open Source in Action 🚀
- [Gemma](https://ai.google.dev/gemma/docs/model_card) as Large Language model via [Ollama](https://ollama.com/)
- [LangChain](https://www.langchain.com/) as a Framework for LLM
- [LangSmith](https://smith.langchain.com/) for developing, collaborating, testing, deploying, and monitoring LLM applications.
- [Chainlit](https://docs.chainlit.io/langchain) for deploying.

## System Requirements

You must have Python 3.10 or later installed. Earlier versions of python may not compile.

## Installation

Follow these steps to set up the project locally:

1. Clone the repository:
```
git clone https://github.com/raghugudla/ragllm.git/
cd your-repo
```

2. Create below directories
```
mkdir data/sources
mkdir data/chroma_db
```

3. Execute below commands
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Launch UI
```
chainlit run main.py
```
