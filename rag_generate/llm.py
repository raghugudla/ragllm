from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


chat_prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="{user_input}",
)

AVAILABLE_MODELS = ["gemma3:270m", "llama3.1"]


def create_llm(model_name: str = "gemma3:270m"):
    """Create an LLM instance for the given model name.
    
    Args:
        model_name: Name of the model to use. Must be one of AVAILABLE_MODELS.
        
    Returns:
        ChatOllama instance configured with the specified model.
        
    Raises:
        ValueError: If model_name is not in AVAILABLE_MODELS.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {AVAILABLE_MODELS}")
    return ChatOllama(temperature=0, model=model_name)


def get_chain(model_name: str = "gemma3:270m"):
    """Get the chain for a given model.
    
    Args:
        model_name: Name of the model to use.
        
    Returns:
        Composed chain with the specified model.
    """
    print(f"Creating chain for {model_name}")
    llm = create_llm(model_name)
    return chat_prompt_template | llm

