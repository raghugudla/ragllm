import chainlit as cl
from dotenv import load_dotenv
from rag_retrieve import retrieve_context
from rag_augment import format_context, extract_doc_names, build_augmented_prompt
from rag_generate import get_chain, AVAILABLE_MODELS
import asyncio

load_dotenv()

@cl.on_chat_start
async def start():
    # Send image first
    await cl.Message(
        elements=[cl.Image(name="logo", display="inline", path="mau.png")],
        content=""
    ).send()

    # Then send model selection message
    await cl.Message(
        content=f"Hello! I am your RAG assistant. **Current model: {AVAILABLE_MODELS[0]}**\n\nSelect a model below or start chatting:",
        actions=[
            cl.Action(name="select_model", value=model, label=f"Use {model}", payload={"model": model})
            for model in AVAILABLE_MODELS
        ]
    ).send()

    # Initialize default model in session
    cl.user_session.set("selected_model", AVAILABLE_MODELS[0])


@cl.on_message
async def main(message: cl.Message):
    selected_model = cl.user_session.get("selected_model") or AVAILABLE_MODELS[0]
    chain = get_chain(selected_model)

    user_q = message.content
    #print(f"User query: {user_q}")

    # Retrieve documents with metadata
    retrieved_docs = retrieve_context(user_q, k=4)
    #print(f"Retrieved docs: {retrieved_docs}")

    final_context = format_context(retrieved_docs)
    #print(f"Final context:\n{final_context}")
    
    doc_names = extract_doc_names(retrieved_docs)

    # Build prompt including context and user query
    combined_prompt = build_augmented_prompt(user_q, final_context)
    #print(f"Combined prompt:\n{combined_prompt}")

    # Invoke LLM chain asynchronously
    response = await asyncio.to_thread(chain.invoke, input={"user_input": combined_prompt})
    #print("DONE!")

    # Send LLM response with model selection buttons
    await cl.Message(
        content=f"**[Model: {selected_model}]**\n\n{response.content}",
        actions=[
            cl.Action(name="select_model", value=model, label=f"Use {model}", payload={"model": model})
            for model in AVAILABLE_MODELS
        ]
    ).send()


@cl.action_callback("select_model")
async def on_model_selected(action: cl.Action):
    model_name = action.payload["model"]
    cl.user_session.set("selected_model", model_name)
    await cl.Message(content=f"✅ Model switched to: **{model_name}**").send()
