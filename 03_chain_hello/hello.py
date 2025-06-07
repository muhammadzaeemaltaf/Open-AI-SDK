import os
import chainlit as cl
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

from dotenv import load_dotenv, find_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


# Step 1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


# Step 2: Model
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=provider)

# Config: defining at run level
model = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)


# Step 3: Agent
agent = Agent(
    name="Zaeem AI Assistant",
    instructions="You are a helpful assistant named 'Cyber Agent' created by Zaeem, a brilliant AI developer.You can answer questions and assist with tasks.",
    model=model
)

# Step 4: Run
result = Runner.run_sync(agent, input="Hello, how are you?", run_config=model)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am Zaeem's AI Assistant. How can I help you?").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    result = await Runner.run(agent, input=history, run_config=model)
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()
