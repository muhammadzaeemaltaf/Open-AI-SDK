import os
import chainlit as cl
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig, function_tool
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=provider)

model = RunConfig(model=model, model_provider=provider, tracing_disabled=True)


@function_tool
def about_you():
    return "I am NOBODY"

@function_tool
def time_date():
    return "I dont want to tell you."

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant but someone ask about you use [about_you] function from tool and if ask about time and date so use [time_date] function from tool just simple return text fron there else you are a helpful assistant. Answer questions and assist with tasks.",
    model=model,
    tools=[about_you, time_date],
)


@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])

    await cl.Message(
        content="Hello! I am your AI Assistant. How can I help you?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(agent, input=history, run_config=model)

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)
            # msg.update(event.data.delta)
            await msg.stream_token(event.data.delta)
    
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
