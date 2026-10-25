# agentic_workflow.py
"""
Simple LangGraph agentic workflow
- 1 Agent
- 3 Tools: get_weather, get_forex, get_news (all free public APIs)
- LLM: ChatOpenAI (gpt-4o-mini)
- Out-of-scope questions → "I don't have much information about this"
"""

import os
import requests
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()


os.environ["LANGSMITH_PROJECT"]="langgraph_agentic_app"


# -------------------------------------------------
# 1. The three tools (free APIs, no keys required)
# -------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Get current weather for any city in the world.
    
    Args:
        city: City name, e.g. 'London', 'New York', 'Tokyo', 'Mumbai'
    """
    try:
        # 1. Geocode the city (Open-Meteo free geocoding)
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_resp = requests.get(geo_url, params={"name": city, "count": 1}, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return f"Could not find location: {city}"

        loc = geo_data["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        name = loc.get("name", city)
        country = loc.get("country", "")

        # 2. Current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        }
        w_resp = requests.get(weather_url, params=params, timeout=10)
        w_resp.raise_for_status()
        current = w_resp.json()["current"]

        # Simple weather code mapping
        codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
        }
        condition = codes.get(current.get("weather_code", -1), "Unknown")

        return (
            f"Weather in {name}, {country}:\n"
            f"- Temperature: {current['temperature_2m']}°C\n"
            f"- Condition: {condition}\n"
            f"- Humidity: {current['relative_humidity_2m']}%\n"
            f"- Wind speed: {current['wind_speed_10m']} km/h"
        )
    except Exception as e:
        return f"Error fetching weather for {city}: {str(e)}"


@tool
def get_forex(base: str = "USD", symbols: str = "EUR,INR,GBP,JPY") -> str:
    """Get latest foreign exchange rates.
    
    Args:
        base: Base currency (default USD). Examples: USD, EUR, INR, GBP
        symbols: Comma-separated target currencies (default EUR,INR,GBP,JPY)
    """
    try:
        url = "https://api.frankfurter.app/latest"
        params = {"from": base.upper(), "to": symbols.upper()}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        rates_str = "\n".join([f"  1 {data['base']} = {rate} {curr}" 
                               for curr, rate in data["rates"].items()])
        return (
            f"Exchange rates (date: {data['date']}):\n"
            f"Base: {data['base']}\n"
            f"{rates_str}"
        )
    except Exception as e:
        return f"Error fetching forex rates: {str(e)}"


@tool
def get_news(limit: int = 5) -> str:
    """Get the latest top news headlines from Hacker News (tech + general interesting stories).
    
    Args:
        limit: Number of top stories to return (1-10, default 5)
    """
    try:
        limit = max(1, min(limit, 10))
        # Top story IDs
        ids_resp = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10
        )
        ids_resp.raise_for_status()
        story_ids = ids_resp.json()[:limit]

        headlines = []
        for i, sid in enumerate(story_ids, 1):
            item = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=8
            ).json()
            title = item.get("title", "No title")
            url = item.get("url", f"https://news.ycombinator.com/item?id={sid}")
            score = item.get("score", 0)
            headlines.append(f"{i}. {title}\n   Score: {score} | {url}")

        return "Top Hacker News stories right now:\n\n" + "\n\n".join(headlines)
    except Exception as e:
        return f"Error fetching news: {str(e)}"


tools = [get_weather, get_forex, get_news]


# -------------------------------------------------
# 2. State
# -------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# -------------------------------------------------
# 3. LLM (Grok-4 via xAI)
# -------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    # api_key is read from XAI_API_KEY env var automatically
)

llm_with_tools = llm.bind_tools(tools)


# -------------------------------------------------
# 4. System prompt – keeps the agent focused
# -------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant that can ONLY answer questions about:
1. Weather (use get_weather)
2. Foreign exchange / currency rates (use get_forex)
3. Latest news headlines (use get_news)

If the user asks about anything outside these three topics, you MUST reply exactly with:
"I don't have much information about this"

Do not invent answers. Use the tools when needed. Be concise and friendly."""


# -------------------------------------------------
# 5. Nodes
# -------------------------------------------------

def agent_node(state: AgentState):
    """Single agent that decides to call tools or answer."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END


# -------------------------------------------------
# 6. Build the simple graph
# -------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "agent")

app = workflow.compile()


# -------------------------------------------------
# 7. Run helper
# -------------------------------------------------

def chat(user_input: str):
    print(f"\n🧑  You: {user_input}")
    print("-" * 60)
    result = app.invoke({"messages": [HumanMessage(content=user_input)]})
    final = result["messages"][-1]
    print(f"🤖  Agent: {final.content}")
    return final.content


# -------------------------------------------------
# 8. Demo
# -------------------------------------------------

if __name__ == "__main__":
    # Make sure you have set: export XAI_API_KEY="your-xai-key"
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set the environment variable XAI_API_KEY")
        exit(1)

    print("Simple Weather / Forex / News Agent (powered by Grok-4)\n")

    # In-scope examples
    chat("What's the weather in Tokyo right now?")
    chat("Show me USD to INR and EUR exchange rates")
    chat("Give me the top 3 news headlines")

    # Out-of-scope example
    chat("Who won the last FIFA World Cup?")