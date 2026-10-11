"""Interactive Google ADK RAG runner. Start with: python main.py"""
from __future__ import annotations

import asyncio
import json
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv

from src.agent import root_agent


async def main() -> None:
    load_dotenv()
    app_name = "document-rag-poc" 
    user_id = "local-user"
    session_id = str(uuid.uuid4())

    sessions = InMemorySessionService()
    await sessions.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=root_agent, app_name=app_name, session_service=sessions)
    
    print("RAG ready. Enter a question (or 'exit' to quit).")
    while True:
        query = input("\nQuestion> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        message = types.Content(role="user", parts=[types.Part(text=query)])
        final_text = ""
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(part.text or "" for part in event.content.parts)
        print(format_response(final_text))


def format_response(text: str) -> str:
    """Keep the terminal response machine-readable even if the model drifts."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").removeprefix("json").strip()
    try:
        payload = json.loads(candidate)
        if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
            raise ValueError("response JSON is missing a string 'answer' field")
        payload.setdefault("status", "answered")
        payload.setdefault("citations", [])
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, ValueError):
        return json.dumps({
            "status": "format_error",
            "answer": text or "The agent returned no text response.",
            "citations": [],
            "message": "The model response was not valid structured JSON; inspect the raw answer above.",
        }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
