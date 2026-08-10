"""Manual (non-framework) agent orchestrator.

Loop: send the conversation + MCP tool schemas to the LLM -> if it requests
tool calls, execute them through the MCP client (never directly) -> feed
results back -> repeat until the model produces a final answer. Every tool
call, its arguments, and a summary of its result are recorded in `trace`, and
every RAG-backed tool call's evidence is collected into `citations`.
"""
import json
from dataclasses import dataclass, field

from groq import BadRequestError, Groq

from app import config
from app.agent import workflows
from app.agent.prompts import SYSTEM_PROMPT
from app.mcp_client.client import MCPClient, MCPToolError

MAX_TOOL_ITERATIONS = 6
MAX_TOOL_CALL_REPAIR_ATTEMPTS = 2

RAG_TOOLS = {"search_policy_documents", "get_policy_section", "check_policy_compliance"}

# Some models served via Groq occasionally emit numeric/boolean tool
# arguments as JSON strings (e.g. "top_k": "4" instead of 4), which Groq's
# API rejects server-side with a 400 before we ever see a tool_call. This
# note is appended and the request retried a couple of times before giving
# up gracefully -- see _create_completion_with_repair below.
_TOOL_ARG_REPAIR_NOTE = {
    "role": "system",
    "content": (
        "Your last tool call used the wrong JSON types for one or more "
        "arguments (e.g. a number written as a string like \"4\" instead of "
        "4, or a boolean written as \"true\" instead of true). Retry the "
        "same tool call with correctly typed JSON arguments."
    ),
}


class ToolCallRepairFailed(Exception):
    """Raised when the model repeatedly produces invalid tool-call JSON and
    cannot be steered back onto a valid call after a few retries."""


def _create_completion_with_repair(client: Groq, messages: list[dict], tools: list[dict]):
    attempt_messages = messages
    last_error: BadRequestError | None = None
    for _ in range(MAX_TOOL_CALL_REPAIR_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=attempt_messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
        except BadRequestError as exc:
            last_error = exc
            attempt_messages = messages + [_TOOL_ARG_REPAIR_NOTE]
    raise ToolCallRepairFailed(str(last_error))


_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    global _groq_client
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env (local) or Render "
            "environment variables (deployed)."
        )
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


@dataclass
class ChatResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    workflow: str = "general_policy_qa"


def _extract_citations(tool_name: str, result: dict) -> list[dict]:
    citations = []
    if tool_name == "search_policy_documents":
        for hit in result.get("results", []):
            citations.append(
                {
                    "doc_id": hit.get("doc_id"),
                    "title": hit.get("title"),
                    "section": hit.get("section"),
                    "snippet": hit.get("snippet"),
                }
            )
    elif tool_name == "get_policy_section":
        for sec in result.get("sections", []):
            citations.append(
                {
                    "doc_id": sec.get("doc_id"),
                    "title": sec.get("title"),
                    "section": sec.get("section"),
                    "snippet": (sec.get("text") or "")[:280],
                }
            )
    elif tool_name == "check_policy_compliance":
        for ev in result.get("evidence", []):
            citations.append(
                {
                    "doc_id": ev.get("doc_id"),
                    "title": ev.get("title"),
                    "section": ev.get("section"),
                    "snippet": ev.get("snippet"),
                }
            )
    return citations


def _dedupe_citations(citations: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for c in citations:
        key = (c.get("doc_id"), c.get("section"))
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


async def handle_message(
    mcp_client: MCPClient,
    message: str,
    employee_id: str | None = None,
    history: list[dict] | None = None,
) -> ChatResult:
    client = get_groq_client()
    tools = mcp_client.openai_tool_schemas()

    user_content = message
    if employee_id:
        user_content = f"[employee_id: {employee_id}] {message}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_content})

    trace: list[dict] = []
    citations: list[dict] = []
    called_tool_names: list[str] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = _create_completion_with_repair(client, messages, tools)
        except ToolCallRepairFailed:
            trace.append(
                {
                    "tool": None,
                    "arguments": None,
                    "result_summary": "Model repeatedly produced invalid tool-call arguments.",
                    "error": "tool_call_repair_failed",
                }
            )
            return ChatResult(
                answer=(
                    "I ran into a technical issue while trying to look up the "
                    "information for this request (the model produced an "
                    "invalid tool call and couldn't self-correct). Please try "
                    "rephrasing your question, or try again."
                ),
                citations=_dedupe_citations(citations),
                trace=trace,
                workflow=workflows.classify_workflow(called_tool_names),
            )
        choice = response.choices[0].message

        if not getattr(choice, "tool_calls", None):
            final_answer = choice.content or ""
            return ChatResult(
                answer=final_answer,
                citations=_dedupe_citations(citations),
                trace=trace,
                workflow=workflows.classify_workflow(called_tool_names),
            )

        messages.append(
            {
                "role": "assistant",
                "content": choice.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.tool_calls
                ],
            }
        )

        for tool_call in choice.tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            arguments = workflows.guard_tool_arguments(name, arguments, message)
            called_tool_names.append(name)

            try:
                result = await mcp_client.call_tool(name, arguments)
                error = None
            except MCPToolError as exc:
                result = {"error": str(exc)}
                error = str(exc)

            if name in RAG_TOOLS and not error:
                citations.extend(_extract_citations(name, result))

            trace.append(
                {
                    "tool": name,
                    "arguments": arguments,
                    "result_summary": json.dumps(result)[:400],
                    "error": error,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    return ChatResult(
        answer=(
            "I wasn't able to finish gathering the information needed to answer "
            "this within the allotted tool-call budget. Could you narrow down "
            "your question?"
        ),
        citations=_dedupe_citations(citations),
        trace=trace,
        workflow=workflows.classify_workflow(called_tool_names),
    )