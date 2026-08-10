"""Small, explicit safety/classification helpers layered on top of the LLM's
own tool-calling decisions. These are deliberately simple (not a rigid state
machine) -- the LLM still does the planning; this module just adds a
defense-in-depth guardrail for irreversible actions and a human-readable
workflow label for the trace.
"""

CONFIRMATION_PHRASES = (
    "yes",
    "yep",
    "yeah",
    "confirm",
    "confirmed",
    "go ahead",
    "please do",
    "please create",
    "sounds good",
    "do it",
    "approved",
    "that works",
    "please proceed",
    "proceed",
)

IRREVERSIBLE_TOOLS = {"create_mock_hr_ticket"}


def is_user_confirming(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(phrase in lowered for phrase in CONFIRMATION_PHRASES)


def guard_tool_arguments(tool_name: str, arguments: dict, latest_user_message: str) -> dict:
    """Defense-in-depth: even if the model tries to set confirm=true, refuse
    to pass that through unless the user's latest message actually reads as
    an explicit confirmation. Returns a (possibly modified) arguments dict.
    """
    if tool_name in IRREVERSIBLE_TOOLS and arguments.get("confirm") is True:
        if not is_user_confirming(latest_user_message):
            safe_args = dict(arguments)
            safe_args["confirm"] = False
            return safe_args
    return arguments


WORKFLOW_TOOL_HINTS = {
    "remote_work_eligibility": {
        "lookup_employee_profile",
        "search_policy_documents",
        "check_policy_compliance",
    },
    "pto_request_guidance": {
        "check_pto_balance",
        "search_policy_documents",
        "create_mock_hr_ticket",
    },
}


def classify_workflow(tool_names: list[str]) -> str:
    tool_set = set(tool_names)
    best_label, best_overlap = "general_policy_qa", 0
    for label, hint_tools in WORKFLOW_TOOL_HINTS.items():
        overlap = len(tool_set & hint_tools)
        if overlap > best_overlap:
            best_label, best_overlap = label, overlap
    return best_label
