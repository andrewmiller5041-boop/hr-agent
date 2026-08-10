"""Mock-structured-data MCP tools: employee/PTO/benefits lookups and the two
mock write actions (ticket creation, email drafting). Nothing here touches a
real system -- everything reads/writes small JSON files under mock_data/.
"""
import json
from datetime import datetime, timezone

from app import config

_EMPLOYEES_FILE = "employees.json"
_PTO_FILE = "pto_balances.json"
_BENEFITS_FILE = "benefits.json"
_TICKETS_FILE = "tickets.json"


def _load(filename: str):
    path = config.MOCK_DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(filename: str, data) -> None:
    path = config.MOCK_DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def lookup_employee_profile(employee_id: str) -> dict:
    """Look up a mock employee profile by employee_id."""
    employees = _load(_EMPLOYEES_FILE)
    for emp in employees:
        if emp["employee_id"] == employee_id:
            return {"found": True, **emp}
    return {"found": False, "error": f"No employee found with id {employee_id!r}"}


def check_pto_balance(employee_id: str) -> dict:
    """Look up a mock employee's current PTO balance and accrual rate."""
    balances = _load(_PTO_FILE)
    record = balances.get(employee_id)
    if record is None:
        return {"found": False, "error": f"No PTO record found for employee {employee_id!r}"}
    return {"found": True, "employee_id": employee_id, **record}


def lookup_benefits_status(employee_id: str) -> dict:
    """Look up a mock employee's benefits eligibility and enrollment status."""
    benefits = _load(_BENEFITS_FILE)
    record = benefits.get(employee_id)
    if record is None:
        return {"found": False, "error": f"No benefits record found for employee {employee_id!r}"}
    return {"found": True, "employee_id": employee_id, **record}


def create_mock_hr_ticket(
    employee_id: str, summary: str, category: str, confirm: bool = False
) -> dict:
    """Create a MOCK HR ticket. No real ticketing system is affected.

    Safety gate: without confirm=True this returns a PREVIEW only and writes
    nothing. The calling agent must obtain explicit user confirmation before
    ever calling this with confirm=True -- this is the irreversible-action
    guardrail enforced at the tool layer (defense in depth alongside the
    orchestrator-level confirmation step).
    """
    if not confirm:
        return {
            "status": "preview",
            "message": (
                "Preview only -- nothing was created. Ask the user to explicitly "
                "confirm, then call this tool again with confirm=true."
            ),
            "would_create": {
                "employee_id": employee_id,
                "summary": summary,
                "category": category,
            },
        }

    tickets = _load(_TICKETS_FILE)
    ticket_id = (
        "TCK-"
        + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        + "-"
        + str(len(tickets) + 1).zfill(3)
    )
    ticket = {
        "ticket_id": ticket_id,
        "employee_id": employee_id,
        "summary": summary,
        "category": category,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tickets.append(ticket)
    _save(_TICKETS_FILE, tickets)
    return {"status": "created", "ticket": ticket}


def draft_hr_email(employee_id: str, purpose: str, context: str = "") -> dict:
    """Draft (but never send) an HR-related email on behalf of an employee."""
    employees = _load(_EMPLOYEES_FILE)
    name = next(
        (e["name"] for e in employees if e["employee_id"] == employee_id), employee_id
    )
    subject = f"{purpose.strip().capitalize()} - {name}"
    body = (
        f"Hi,\n\nThis note is regarding: {purpose.strip()}.\n\n"
        f"{context.strip()}\n\n"
        f"Thanks,\n{name}"
    )
    return {
        "status": "draft_only",
        "subject": subject,
        "body": body,
        "note": "This is a draft only -- it was not sent. Review and send manually.",
    }
