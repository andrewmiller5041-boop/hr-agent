"""MCP server exposing 8 HR tools over stdio.

Run standalone for manual testing:
    python mcp_server/server.py

The FastAPI app spawns this same script as a subprocess (see
app/mcp_client/client.py) and talks to it over stdio using the MCP protocol --
the agent never calls these Python functions directly.
"""
import sys
from pathlib import Path

# Ensure the repo root is importable as `app.*` and `mcp_server.*` regardless
# of the working directory this script is launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_server.tools import hr_data_tools, policy_tools  # noqa: E402

mcp = FastMCP("hr-agent-tools")


# ---- RAG / policy-evidence tools -------------------------------------------

@mcp.tool()
def search_policy_documents(query: str, top_k: int = 4) -> dict:
    """Semantic search over the company policy corpus (RAG). Returns the
    top_k most relevant chunks, each with doc_id, title, section, and a
    source snippet suitable for citation."""
    return policy_tools.search_policy_documents(query, top_k)


@mcp.tool()
def get_policy_section(doc_id: str, section: str = "") -> dict:
    """Directly fetch section(s) of a known policy document by its doc_id
    (e.g. 'pto-policy'), optionally filtered to a section whose heading
    contains the given text."""
    return policy_tools.get_policy_section(doc_id, section or None)


@mcp.tool()
def check_policy_compliance(scenario: str, policy_area: str = "") -> dict:
    """Retrieve grounded policy evidence relevant to a described real-world
    scenario (e.g. an expense or a remote-work request) so a compliance
    determination can be made from cited text rather than guesswork."""
    return policy_tools.check_policy_compliance(scenario, policy_area or None)


# ---- Mock structured-data tools --------------------------------------------

@mcp.tool()
def lookup_employee_profile(employee_id: str) -> dict:
    """Look up a mock employee profile (role, department, location, manager,
    employment type, hire date) by employee_id, e.g. 'E1001'."""
    return hr_data_tools.lookup_employee_profile(employee_id)


@mcp.tool()
def check_pto_balance(employee_id: str) -> dict:
    """Look up a mock employee's current PTO balance, accrual rate, and any
    pending requests."""
    return hr_data_tools.check_pto_balance(employee_id)


@mcp.tool()
def lookup_benefits_status(employee_id: str) -> dict:
    """Look up a mock employee's benefits eligibility and current
    enrollment (medical plan, dental/vision, 401k, life insurance)."""
    return hr_data_tools.lookup_benefits_status(employee_id)


@mcp.tool()
def create_mock_hr_ticket(
    employee_id: str, summary: str, category: str, confirm: bool = False
) -> dict:
    """Create a MOCK HR ticket (no real system is affected). Without
    confirm=true this only returns a preview and writes nothing -- the agent
    must get explicit user confirmation before calling this with
    confirm=true."""
    return hr_data_tools.create_mock_hr_ticket(employee_id, summary, category, confirm)


@mcp.tool()
def draft_hr_email(employee_id: str, purpose: str, context: str = "") -> dict:
    """Draft (but never send) an HR-related email on behalf of an employee.
    Always returns text for the user to review; this tool cannot send mail."""
    return hr_data_tools.draft_hr_email(employee_id, purpose, context)


if __name__ == "__main__":
    mcp.run()
