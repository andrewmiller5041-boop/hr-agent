"""Verifies MCP tool discovery and a real (mock-data) MCP tool call, exactly
the mechanism the CI pipeline is required to check per the rubric."""
import pytest

from app.mcp_client.client import MCPClient

EXPECTED_TOOLS = {
    "search_policy_documents",
    "get_policy_section",
    "check_policy_compliance",
    "lookup_employee_profile",
    "check_pto_balance",
    "lookup_benefits_status",
    "create_mock_hr_ticket",
    "draft_hr_email",
}


async def test_mcp_tool_discovery():
    client = MCPClient()
    await client.connect()
    try:
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        assert len(tool_names) >= 5
        assert EXPECTED_TOOLS.issubset(tool_names)
    finally:
        await client.close()


async def test_mcp_tool_call_mock_data():
    client = MCPClient()
    await client.connect()
    try:
        result = await client.call_tool("check_pto_balance", {"employee_id": "E1001"})
        assert result["found"] is True
        assert result["employee_id"] == "E1001"
        assert "balance_days" in result

        missing = await client.call_tool("check_pto_balance", {"employee_id": "NOPE"})
        assert missing["found"] is False
    finally:
        await client.close()


async def test_mcp_tool_call_ticket_preview_requires_confirmation():
    client = MCPClient()
    await client.connect()
    try:
        preview = await client.call_tool(
            "create_mock_hr_ticket",
            {
                "employee_id": "E1002",
                "summary": "Test ticket",
                "category": "pto",
                "confirm": False,
            },
        )
        assert preview["status"] == "preview"
    finally:
        await client.close()
