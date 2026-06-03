from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def create_scrape_tool():

    @tool
    async def scrape_website(url: str) -> str:
        """
        Scrape a website and return the page content.
        """

        # Starta Playwright-MCP först när toolen används, så appen inte öppnar en session i onödan.
        client = MultiServerMCPClient(
            {
                "playwright": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["@playwright/mcp"]
                }
            }
        )

        async with client.session("playwright") as session:

            tools = await load_mcp_tools(session)

            # Plocka ut de två MCP-tools som behövs för en enkel sidläsning.
            navigate_tool = next(
                t for t in tools
                if t.name == "browser_navigate"
            )

            snapshot_tool = next(
                t for t in tools
                if t.name == "browser_snapshot"
            )

            nav_result = await navigate_tool.ainvoke({
                "url": url
            })

            snapshot_result = await snapshot_tool.ainvoke({})

            return f"""
            Navigation:
            {nav_result}

            --------------------------------

            Snapshot:
            {snapshot_result}
            """

    return scrape_website
