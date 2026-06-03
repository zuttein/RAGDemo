from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# Wrapper runt Playwright MCP-tools.
# LLM:en ser endast scrape_website medan wrappern
# hanterar de underliggande Playwright-anropen.

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
        
        # Öppna en stateful MCP-session.
        async with client.session("playwright") as session:
            
            # Hämta Playwright-tools från den aktiva sessionen.
            tools = await load_mcp_tools(session)

            
            navigate_tool = next(
                t for t in tools
                if t.name == "browser_navigate"
            )

            snapshot_tool = next(
                t for t in tools
                if t.name == "browser_snapshot"
            )
            
            # Navigera till önskad webbsida
            nav_result = await navigate_tool.ainvoke({
                "url": url
            })
            
            # Hämta innehåll från samma browser-session
            snapshot_result = await snapshot_tool.ainvoke({})

            return f"""
            Navigation:
            {nav_result}

            --------------------------------

            Snapshot:
            {snapshot_result}
            """

    return scrape_website
