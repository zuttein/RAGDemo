import asyncio
import re
from mcp import ClientSession
from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters
)

def clean_text(text):

    text = re.sub(r"\[ref=.*?\]", "", text)
    text = re.sub(r"\[cursor=.*?\]", "", text)
    text = re.sub(r"- /url:.*", "", text)
    text = re.sub(r"- img.*", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


async def scrape_with_mcp(url):

    server_params = StdioServerParameters(
    command="npx",
    args=["@playwright/mcp"]
)

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()
            print("\nMCP INITIALIZED\n")

            tools = await session.list_tools()

            print(tools)

            await session.call_tool(
                "browser_navigate",
                {
                    "url": url
                }
            )
            print(f"\nNAVIGATED TO: {url}\n")

            snapshot = await session.call_tool(
                "browser_snapshot",
                {}
            )
            print("\nSNAPSHOT RECEIVED\n")
            
            raw_text = snapshot.content[0].text
            
            cleaned_text = clean_text(raw_text)

            return cleaned_text