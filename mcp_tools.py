import os

import asyncio

from mcp_client import scrape_with_mcp


def scrape_website(url):

    result = asyncio.run(
        scrape_with_mcp(url)
    )

    return result