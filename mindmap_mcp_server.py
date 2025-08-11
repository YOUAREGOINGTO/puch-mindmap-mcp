import asyncio
import os
from typing import List, Union

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent

from mindmap_logic import generate_mindmap_func

# --- Load Environment Variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Authentication Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-mindmap-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- MCP Server Setup ---
mcp = FastMCP(
    "Mind Map MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: generate_mindmap ---
@mcp.tool(
    description="Generates a mind map based on a conversation. It will ask clarifying questions if needed."
)
async def generate_mindmap(
    puch_user_id: str,
    user_input: str,
) -> List[Union[TextContent, ImageContent]]:
    return await generate_mindmap_func(puch_user_id, user_input)

# --- Run MCP Server ---
async def main():
    print("🚀 Starting Mind Map MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
