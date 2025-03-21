import asyncio
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
    # methods will go here

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
    
    def tool_format_for_llm(self, tool) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in tool.inputSchema:
            for param_name, param_info in tool.inputSchema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in tool.inputSchema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"\nTool: {tool.name}\nDescription: {tool.description}\nArguments:\n{chr(10).join(args_desc)}\n"
    
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        tools = []
        response = await self.session.list_tools()
        tools_description = "\n".join([self.tool_format_for_llm(tool) for tool in response.tools])

        print(tools_description)
        return tools_description

    async def one_chat(self, query):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")

        try:
            response = await self.process_query(query)
            print("\n" + response)

        except Exception as e:
            print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():

    client = MCPClient()

    try:
        await client.connect_to_server('./data_juicer/tools/mcp_server.py')
        await client.one_chat('hello!')
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
