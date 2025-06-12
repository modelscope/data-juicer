import asyncio
import inspect
import unittest
from typing import Optional, Any
from contextlib import AsyncExitStack
from loguru import logger

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

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

    async def process_tool(self, tool_dict: str):
        """Process the LLM response and execute tools if needed.

        Args:
            tool_dict: The response from the LLM.

        Returns:
            Tool execution result and tool called.
        """
        import json

        try:
            tool_call = tool_dict
            if tool_call:
                response = await self.session.list_tools()
                for tool_name, tool_arg in tool_dict.items():
                    try:
                        result = await self.session.call_tool(tool_name, tool_arg)
                        return f"Tool execution result: {result}", tool_call
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        logger.error(error_msg)
                        return error_msg, tool_call

                return f"No tool found: {tool_call['tool']}"
            return tool_dict, tool_call
        except json.JSONDecodeError:
            return tool_dict, None

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


class MCPServerTest(DataJuicerTestCaseBase):

    async def _run_test(self, tool_dict, target):

        client = MCPClient()

        try:
            await client.connect_to_server(
                "./data_juicer/tools/DJ_mcp_recipe_flow.py"
            )
            response, tool_call = await client.process_tool(tool_dict)
            logger.info(str(response))
            self.assertIsNotNone(tool_call)
        finally:
            await client.cleanup()
            await asyncio.sleep(10)  # make sure cleanup done

    def test_text_len_filter(self):
        query = {
            "run_data_recipe": {
                "dataset_path": "./demos/data/demo-dataset.jsonl",
                "process": [
                    {"text_length_filter": {"min_len": 10}},
                    {
                        "language_id_score_filter": {
                            "lang": "zh",
                            "min_score": 0.8,
                        }
                    },
                ],
                "np": 2,
            }
        }
        import sys

        asyncio.run(self._run_test(query, "text_length_filter"), debug=True)


if __name__ == "__main__":
    unittest.main()
