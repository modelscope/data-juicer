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
    def __init__(self, api_model):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model_key = prepare_model(model_type='api',
                                    model=api_model)

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
    
    async def process_query(self, query: str):
        """Process a query using Claude and available tools"""

        tools = []
        response = await self.session.list_tools()
        tools_description = "\n".join([self.tool_format_for_llm(tool) for tool in response.tools])

        model_api = get_model(self.model_key)

        system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else. Please omit the "
                "variables that were not mentioned.:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )

        messages = [{"role": "system", "content": system_message}]

        messages.append({"role": "user", "content": query})

        llm_response = model_api(messages)
        logger.info(f"\nAssistant: {str(llm_response)}")

        result, tool_call = await self.process_llm_response(llm_response)

        if result != llm_response:
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "system", "content": result})

            final_response = model_api(messages)
            logger.info("\nFinal response: %s", final_response)
            messages.append(
                {"role": "assistant", "content": final_response}
            )
        else:
            messages.append({"role": "assistant", "content": llm_response})

        await self.cleanup()

        return messages, tool_call

    async def process_llm_response(self, llm_response: str):
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            Tool execution result and tool called.
        """
        import json

        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logger.info(f"Executing tool: {tool_call['tool']}")
                logger.info(f"With arguments: {tool_call['arguments']}")

                response = await self.session.list_tools()
                tools = response.tools
                if any(tool.name == tool_call["tool"] for tool in tools):
                    try:
                        result = await self.session.call_tool(tool_call["tool"], tool_call["arguments"])

                        if isinstance(result, dict) and "progress" in result:
                            progress = result["progress"]
                            total = result["total"]
                            percentage = (progress / total) * 100
                            logger.info(
                                f"Progress: {progress}/{total} "
                                f"({percentage:.1f}%)"
                            )

                        return f"Tool execution result: {result}", tool_call
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        logger.error(error_msg)
                        return error_msg, tool_call

                return f"No tool found: {tool_call['tool']}"
            return llm_response, tool_call
        except json.JSONDecodeError:
            return llm_response, None

    async def one_chat(self, query):
        """Run an interactive chat loop"""
        logger.info("\nMCP Client Started!")

        try:
            response, tool_call = await self.process_query(query)
            return response, tool_call

        except Exception as e:
            return (f"\nError: {str(e)}", None)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


class MCPServerTest(DataJuicerTestCaseBase):
    # before running this test, set below environment variables:
    # export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
    # export OPENAI_API_KEY=your_dashscope_key
    api_model = 'qwen2.5-72b-instruct'

    async def _run_test(self, query, target):

        client = MCPClient(self.api_model)

        try:
            await client.connect_to_server('./data_juicer/tools/mcp_server.py')
            response, tool_call = await client.one_chat(query)
            logger.info(str(response))
            self.assertIsNotNone(tool_call)
            self.assertEqual(tool_call["tool"], target)
        finally:
            await client.cleanup()
            await asyncio.sleep(10)     # make sure cleanup done

    def test_text_len_filter(self):
        query = 'Remove samples whose text length is less than 10 and more than 50 in dataset "./demos/data/demo-dataset.jsonl".'
        import sys
        asyncio.run(self._run_test(query, 'text_length_filter'), debug=True)


if __name__ == '__main__':
    unittest.main()
