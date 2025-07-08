import os
import dashscope
from http import HTTPStatus

ORIGINAL_BASE_HTTP_API_URL = dashscope.base_http_api_url

def setup_copilot_client():
    """
    Configures the dashscope client to target a specified Copilot service endpoint.
    """
    # Set the API endpoint from an environment variable, with a local default.
    service_url = os.getenv("COPILOT_SERVICE_URL", "http://127.0.0.1:6009/api")
    dashscope.base_http_api_url = service_url
    print(f"Copilot client is set to target: {service_url}")

def reset_copilot_client():
    """
    Resets the dashscope client to its original API endpoint.
    """
    dashscope.base_http_api_url = ORIGINAL_BASE_HTTP_API_URL


def call_copilot_service(copilot_chat_history: list = []):
    """
    Calls the Copilot service with the user query and streams the response.

    Args:
        user_query (str): The input text from the user.

    Yields:
        str: Chunks of the assistant's response or an error message.
    """
    # Retrieve model name and API key from environment variables, using dummy defaults.
    model_name = os.getenv("COPILOT_MODEL_NAME", "dummy")
    api_key = os.getenv("DASHSCOPE_API_KEY", "dummy")
    
    setup_copilot_client()

    try:
        # Make a streaming API call to the language model.
        responses = dashscope.Generation.call(
            model=model_name,
            api_key=api_key,
            messages=copilot_chat_history,
            result_format="messages",  # Specifies the response format.
            stream=True,  # Enables streaming mode.
            
        )

        last_yielded_content = ""
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                # Extract the cumulative content from the current stream packet.
                current_cumulative_content = (
                    response.output.choices[0].messages[0].get("content", "")
                )

                # Check if the new content is an extension of the previous content.
                if current_cumulative_content.startswith(last_yielded_content):
                    # Calculate the incremental new text chunk.
                    new_chunk = current_cumulative_content[len(last_yielded_content) :]
                    if new_chunk:
                        yield new_chunk  # Yield only the new part of the message.
                        last_yielded_content = current_cumulative_content
                else:
                    # Fallback: If the stream is out of sequence, yield the entire new content.
                    yield current_cumulative_content
                    last_yielded_content = current_cumulative_content

            else:
                # Handle API-level errors (e.g., bad requests, server errors).
                error_msg = f"API Error: {response.code} - {response.message}"
                yield error_msg
                # return  # Terminate the generator on error.

    except Exception as e:
        # Handle connection errors or other exceptions during the API call.
        error_msg = f"Failed to connect to Copilot service. Make sure it's running. Error: {str(e)}"
        yield error_msg
    
    finally:
        reset_copilot_client()
