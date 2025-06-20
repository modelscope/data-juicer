from http import HTTPStatus
import dashscope
from loguru import logger
import json
import yaml
import re
import json
from prompts import SYSTEM_PROMPT


def query(message="Hi"):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': message}
    ]
    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        return response['output']['choices'][0]['message']['content']
    else:
        return 'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        )

def construct_op_dict(json_path="./configs/op_dict.json"):
    op_dict = {}
    with open(json_path, 'r') as json_file:
        ops = json.load(json_file)
    for op in ops:
        op_dict[op['class_name']] = op
    return op_dict


def consult(prompt):
    response = query(prompt)
    try:
        suggestions = parse_suggestion(response)
    except Exception as e:
        logger.warning(e)
        return {}, "No suggestion found."
    suggestion_txt = ""
    for _, s in suggestions.items():
        op_name = s['op_name']
        action = s['action']
        reason = s['reason']
        if action == 'modify':
            arg_name, value = s['arg_name'], s['value']
            suggestion_txt += f"Suggestion: {action} {op_name}:{arg_name} to {value}\n Reason: {reason}\n\n"
        else:
            suggestion_txt += f"Suggestion: {action} {op_name}\n Reason: {reason}\n\n"
    if suggestion_txt == "":
        suggestion_txt = "Current setting is proper. No suggestion found."
    return suggestions, suggestion_txt


def parse_suggestion(response):
    json_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.DOTALL)
    suggestions = json.loads(json_str)
    return suggestions


if __name__ == '__main__':
    query()
