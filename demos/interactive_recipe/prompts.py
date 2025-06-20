SYSTEM_PROMPT = \
"""\
You are an expert data preprocessing assistant, \
specializing in handling multimodal data including text, images, videos, and other AI model-related data. \
Please help me to develop proper operators to process a dataset.\
"""


def dataset_demonstration(
    dataset_snapshots,
    n=3,
):
    res = ""
    for i in range(max(n, len(dataset_snapshots))):
        res += f"- {dataset_snapshots[i]}\n"
    return res


def single_op_arg_info(
    arg_state,
):
    res = \
f"""\
\targ name: {arg_state['name']}
\targ description: {arg_state['desc']}
\targ value: {arg_state['v']}
"""
    if arg_state.get('options', None) is not None:
        res += f"\toptions: {arg_state['options']}\n"
    if arg_state.get('min', None) is not None:
        res += f"\tmin: {arg_state['min']}\n"
    if arg_state.get('max', None) is not None:
        res += f"\tmax: {arg_state['max']}\n"
    return res[:-1]

def single_op_info(
    op_state,
):
    res = \
f"""\
op name: {op_state['name']}\n
op description: {op_state['desc']}\n
op enabled: {op_state['enabled']}\n
args:\n\
"""
    for arg in op_state['args']:
        res += single_op_arg_info(op_state['args'][arg])
        res += '\n\n'
    return res[:-1]
    
    
OUTPUT_INSTRUCTIONS = \
"""\
Please relpy with your suggestions regarding the operator(s). \
You can do the following actions to edit an operator:
- 1. enable: Enable a disabled operator
- 2. disable: Disable an enabled operator
- 3. modify: Modify an argument of an operator
You are required to reply with a json block, \
containing all actions you suggest to apply, with reasons for justification.
An example response is:
```json
{
    "1": {'op_name': op_name_1, 'action': 'enable', 'reason': reason to enable op1},
    "2": {'op_name': op_name_2, 'action': 'disable', 'reason': reason to disable op2},
    "3": {'op_name': op_name_3, 'action': 'modify', 'arg_name': arg_name_1, 'value': arg_value_1, 'reason': reason to modify arg1 as arg_value_1},
}
```
Please note that when you try to perform an action, you should ensure the action is valid.
More specifically, you should ensure the op_name and arg_name are correct.
You should not give any suggestions that will not take effect, e.g., enable an already enabled operator, disable an already disabled operator, or modify an augment to its current value.
When you try to perform a 'modify' action, you should ensure: 
1. the new value lies in options, if options are available.
2. the new value lies in [v_min, v_max], if v_min and v_max are available.


# Additional Instructions
1. You should consider if the function of the operator aligns with the task description.
2. Your response should prioritize the user instruction if available.
3. Your response should not include anything other than a json block.
"""

ATTRIBUTION_PROMPT = """
You are a meticulous data quality assessor for LLM training. Analyze each data sample across multiple quality dimensions and provide numerical scores with reasoning. Follow these guidelines:

1. Evaluation Dimensions
Score each dimension (1-5 scale: 1=lowest, 5=highest):
- Accuracy: Factual correctness & verifiability
- Grammar: Linguistic correctness & fluency
- Informativeness: Depth/utility of content
- Coherence: Logical structure & consistency

2. Scoring Protocol
- Base scores on concrete evidence from text
- Flag samples needing human review (confidence <90%)
- Compare with similar data points for consistency
- Penalize hallucination/misinformation severely

3. Output Format
json
{
  "dimension_scores": {
    "accuracy": ,
    "grammar": ,
    "informativeness": ,
    "coherence":
  },
  "flags": ["syntax_error", "insufficient_information", ...],
  "rationale": "Concise technical analysis",
  "recommendation": ["keep", "review", "discard"]
}
4. Special Instructions
- Prioritize factual integrity over stylistic qualities
- Treat unverified medical/legal claims as high-risk
- Contextualize cultural references appropriately
- Response a json dict

Example Response:

json
{
  "dimension_scores": {
    "accuracy": 2,
    "grammar": 4,
    "informativeness": 4,
    "coherence": 2
  },
  "flags": ["accuracy_concern", "logical_confusion"],
  "rationale": "The text provides rich information but suffers from logical confusion and lacks contextual coherence. Excellent grammatical structure offset by factual inaccuracies.",
  "recommendation": "review"
}
"""

def single_op_prompt(
    op_state,
    task_description=None,
    user_prompt=None,
):
    return \
f"""\
# Task Description
{task_description}

# Operator Info
{single_op_info(op_state)}

# User Instructions
{user_prompt}

# Output Instructions
{OUTPUT_INSTRUCTIONS}
"""

