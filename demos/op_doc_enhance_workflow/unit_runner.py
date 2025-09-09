import os
import json
from utils.parse_class import literal_eval_universal
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper import *
from pathlib import Path

examples = []

example_path = os.path.join(os.path.dirname(__file__), "examples.json")

with open(example_path, "r") as f:
    examples = json.load(f)

for k, v in examples.items():
    if "blur_mapper" in k and v:
        v_1 = list(v.values())[0]
        if not v_1.get("tgt"):
            print(k)
            break

e = v_1["ds"]

d = v_1["op_code"]

c = literal_eval_universal(e)
dataset = Dataset.from_list(c)
op = eval(d)

result = op.run(dataset)

print(result.to_list())