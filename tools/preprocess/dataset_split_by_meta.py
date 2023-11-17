import os
import fire
import json


SPLIT_FP = {}
SPLIT_LINE_CNT = {}


def gen_data(filepath, split_key):
    with open(filepath) as fp:
        for line in fp:
            line = json.loads(line)
            sk = line["meta"].get(split_key)
            if not sk:
                sk = "none"
            sk = sk.lower().replace(" ", "_")
            yield sk, line


def write_data(sk, line, outpath):
    if sk not in SPLIT_FP:
        print(f"open fp: {sk}.")
        SPLIT_FP[sk] = open(outpath, "w")
        SPLIT_LINE_CNT[sk] = 0
    SPLIT_FP[sk].write(json.dumps(line, ensure_ascii=False))
    SPLIT_FP[sk].write("\n")
    SPLIT_LINE_CNT[sk] += 1


def close():
    for sk, fp in SPLIT_FP.items():
        print(f"close fp: {sk}, {SPLIT_LINE_CNT[sk]} lines.")
        fp.close()


def main(src_dir, target_dir, split_key="Dataset"):
    for filename in os.listdir(src_dir):
        filepath = os.path.join(src_dir, filename)
        name, postfix = filename.split(".")

        for sk, line in gen_data(filepath, split_key):
            outpath = os.path.join(target_dir, f"{name}_{sk}.{postfix}")
            write_data(sk, line, outpath)

    close()


if __name__ == "__main__":
    fire.Fire(main)
