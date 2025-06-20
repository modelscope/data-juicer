import os
from openai import OpenAI
import numpy as np
from scipy import spatial
import scipy.stats

from data_juicer.utils.constant import Fields

ATTRIBUTION_KEY = "__attribution__"

def _cosine_similarity(a, b):
    return 1 - float(spatial.distance.cosine(a, b))


class TextEmbdSimilarityAttributor:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    def get_embed(self, text_input):
        completion = self.client.embeddings.create(
            model="text-embedding-v4",
            input=text_input,
            dimensions=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            encoding_format="float"
        )
        embed = completion.to_dict()["data"][0]["embedding"]
        return embed

    def run(self, dataset, valid_dataset, stats_keys=None):
        embeds = [self.get_embed(d["text"]) for d in dataset]
        valid_embeds = [self.get_embed(d["text"]) for d in valid_dataset]
        scores = [sum([_cosine_similarity(e, ve) for ve in valid_embeds]) / len(valid_embeds) for e in embeds]
        if ATTRIBUTION_KEY not in dataset.features:
            dataset = dataset.add_column(name=ATTRIBUTION_KEY,
                                         column=[{}] * dataset.num_rows)
        for d, score in zip(dataset, scores):
            d[ATTRIBUTION_KEY]["text_embd_sim"] = score

        stats = {}
        dj_stats = dataset[Fields.stats]
        if stats_keys is None:
            stats_keys = list(dj_stats[0].keys())
        for s in dj_stats:
            for k in stats_keys:
                if k not in stats:
                    stats[k] = []
                stats[k].append(s[k])
        attribution_result = {}
        if stats_keys is None:
            stats_keys = stats.columns
        for stats_key in stats_keys:
            pearsonr = scipy.stats.pearsonr(stats[stats_key], scores)[0]
            attribution_result[stats_key] = {"pearsonr": pearsonr}

        return dataset, attribution_result