import os
from openai import OpenAI
from scipy import spatial
import pandas as pd


ATTRIBUTION_KEY = "__attribution__"

def _cosine_similarity(a, b):
    return 1 - float(spatial.distance.cosine(a, b))


class TextEmbdSimilarityAttributor:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    def get_embed(self, text_input):
        completion = self.client.embeddings.create(
            model="text-embedding-v4",
            input=text_input,
            dimensions=1024,
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
                                         column=[{"text_embedding_similarity": s} for s in scores])
        return pd.DataFrame(
            {
                "text": [d["text"] for d in dataset],
                ATTRIBUTION_KEY: [d[ATTRIBUTION_KEY] for d in dataset]
            }
        )
