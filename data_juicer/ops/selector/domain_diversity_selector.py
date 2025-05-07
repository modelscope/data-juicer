import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from pydantic import Field, PositiveInt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing_extensions import Annotated

from data_juicer.ops.base_op import OPERATORS, Selector


@OPERATORS.register_module('domain_diversity_selector')
class DomainDiversitySelector(Selector):
    """Selector to select samples based on the data's domain diversity. """

    def __init__(self,
                 select_ratio: Optional[Annotated[float,
                                                  Field(ge=0, le=1)]] = None,
                 init_k: PositiveInt = None,
                 ebd_dim: PositiveInt = 512,
                 strategy: str = 'inter',
                 model_path: str = 'api',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param select_ratio: The ratio to select.
        :param init_k: The value of k in k-means algorithm.
        :param ebd_dim: The embedding's dimension via API.
        :param strategy: 'inter' - Domain's inter diversity,
        'intra' - Domain's intra diversity, 'global' - Diversity to global centroid.
        :model_path: Local embedding model path, else set 'api' for API embedding.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.select_ratio = select_ratio
        self.init_k = init_k
        self.ebd_dim = ebd_dim
        self.strategy = strategy
        self.model_path = model_path
        self.api_key = os.getenv('OPENAI_API_KEY', 'api-key')
        self.base_url = os.getenv(
            'OPENAI_BASE_URL',
            'https://dashscope.aliyuncs.com/compatible-mode/v1/')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def dataset_embedding(self, dataset):

        embeddings = []

        if self.model_path in ['api', 'API']:
            # Embeddings extract via API
            for entry in tqdm(dataset,
                              desc='Processing embeddings',
                              unit='entry'):
                text = entry['text']
                completion = self.client.embeddings.create(
                    model='text-embedding-v3',
                    input=text,
                    dimensions=self.ebd_dim,
                    encoding_format='float')
                embedding = completion.data[0].embedding
                embeddings.append(embedding)
        else:
            # Embeddings extract via local models
            model = SentenceTransformer(self.model_path,
                                        trust_remote_code=True)

            for entry in tqdm(dataset,
                              desc='Processing embeddings',
                              unit='entry'):
                text = entry['text']
                with torch.no_grad():
                    embedding = model.encode(text).tolist()
                embeddings.append(embedding)

        return np.array(embeddings)

    def domain_diversity_status(self, dataset):

        data_status = []

        embeddings_array = self.dataset_embedding(dataset)
        global_centroid = np.mean(embeddings_array, axis=0)

        # K-means cluster
        kmeans = KMeans(n_clusters=self.init_k, random_state=42)
        labels = kmeans.fit_predict(embeddings_array)

        centroid_embeddings = []
        for label in np.unique(labels):
            label_embeddings = embeddings_array[labels == label]
            centroid = np.mean(label_embeddings, axis=0)
            centroid_embeddings.append(centroid)

        centroid_embeddings = np.array(centroid_embeddings)

        # Sample-level cos-similarity to other centroids
        for i, entry in tqdm(enumerate(dataset),
                             total=len(dataset),
                             desc='Calculating similarity:'):
            current_embedding = embeddings_array[i]
            current_label = int(labels[i])

            similarities = []
            for j, centroid in enumerate(centroid_embeddings):
                if j != current_label:
                    similarity = F.cosine_similarity(
                        torch.tensor(current_embedding).unsqueeze(0),
                        torch.tensor(centroid).unsqueeze(0)).item()
                    similarities.append(similarity)

            own_centroid_similarity = F.cosine_similarity(
                torch.tensor(current_embedding).unsqueeze(0),
                torch.tensor(
                    centroid_embeddings[current_label]).unsqueeze(0)).item()

            global_centroid_similarity = F.cosine_similarity(
                torch.tensor(current_embedding).unsqueeze(0),
                torch.tensor(global_centroid).unsqueeze(0)).item()
            total_similarity = sum(similarities)

            data_status.append({
                'text': entry['text'],
                'label': current_label,
                'similarity_with_other_centroids': similarities,
                'total_similarity': total_similarity,
                'similarity_with_own_centroid': own_centroid_similarity,
                'global_centroid_similarity': global_centroid_similarity,
                'original_index': i
            })

        return data_status, labels

    def diversity_process(self, dataset):
        data_status, labels = self.domain_diversity_status(dataset)
        select_indices = []

        for label in np.unique(labels):
            label_data_status = [
                item for item in data_status if item['label'] == label
            ]

            # Related to the strategy
            if self.strategy == 'inter':
                label_data_status.sort(key=lambda x: x['total_similarity'])
            elif self.strategy == 'intra':
                label_data_status.sort(
                    key=lambda x: x['similarity_with_own_centroid'],
                    reverse=True)
            elif self.strategy == 'global':
                label_data_status.sort(
                    key=lambda x: x['global_centroid_similarity'])
            else:
                raise ValueError(
                    "Invalid strategy. Use 'inter', 'intra' or 'global'.")

            num_to_select = max(
                1, int(self.select_ratio * len(label_data_status)))
            selected_indices = [
                item['original_index']
                for item in label_data_status[:num_to_select]
            ]
            select_indices.extend(selected_indices)
        select_dataset = dataset.select(select_indices)
        return select_dataset

    def process(self, dataset):

        if len(dataset) <= 1:
            return dataset
        if self.select_ratio is None:
            return dataset

        select_dataset = self.diversity_process(dataset)
        return select_dataset
