import json
import numpy as np
import torch
from angle_emb import AnglE
import batched

def load_json(file_path):
    """JSON 파일을 로드하는 함수"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def find_tags_by_ids(data, movie_ids):
    """
    특정 영화 ID의 태그를 추출하는 함수.
    각 태그는 (movie_id, timestamp, tag_text) 튜플 형태로 반환합니다.
    """
    tags = []
    for item in data:
        if item['id'] in movie_ids:
            for annotation in item['annotations']:
                tags.append((item['id'], annotation['timestamp'], annotation['tag']))
    return tags

def precompute_tag_embeddings(tags, batch_size=64):
    """
    tags 리스트의 태그 텍스트를 미리 인코딩하여 임베딩 리스트를 반환하는 함수.
    태그 임베딩은 angle 모델을 사용하여 계산합니다.
    """
    tag_texts = [tag[2] for tag in tags]
    embeddings = []
    angle.encode = batched.dynamically(angle.encode, batch_size=64)
    for i in range(0, len(tag_texts), batch_size):
        batch_texts = tag_texts[i:i+batch_size]
        batch_embeddings = angle.encode(batch_texts)
        embeddings.extend(batch_embeddings)
        torch.cuda.empty_cache()
    return embeddings

# 파일 경로 및 모델 로드
file_path = "/data/ephemeral/home/filtered_json_461.json"
data = load_json(file_path)

# 만약 movie_ids가 비어있다면 전체 영화 ID 사용
movie_ids = [item['id'] for item in data]
tags = find_tags_by_ids(data, movie_ids)

# 사전 학습된 Angle 모델 로드 (GPU 사용)
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

# 태그 임베딩 미리 계산
tag_embeddings = precompute_tag_embeddings(tags, batch_size=10)

# 임베딩은 NumPy 배열로 변환하여 저장 (예: tag_embeddings.npy)
np.save("tag_embeddings.npy", np.array(tag_embeddings))

print("태그 임베딩이 'tag_embeddings.npy'파일로 저장되었습니다.")