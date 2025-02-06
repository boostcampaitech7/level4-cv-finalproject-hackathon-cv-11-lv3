import os
import glob 
import json
import numpy as np
import torch 
from angle_emb import AnglE, Prompts

angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

def process_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    file_embeddings = {}

    for video_id, content in data.items():
        timestamps = content.get("timestamps", [])
        sentences  = content.get("sentences" , [])

        timeline_embeddings = {}

        for idx, sentence in enumerate(sentences):
            if idx < len(timestamps):
                ts     = timestamps[idx]
                ts_key = f"{ts[0]}_{ts[1]}"
            else:
                ts_key = f"idx_{idx}"  

            embedding  = angle.encode({'text': sentence}, to_numpy=True, prompt=Prompts.C)
            timeline_embeddings[ts_key] = embedding[0]  

        file_embeddings[video_id] = timeline_embeddings

    return file_embeddings

json_folder    = "json을 저장한 곳 -> 인턴 VL 결과를 돌리고 json으로 저장할 것이라고 생각하고 진행했음"
all_json_files = glob.glob(os.path.join(json_folder, "*.json"))
all_embeddings = {}

for json_file in all_json_files:
    file_embeds = process_json_file(json_file)

    for vid, emb_dict in file_embeds.items():
        all_embeddings[vid] = emb_dict
        print(f"Processed video id: {vid} from file: {os.path.basename(json_file)}")

np.savez("new_videos.npz", **all_embeddings)
print("모든 임베딩이 'new_videos.npz' 파일로 저장되었습니다.")
# 그리고 폴더를 비워야할듯?