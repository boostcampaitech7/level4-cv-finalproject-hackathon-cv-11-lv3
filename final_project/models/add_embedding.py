import os
import glob
import json
import numpy     as     np
import torch
from   angle_emb import AnglE, Prompts
from   time      import time

class EmbeddingProcessor:
    def __init__(self, 
                 existing_npz_path : str, 
                 json_folder       : str, 
                 updated_npz_path  : str,
                 model_name        : str = 'WhereIsAI/UAE-Large-V1', 
                 pooling_strategy  : str = 'cls', 
                 device            : str = 'cuda'):
        
        self.existing_npz_path = existing_npz_path
        self.json_folder       = json_folder
        self.updated_npz_path  = updated_npz_path
        self.device            = device if torch.cuda.is_available() else 'cpu'
        self.angle             = AnglE.from_pretrained(model_name, pooling_strategy=pooling_strategy)
        
        if self.device.lower() == 'cuda':
            self.angle = self.angle.cuda()

        self.merge_embeddings()
        
    def __enter__(self):
        self.start_time = time()
        print(f"🚀 [EmbeddingProcessor] 시작됨...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time() - self.start_time
        print(f"⏳ [EmbeddingProcessor] 전체 실행 시간: {elapsed_time:.2f}초")
    
    def __process_json_file(self, json_path : str):
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        file_embeddings = {}
        
        for video_id, content in data.items():
            timestamps          = content.get("timestamps", [])
            sentences           = content.get("sentences", [])
            timeline_embeddings = {}

            for idx, sentence in enumerate(sentences):
                if idx < len(timestamps):
                    ts     = timestamps[idx]
                    ts_key = f"{ts[0]}_{ts[1]}"
                else:
                    ts_key = f"idx_{idx}"
                    
                embedding                   = self.angle.encode({'text': sentence}, 
                                                                 to_numpy = True, 
                                                                 prompt   = Prompts.C)
                timeline_embeddings[ts_key] = embedding[0]
                
            file_embeddings[video_id]       = timeline_embeddings
        return file_embeddings

    @staticmethod
    def load_existing_npz(npz_path : str):
        if os.path.exists(npz_path):
            npz_data  = np.load(npz_path,
                                allow_pickle = True)
            data_dict = {key: npz_data[key] for key in npz_data.files}
            npz_data.close()
            return data_dict
        else:
            return {}

    def merge_embeddings(self):
        all_embeddings = self.load_existing_npz(self.existing_npz_path)
        all_json_files = glob.glob(os.path.join(self.json_folder, "*.json"))
        
        for json_file in all_json_files:
            json_title = os.path.splitext(os.path.basename(json_file))[0]
            
            if json_title in all_embeddings:
                print(f"이미 존재하는 video_id '{json_title}'의 JSON 파일은 건너뜁니다.")
                continue

            new_embeds = self.__process_json_file(json_file)
            
            for video_id, emb_dict in new_embeds.items():
                if video_id in all_embeddings:
                    print(f"이미 존재하는 비디오 id: {video_id} (건너뜁니다) from 파일: {os.path.basename(json_file)}")
                    
                else:
                    all_embeddings[video_id] = emb_dict
                    print(f"추가된 비디오 id: {video_id} from 파일: {os.path.basename(json_file)}")
                    
        np.savez(self.updated_npz_path, **all_embeddings)
        print(f"모든 임베딩이 '{self.updated_npz_path}' 파일에 저장되었습니다.")
