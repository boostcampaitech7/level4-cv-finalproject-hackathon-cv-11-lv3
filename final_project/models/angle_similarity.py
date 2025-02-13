import torch
import numpy           as np
from   time            import time
from   angle_emb       import AnglE
from   angle_emb.utils import cosine_similarity

MODEL_NAME = 'WhereIsAI/UAE-Large-V1'

class AngleSimilarity:
    def __init__(self,
                 input_text       : str, 
                 npz_file         : str,
                 top_k            : int = 5,
                 model_name       : str = MODEL_NAME, 
                 pooling_strategy : str = 'cls', 
                 device           : str = 'cuda'):
        
        self.input_text       = input_text
        self.npz_file         = npz_file
        self.top_k            = top_k
        self.device           = device if torch.cuda.is_available() else 'cpu'
        self.angle            = AnglE.from_pretrained(model_name, pooling_strategy=pooling_strategy)
        # self.angle            = AnglE.from_pretrained(model_name, pooling_strategy=pooling_strategy)
        if self.device.lower() == 'cuda':
            self.angle = self.angle.cuda()
        
        self.results    = self.compute_angle_similarity()
        
    def __enter__(self):
        self.start_time = time()
        print(f"🚀 [AngleSimilarity] 시작됨...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time() - self.start_time
        print(f"⏳ [AngleSimilarity] 전체 실행 시간: {elapsed_time:.2f}초")
    
    def __load_embeddings(self):
        return np.load(self.npz_file, 
                       allow_pickle = True)
    
    def __compute_text_embedding(self):
        return self.angle.encode([self.input_text], 
                                  to_numpy = True)[0]
    
    def compute_angle_similarity(self):
        data            = self.__load_embeddings()
        input_embedding = self.__compute_text_embedding()
        results         = []
        
        for video_id in data.files:
            timeline_embeddings = data[video_id].item()
            
            for ts_key, embedding in timeline_embeddings.items():
                sim    = cosine_similarity(input_embedding, embedding)
                results.append((video_id, ts_key, sim))
                
        results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
        
        return results_sorted[:self.top_k]

