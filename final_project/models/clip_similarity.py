import os
import gc
import cv2
import glob
import torch
from   PIL          import Image
from   time         import time
from   transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"

class ClipVideoProcessor:
    def __init__(self):
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model     = None
        self.clip_processor = None
        
    def __enter__(self):
        self.start_time = time()
        print(f"🚀 [ClipVideoProcessor] 시작됨...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time() - self.start_time
        print(f"⏳ [ClipVideoProcessor] 전체 실행 시간: {elapsed_time:.2f}초")
    
    def __load_model(self):
        if self.clip_model is None or self.clip_processor:
            self.clip_model     = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
    def __unload_model(self):
        if self.clip_model is not None:
            torch.cuda.synchronize()
            del self.clip_model
            del self.clip_processor
            self.clip_model     = None
            self.clip_processor = None
            gc.collect()
            torch.cuda.empty_cache()
            
    def find_video_file_by_movie_id(self, 
                                    video_dir : str, 
                                    movie_id  : str):
        
        pattern = os.path.join(video_dir, f"*{movie_id}*.mp4")
        files   = glob.glob(pattern)
        
        if files:
            return files[0]
        else:
            return None

    def find_best_frame_in_interval(self, 
                                    video_file        : str, 
                                    start_timestamp   : int, 
                                    end_timestamp     : int, 
                                    input_text        : str, 
                                    sampling_interval : int = 500):
        self.__load_model()
        
        text_inputs   = self.clip_processor(text           = [input_text], 
                                            return_tensors = "pt", 
                                            padding        = True
                                            ).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            
        text_features = text_features / text_features.norm(dim = -1, keepdim = True)
        
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            print("Error: 영상 파일을 열 수 없습니다.", video_file)
            return None, None, None

        best_similarity = -1.0
        best_frame      = None
        best_time       = None
        t               = start_timestamp
        
        while t <= end_timestamp:
            cap.set(cv2.CAP_PROP_POS_MSEC, t)
            ret, frame   = cap.read()
            
            if not ret:
                t += sampling_interval
                continue
            
            frame_rgb    = cv2.cvtColor(frame, 
                                      cv2.COLOR_BGR2RGB)
            pil_image    = Image.fromarray(frame_rgb)
            image_inputs = self.clip_processor(images         = pil_image, 
                                               return_tensors = "pt"
                                               ).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**image_inputs)
                
            image_features = image_features / image_features.norm(dim = -1, keepdim = True)
            similarity     = (text_features * image_features).sum(dim = -1).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_frame      = frame.copy()
                best_time       = t
                
            t += sampling_interval
        
        cap.release()
        
        self.__unload_model()
        return best_frame, best_time, best_similarity
