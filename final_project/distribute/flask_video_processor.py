import os
import requests
from   models.add_embedding import EmbeddingProcessor
from   models.frame_extract import FrameExtractor

class FlaskVideoProcessor:
    BASE_SAVE_DIR        = '/data/ephemeral/ys/test_videos'
    RESULT_DIR           = '/data/ephemeral/ys/test_jsons'
    EXTRACTED_FRAMES_DIR = '/data/ephemeral/ys/extracted_frames'
    SERVER1_WEBHOOK_URL  = "https://joins-odd-expense-local.trycloudflare.com/trigger_download"
    SERVER3_WEBHOOK_URL  = "https://missouri-detroit-proc-por.trycloudflare.com/trigger_download"
    
    def __init__(self):
        os.makedirs(self.BASE_SAVE_DIR, exist_ok = True)
        os.makedirs(self.RESULT_DIR, exist_ok = True)
        os.makedirs(self.EXTRACTED_FRAMES_DIR, exist_ok = True)
    
    def save_videos(self, uploaded_videos : list):
        file_save_info = []
        
        for idx, uploaded_video in enumerate(uploaded_videos):
            save_dir  = os.path.join(self.BASE_SAVE_DIR, 
                                    f'folder_{(idx % 3) + 1}')
            os.makedirs(save_dir, exist_ok = True)
            
            file_path = os.path.join(save_dir,
                                     uploaded_videos.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
                
            file_save_info.append({"filename": uploaded_video.name, 
                                   "folder": save_dir})
            return file_save_info
        
    def notify_server(self, file_save_info):
        payload = {"folders": [info["folder"] for info in file_save_info]}
        responses = {
            "server1": requests.post(self.SERVER1_WEBHOOK_URL, json=payload),
            "server3": requests.post(self.SERVER3_WEBHOOK_URL, json=payload)
        }
        return responses
    
    def check_json_files(self, expected_files : list):
        existing_files = os.listdir(self.RESULT_DIR)
        missing_files  = [fname for fname in expected_files if fname not in existing_files]
        return missing_files, existing_files
    
    def run_frame_extracting(self, translated_text):
        updated_npz_path  = "/data/ephemeral/ys/embeddings_updated.npz"
        existing_npz_path = "/data/ephemeral/movie_clip_AnglE_UAE_Large_V1_features.npz"
        
        embedding = EmbeddingProcessor(existing_npz_path = existing_npz_path,
                                       json_folder       = self.RESULT_DIR,
                                       updated_npz_path  = updated_npz_path)

        extractor = FrameExtractor(video_dir1        = "/data/ephemeral/home/videos_movieclips",
                                   video_dir2        = self.BASE_SAVE_DIR,
                                   npz_file          = updated_npz_path,
                                   output_dir        = self.EXTRACTED_FRAMES_DIR,
                                   top_k             = 5,
                                   sampling_interval = 500)
        return extractor.extract_frames(translated_text)