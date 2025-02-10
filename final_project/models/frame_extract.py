import os
import cv2
from   time                    import time
from   models.angle_similarity import AngleSimilarity
from   models.clip_similarity  import ClipVideoProcessor

class FrameExtractor:
    def __init__(self, 
                 video_dir1        : str, 
                 video_dir2        : str,
                 npz_file          : str,
                 output_dir        : str = "extracted_frames",
                 top_k             : int = 5, 
                 sampling_interval : int = 500):
        
        self.video_dir1 = video_dir1
        self.video_dir2 = video_dir2
        self.npz_file = npz_file
        self.output_dir = output_dir
        self.top_k = top_k
        self.sampling_interval = sampling_interval
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.clip_processor = ClipVideoProcessor()
    
    def __enter__(self):
        self.start_time = time()
        print(f"🚀 [FrameExtractor] 시작됨...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time() - self.start_time
        print(f"⏳ [FrameExtractor] 전체 실행 시간: {elapsed_time:.2f}초")
    
    @staticmethod
    def parse_timestamp_key(ts_key):

        parts = ts_key.split("_")
        if len(parts) == 2:
            try:
                start = float(parts[0])
                end = float(parts[1])
                return start, end
            
            except ValueError:
                pass
        try:
            ts = float(ts_key)
            return ts, ts + 2000
        
        except ValueError:
            return 0, 2000
    
    def extract_frames(self, 
                       input_text : str):
        
        angle_model  = AngleSimilarity(input_text, 
                                       self.npz_file, 
                                       self.top_k)
        top_results  = angle_model.results
        saved_frames = []
        
        for idx, (video_id, ts_key, angle_sim) in enumerate(top_results, start=1):
            start_timestamp, end_timestamp = self.parse_timestamp_key(ts_key)
            video_file     = self.clip_processor.find_video_file_by_movie_id(self.video_dir1, 
                                                                             video_id)
            
            if video_file is None:
                video_file = self.clip_processor.find_video_file_by_movie_id(self.video_dir2, 
                                                                             video_id)
                
            if video_file is None:
                print(f"Error: 영상 파일을 찾을 수 없습니다. Movie ID: {video_id}")
                continue
            
            best_frame, best_time, clip_sim = self.clip_processor.find_best_frame_in_interval(video_file, 
                                                                                              start_timestamp, 
                                                                                              end_timestamp, 
                                                                                              input_text, 
                                                                                              self.sampling_interval)
            
            if best_frame is None:
                print(f"Movie ID: {video_id} 구간 ({start_timestamp}ms ~ {end_timestamp}ms)에서 프레임 추출 실패")
                continue
            
            output_frame_path = os.path.join(self.output_dir, f"extracted_frame_{idx}.jpg")
            
            cv2.imwrite(output_frame_path, best_frame)
            
            saved_frames.append({
                                 "movie_id"         : video_id,
                                 "time_range"       : f"{start_timestamp/1000:.1f} ~ {end_timestamp/1000:.1f}초",
                                 "best_time"        : best_time,
                                 "angle_similarity" : angle_sim,
                                 "output_frame_path": output_frame_path
                                })
            
        return saved_frames


