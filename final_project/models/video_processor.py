import subprocess
import re
import torch
from   decord             import (VideoReader, 
                                  cpu)
import numpy              as      np
from   concurrent.futures import  ThreadPoolExecutor
from   time               import  time

class VideoProcessor:
    def __init__(self,
                 num_seg        = 8,
                 shot_threshold = 0.3):
        self.num_seg        = num_seg
        self.shot_threshold = shot_threshold
        
    def __enter__(self):
        self.start_time = time()
        print(f"🚀 [VideoProcessor] 시작됨...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time() - self.start_time
        print(f"⏳ [VideoProcessor] 전체 실행 시간: {elapsed_time}초")
    
    def __get_video_duration(self, video_path: str):
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return None
    
    def __get_shot_boundaries(self,
                            video_path : str):
        print(f"📸 [시작] 쇼트 추출 중: {video_path}")
        cmd           = [
                         "ffmpeg",
                         "-nostdin",
                         "-i", video_path,
                         "-filter_complex", f"select='gt(scene,{self.shot_threshold})', showinfo",
                         "-f", "null", "-"
                        ]
        
        proc          = subprocess.run(cmd,
                                    stderr = subprocess.PIPE,
                                    stdout = subprocess.PIPE)
        
        stderr_output = proc.stderr.decode("utf-8")
        
        pattern       = re.compile(r"pts_time:(\d+\.\d+)")
        times         = [float(match.group(1)) for match in pattern.finditer(stderr_output)]
        
        video_duration = self.__get_video_duration(video_path)
        if video_duration is None:
            raise ValueError(f"🚨 비디오 길이를 가져올 수 없습니다: {video_path}")
        
        if times and times[0] != 0.0:
            times = [0.0] + times
            
        if not times or times[-1] < video_duration:
            times.append(video_duration)
            
        print(f"✅ [완료] 쇼트 추출 완료: {video_path} (총 {len(times)}개 쇼트)")
        return times
    
    def __extract_frames_decord(self, video_path: str, shot_times: list):
        print(f"📸 [시작] 프레임 추출 중: {video_path}")
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        except Exception as e:
            print(f"🚨 [에러] 비디오 파일을 열 수 없습니다: {video_path}")
            print(f"🔹 [에러 내용] {e}")
            return None, None, None

        total_frames = len(vr)
        fps = vr.get_avg_fps()
        total_duration = total_frames / fps 

        shot_frames = []
        timestamps = []

        for i in range(len(shot_times) - 1):
            start_time = shot_times[i]
            end_time = shot_times[i + 1]

            start_shot = int(start_time * fps)
            end_shot = int(end_time * fps)

            start_shot = min(start_shot, total_frames - 1)
            end_shot = min(end_shot, total_frames - 1)

            indices = np.linspace(start_shot, end_shot, self.num_seg).astype(int)
            indices = np.clip(indices, 0, total_frames - 1) 
            
            frames = vr.get_batch(indices).asnumpy()
            frames = torch.from_numpy(frames).to(torch.float32).div(255.0).cuda()
            frames = frames.permute(0, 3, 1, 2)
            frames = torch.nn.functional.interpolate(frames, size=(448, 448), mode='bicubic', align_corners=False)

            mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
            frames = (frames - mean) / std

            shot_frames.append(frames)
            timestamps.append((start_time, end_time))

        print(f"✅ [완료] 프레임 추출 완료: {video_path} (총 {len(shot_frames)}개 프레임)")
        return shot_frames, timestamps, total_duration
    
    def process_videos(self, 
                       video_paths : list):
        with ThreadPoolExecutor() as executor:
            shot_times_list = list(executor.map(self.__get_shot_boundaries,
                                                video_paths))
            
        shot_frames_list = []
        durations = []
        
        with ThreadPoolExecutor() as executor:
            for video_path, shot_times in zip(video_paths, shot_times_list):
                shot_frames, timestamps, duration = self.__extract_frames_decord(video_path, 
                                                                                 shot_times)
                shot_frames_list.append((shot_frames, timestamps))
                durations.append(duration)
                
        return shot_frames_list, shot_times_list, durations