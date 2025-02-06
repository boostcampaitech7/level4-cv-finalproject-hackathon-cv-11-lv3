from torchvision import transforms as T
from decord      import VideoReader, cpu
from moviepy     import VideoFileClip
from PIL         import Image
from pydub       import AudioSegment
from time        import time
import torch
import numpy     as np
import librosa
import tempfile

IMAGENET_MEAN       = (0.485, 0.456, 0.406)
IMAGENET_STD        = (0.229, 0.224, 0.225)

def time_count(func):
    def wrapper(*arg, **kwargs):
        start_time = time()
        result = func(*arg, **kwargs)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"⏳{func.__name__} 실행 시간: {elapsed_time:.6f}초")
        return result
    return wrapper

def time_to_seconds(time_str : str):
    try:
        hh, mm, ss, ff = map(int, time_str.split(':'))
        return hh * 3600 + mm * 60 + ss + ff / 30
    
    except ValueError:
        raise ValueError("Time format should be HH:MM:SS:FF")

def get_video_duration(video_path : str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_path.read())
        temp_video_path = temp_file.name
    try:
        with VideoFileClip(temp_video_path) as video:
            duration    = video.duration
            return temp_video_path, duration
    except Exception as e:
        print(f"❌ 비디오 정보를 가져오는 중 오류 발생: {e}")
        return None,None

def get_default_times(video_duration):
    start_time   = "00:00:00:00"

    if video_duration:
        hh       = int(video_duration // 3600)
        mm       = int((video_duration % 3600) // 60)
        ss       = int(video_duration % 60)
        end_time = f"{hh:02d}:{mm:02d}:{ss:02d}:00"
    else:
        end_time = "00:00:30:00"

    return start_time, end_time

def cut_video_moviepy(video_path: str, 
                      start_time: int, 
                      end_time: int):
    try:
        print(f"🎬 비디오 파일 처리 중: {video_path}")
        with VideoFileClip(video_path) as video:
            duration       = video.duration
            print(f"📏 비디오 길이: {duration} 초")
            print(f"🎞️ 프레임 속도(FPS): {video.fps}")
            print(f"📐 비디오 해상도: {video.size}")
            
            if start_time >= duration:
                raise ValueError("❌ 시작 시간이 비디오 길이를 초과합니다.")
            
            if end_time > duration:
                print(f"⚠️ 종료 시간이 비디오 길이를 초과했습니다. {duration}초로 조정합니다.")
                end_time   = duration
            
            if start_time >= end_time:
                raise ValueError("❌ 시작 시간이 종료 시간보다 크거나 같습니다.")
            
            subclip = video.subclipped(start_time, end_time)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out:
                output_path = temp_out.name
                print(f"💾 잘라낸 비디오 저장 중: {output_path}")
                
                subclip.write_videofile(output_path,
                                        codec="libx264",
                                        audio_codec="aac")
                
            print("✅ 비디오 잘라내기 완료!")
            return output_path, end_time
    
    except Exception as e:
        raise RuntimeError(f"⚠️ MoviePy 오류 발생: {e}")
    
def extract_audio_np(video_path : str, 
                     target_sr  = 16000):
    
    clip            =  VideoFileClip(video_path)
    audio_clip      =  clip.audio
    audio_array     =  audio_clip.to_soundarray(fps=target_sr)
    clip.close()
    
    if audio_array.ndim == 2:
        audio_array =  np.mean(audio_array, axis=1)
        
    return audio_array, target_sr

def extract_auido(video_path : str):
    with tempfile.NamedTemporaryFile(suffix = ".wav",
                                     delete = False) as temp_wav:
        temp_wav_path = temp_wav.name
        
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(temp_wav_path, codec="pcm_s16le")
    clip.audio.close()
    return temp_wav_path

def build_transform(input_size : int):
    return T.Compose([
           T.Lambda(lambda img : img.convert("RGB") if img.mode != "RGB" else img),
           T.Resize((input_size, input_size), interpolation = T.InterpolationMode.BICUBIC),
           T.ToTensor(),
           T.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
    ])

def get_index(bound        = None,
              fps          = 64, 
              max_frame    = 0, 
              first_idx    = 0, 
              num_segments = 32):
    
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
        
    start_idx      = max(first_idx, round(start * fps))
    end_idx        = min(round(end * fps), max_frame)
    seg_size       = float(end_idx - start_idx) / num_segments
    
    frame_indices  = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    
    return frame_indices

def load_video(video_path  : str, 
               bound       = None, 
               input_size  = 448,  
               num_segment = 32):
    
    vr                = VideoReader(video_path, 
                                    ctx         = cpu(0), 
                                    num_threads = 1)
    max_frame         = len(vr) - 1
    fps               = float(vr.get_avg_fps())
    
    pixel_values_list = []
    timestamp         = []
    
    transform         = build_transform(input_size)
    frame_indices     = get_index(bound        = bound, 
                                  fps          = fps, 
                                  max_frame    = max_frame, 
                                  first_idx    = 0, 
                                  num_segments = num_segment)
    
    for frame_idx in frame_indices:
        img           = Image.fromarray(
                        vr[frame_idx].asnumpy()
                        ).convert("RGB")
        
        img           = transform(img)
        pixel_values_list.append(img)
        
        seconds       = frame_idx / fps
        minutes       = int(seconds // 60)
        sec           = seconds % 60
        timestamp_str = f"{minutes:02d}:{sec:05.2f}"
        timestamp.append(timestamp_str)
        
    pixel_value       = torch.stack(pixel_values_list)
    return pixel_value, timestamp

def extract_audio_np(video : str):
    clip            = VideoFileClip(video)
    audio_clip      = clip.audio
    audio_array     = audio_clip.to_soundarray()
    clip.close()
    
    if audio_array.ndim == 2:
        audio_array = np.mean(audio_array, 
                              axis = 1)

    audio_array     = librosa.resample(audio_array, 
                                       orig_sr   = audio_clip.fps,
                                       target_sr = 16000)
    
    audio_array     = librosa.util.normalize(audio_array) * 0.95
    return audio_array