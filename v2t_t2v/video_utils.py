from torchvision import transforms as T
from decord      import VideoReader, cpu
from moviepy     import VideoFileClip
from PIL         import Image
import torch
import numpy     as np

IMAGENET_MEAN       = (0.485, 0.456, 0.406)
IMAGENET_STD        = (0.229, 0.224, 0.225)

def time_to_seconds(time_str):
    try:
        hh, mm, ss, ff = map(int, time_str.split(':'))
        return hh * 3600 + mm * 60 + ss + ff / 30
    except ValueError:
        raise ValueError("Time format should be HH:MM:SS:FF")

def extract_audio_np(video_path, 
                     target_sr=16000):
    
    clip            =  VideoFileClip(video_path)
    audio_clip      =  clip.audio
    audio_array     =  audio_clip.to_soundarray(fps=target_sr)
    clip.close()
    
    if audio_array.ndim == 2:
        audio_array =  np.mean(audio_array, axis=1)
        
    return audio_array, target_sr

def build_transform(input_size):
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

def load_video(video_path, 
               bound       = None, 
               input_size  = 448,  
               num_segment = 32):
    
    vr                = VideoReader(video_path, 
                                    ctx         = cpu(0), 
                                    num_threads = 1,
                                    num_segment = num_segment)
    max_frame         = len(vr) - 1
    fps               = float(vr.get_avg_fps())
    
    pixel_values_list = []
    timestamp         = []
    
    transform         = build_transform(input_size)
    frame_indices     = get_index(bound        = bound, 
                                  fps          = fps, 
                                  max_frame    = max_frame, 
                                  first_idx    = 0, 
                                  num_segments = 32)
    
    for frame_idx in frame_indices:
        img           = Image.fromarray(
                        vr[frame_idx].asnumpy()
                        ).convert("RGB")
        
        img           = transform(img)
        pixel_values_list.append(img)
        
        seconds       = frame_idx / fps
        minutes       = int(seconds // 60)
        sec           = seconds & 60
        timestamp_str = f"{minutes:02d}:{sec:05.2f}"
        timestamp.append(timestamp_str)
        
    pixel_value       = torch.stack(pixel_values_list)
    return pixel_value, timestamp