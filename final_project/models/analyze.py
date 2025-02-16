import  gc
import  os
import  cv2
import  json
import  torch
import  numpy                  as     np
from    tqdm                   import tqdm
from    time                   import time
from    transformers           import (AutoTokenizer,
                                       AutoModel)
from    decord                 import (VideoReader,
                                       cpu)
from    PIL                    import Image
from    torchvision            import transforms      as T
from    models.audio_model     import AudioExtractor
from    models.video_processor import VideoProcessor
    
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

LORA_MODEL_PATH  = "/data/ephemeral/home/checkpoint-2456"
BASE_MODEL_PATH  = "/data/ephemeral/home/InternVL2_5-1B-MPO"

PROMPT           = (
                   "Please carefully watch the following video scene and describe it in as much detail as possible. "
                   "Focus on the following aspects:\n"
                   "1) If there are any people present, provide a thorough description of them (physical appearance, clothing, expressions, etc.), but do not describe them in list form.\n"
                   "2) If there are no people, do NOT apologize or disclaim; instead, describe the environment or objects in thorough detail as part of a flowing narrative.\n"
                   "3) Describe each person's actions or movements in detail: gestures, body language, eye contact, posture, and any physical interactions with objects, ensuring they are seamlessly integrated into the overall description.\n"
                   "4) Provide context for the situation and setting (indoors/outdoors, lighting, time of day, weather, etc.) and explain what is happening, making it part of a unified story.\n"
                   "5) Include any relevant objects, environmental details, or interactions that add meaning. For example, mention notable items and how they are positioned or used, incorporating them into the broader narrative.\n"
                   "6) Avoid repetitive phrases like 'the scene...' at the beginning of every sentence. Instead, aim for a natural flow of description.\n"
                   "7) Be thorough and vivid, but clarify if certain details are assumptions rather than visible facts.\n"
                   "8) Finally, combine all details into a single, cohesive, and natural narrative. Avoid listing information as separate points and ensure that the description reads like a story or article."
                   )

AUDIO_PROMPT = (
                "As an expert in video scene analysis, your task is to generate a vivid, coherent narrative that seamlessly integrates both the visual elements of the scene and the provided dialogue. "
                "Follow these detailed instructions:\n\n"
                "1. **Context Extraction**: Carefully analyze the dialogue transcript to capture the underlying emotions, intentions, and narrative cues. Identify nuances in tone, mood, and context that may affect the overall scene.\n\n"
                "2. **Visual Alignment**: Use the insights obtained from the dialogue to enhance the visual description. Integrate these audio cues with the visual guidelines provided in the main prompt—describing people, actions, gestures, the environment, and objects in detail.\n\n"
                "3. **Coherent Narrative Formation**: Synthesize the auditory and visual information into a single, flowing narrative. Ensure that the description is cohesive, avoiding disjointed lists or repetitive structures, and that the dialogue seamlessly informs the narrative.\n\n"
                "4. **Modern Detailing Techniques**: Apply advanced narrative strategies such as chain-of-thought reasoning and context-aware inference. When certain details are not explicitly visible, make reasonable assumptions while clearly distinguishing between observed facts and inferred details.\n\n"
                "5. **Dialogue Integration**: Incorporate the provided dialogue naturally within your narrative. The dialogue should enhance and complement the visual details rather than stand alone.\n\n"
                "Dialogue Transcript: '{transcript}'"
               )

GENERATION_CONFIG = {
                    "max_new_tokens": 512,
                    "do_sample": False,
                    "num_beams": 2,
                    "early_stopping": True,     
                    "no_repeat_ngram_size": 3,  
                    "length_penalty": 1.0,
                    }

class AnalyzeVideo:
    def __init__(self, 
                 use_audio  : bool = True,
                 num_seg    : int  = 32,
                 input_size : int  = 448,
                 batch_size : int  = 4):
        self.use_audio  = use_audio
        self.num_seg    = num_seg
        self.input_size = input_size
        self.batch_size = batch_size
        self.model      = None
        self.tokenizer  = None
        
    def __enter__(self):
        self.start_time = time()
        print(f"🚀 [AnalyzeVideo] 시작됨...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time() - self.start_time
        print(f"⏳ [AnalyzeVideo] 전체 실행 시간: {elapsed_time}초")
    
    def __load_model(self):
        if self.model is None or self.tokenizer is None:
            self.model     = AutoModel.from_pretrained(LORA_MODEL_PATH,
                                                       torch_dtype       = torch.bfloat16,
                                                       low_cpu_mem_usage = True,
                                                       use_flash_attn    = False,
                                                       trust_remote_code = True,
                                                       ).eval().to("cuda")
            self.model     = torch.compile(self.model)
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH,
                                                           trust_remote_code = True,
                                                           use_fast          = True)
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
    
    def __unload_model(self):
        if self.model is not None:
            torch.cuda.synchronize()
            del self.model
            del self.tokenizer
            self.model     = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
    
    def __build_transform(self):
        return T.Compose([
               T.Resize((self.input_size, self.input_size),
                         interpolation = T.InterpolationMode.BICUBIC),
               T.ToTensor(),
               T.Normalize(mean = IMAGENET_MEAN,
                           std  = IMAGENET_STD)
               ])
        
    def __get_index(self,
                  bound     = None,
                  fps       : int = 64,
                  max_frame : int = 0,
                  first_idx : int = 0):
        
        start, end = bound if bound else (-100000, 100000)
        
        start_idx  = max(first_idx, round(start * fps))
        end_idx    = min(round(end * fps), max_frame)
        seg_size   = (end_idx - start_idx) / self.num_seg
        
        frame_indice = np.linspace(start_idx + (seg_size / 2),
                                   end_idx - (seg_size / 2),
                                   self.num_seg,
                                   dtype = int)
        return frame_indice
        
    def __load_video(self, video_path : str):
        vr                = VideoReader(video_path,
                                        ctx         = cpu(0), 
                                        num_threads = 1)
        
        max_frame         = len(vr) - 1
        fps               = float(vr.get_avg_fps())
            
        transform         = self.__build_transform()
        frame_indices     = self.__get_index(fps = fps, max_frame = max_frame)
        frames            = vr.get_batch(frame_indices).asnumpy()
        
        pixel_values_list = []
        timestamp         = []
        
        for i, frame_np in enumerate(frames):
            img     = Image.fromarray(frame_np)
            img     = transform(img)
            pixel_values_list.append(img)
            
            seconds = frame_indices[i] / fps
            timestamp.append(f"{int(seconds // 60):02d}:{seconds % 60:05.2f}")
        pixel_value = torch.stack(pixel_values_list)
        return pixel_value, timestamp
    
    def __load_video_gpu(self, video_path : str):
        vr = VideoReader(video_path,
                         ctx         = cpu(0), 
                         num_threads = 1)
        
        max_frame = len(vr) - 1
        
        fps = float(vr.get_avg_fps())
        
        start_idx = 0
        end_idx = max_frame
        seg_size = (end_idx - start_idx) / self.num_seg
        
        frame_indices = np.linspace(start_idx + seg_size / 2,
                                    end_idx - seg_size / 2,
                                    self.num_seg,
                                    dtype=int)
        
        frames_np = vr.get_batch(frame_indices).asnumpy()
        
        frames = torch.from_numpy(frames_np).to(torch.bfloat16).div(255.0)
        
        frames = frames.permute(0, 3, 1, 2).cuda()
        
        frames = torch.nn.functional.interpolate(frames, 
                                                 size=(self.input_size, self.input_size), 
                                                 mode='bicubic', 
                                                 align_corners=False)
        
        mean = torch.tensor(IMAGENET_MEAN, device=frames.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=frames.device).view(1, 3, 1, 1)
        
        frames = (frames - mean) / std

        timestamps = []
        
        for idx in frame_indices:
            seconds = idx / fps
            timestamps.append(f"{int(seconds // 60):02d}:{seconds % 60:05.2f}")
        
        return frames, timestamps
        
    def __transcribe_audio(self, video_path : str):
        with AudioExtractor(video_path = video_path, mode = "DL") as extractor:
            transcript = extractor.transcript()
            
        return transcript.strip()
    
    def __extract_segments_frames(self, 
                                  video_path         : str, 
                                  segments           : int = 13, 
                                  frames_per_segment : int = 3, 
                                  input_size         : int = 448):
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)

        transform = T.Compose([
                    T.Resize((input_size, input_size),
                              interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
                    ])

        segments_frames     = []
        segments_timestamps = []

        for seg_idx in range(segments):
            seg_start = int(total_frames * seg_idx / segments)
            seg_end   = int(total_frames * (seg_idx + 1) / segments)
            if seg_end <= seg_start:
                continue

            frame_indices = np.linspace(seg_start, seg_end - 1, frames_per_segment, dtype=int)
            
            frames_list   = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame   = cap.read()
                
                if not ret:
                    continue
                
                frame        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil    = Image.fromarray(frame)
                frame_tensor = transform(frame_pil)
                frames_list.append(frame_tensor)

            if len(frames_list) != frames_per_segment:
                continue

            segment_tensor = torch.stack(frames_list, dim=0)
            segments_frames.append(segment_tensor)
            segments_timestamps.append([int((seg_start / fps) * 1000), int((seg_end / fps) * 1000)])

        cap.release()
        return segments_frames, segments_timestamps, int(total_frames / fps * 1000)

    def analyze(self,
                video_path,
                prompt = PROMPT):
        
        self.__load_model()
        
        pix_val, timestamp = self.__load_video_gpu(video_path=video_path)
        
        pix_val            = pix_val.to(torch.bfloat16).cuda()
        
        if self.use_audio:
            transcript   = self.__transcribe_audio(video_path = video_path)
            audio_prompt = AUDIO_PROMPT.format(transcript = transcript)
            prompt      += audio_prompt
        
        video_prefix = f"The following frames from the video are provided: {pix_val.size(0)}"
        query        = video_prefix + prompt
        
        response, _  = self.model.chat(self.tokenizer,
                                       pix_val,
                                       query,
                                       GENERATION_CONFIG,
                                       history        = None,
                                       return_history = True)

        self.__unload_model()
        return response
    
    def batch_analyze(self,
                      video_paths : list,
                      output_path : str,
                      prompt      : str = PROMPT):
        
        self.__load_model()
        
        with VideoProcessor(num_seg = self.num_seg, shot_threshold = 0.3,) as vp:
            shot_frames_list, shot_times_list, durations = vp.process_videos(video_paths = video_paths)
        
        for vid_idx, ((shot_frames, shot_times), duration) in enumerate(zip(shot_frames_list, durations)):
            video_path = video_paths[vid_idx]
            video_id   = os.path.basename(video_path)[-15:-4]
            
            response   = []
            
            for i in tqdm(range(0, len(shot_frames), self.batch_size), desc=f"Processing {video_id}", unit="batch"):
                mini_batch_frame = shot_frames[i : i + self.batch_size]
                pix_values = torch.cat(mini_batch_frame).to(torch.bfloat16).cuda()
                
                batch_prompts = [prompt] * len(mini_batch_frame)
                num_patches_list = [self.num_seg] * len(mini_batch_frame)
                
                batch_response = self.model.batch_chat(self.tokenizer,
                                                           pix_values,
                                                           batch_prompts,
                                                           GENERATION_CONFIG,
                                                           num_patches_list = num_patches_list,
                                                           history = None)
                response.extend(batch_response)
            
            json_data = {
                video_id: {
                    "duration": int(duration * 1000) if shot_times else 0,
                    "timestamps": [[int(s * 1000), int(e * 1000)] for s, e in shot_times],
                    "sentences": response
                }
            }
            
            os.makedirs(output_path, exist_ok=True)
            json_path = os.path.join(output_path, f"{video_id}.json")
            
            with open(json_path, 'w', encoding="utf-8") as json_file:
                json.dump(json_data, json_file, indent=4, ensure_ascii=False)
                
            print(f"✅ JSON saved: {json_path}")
            
        self.__unload_model()
        return response

    def fast_batch_analyze(self,
                  video_paths: list,
                  output_path: str,
                  prompt: str = PROMPT):
        self.__load_model()

        shot_frames_list = []
        shot_times_list  = []
        durations_list   = []

        for video_path in video_paths:
            frames, timestamps, duration = self.__extract_segments_frames(video_path=video_path)
            shot_frames_list.append(frames)
            shot_times_list.append(timestamps)
            durations_list.append(duration)

        for vid_idx, ((shot_frames, shot_times), duration) in enumerate(zip(zip(shot_frames_list, shot_times_list), durations_list)):
            video_path = video_paths[vid_idx]
            
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            
            if len(base_name) < 20:
                video_id = base_name
            else:
                video_id = base_name[-15:-4]

            response   = []

            for i in tqdm(range(0, len(shot_frames), self.batch_size), desc=f"Processing {video_id}", unit="batch"):
                mini_batch_frame = shot_frames[i : i + self.batch_size]
                pix_values       = torch.cat(mini_batch_frame).to(torch.bfloat16).cuda()

                batch_prompts    = [prompt] * len(mini_batch_frame)
                num_patches_list = [self.num_seg] * len(mini_batch_frame)

                batch_response   = self.model.batch_chat(self.tokenizer,
                                                         pix_values,
                                                         batch_prompts,
                                                         GENERATION_CONFIG,
                                                         num_patches_list=num_patches_list,
                                                         history=None)
                response.extend(batch_response)

            json_data = {
                video_id: {
                    "duration": duration,
                    "timestamps": shot_times,
                    "sentences": response
                }
            }

            os.makedirs(output_path, exist_ok=True)
            json_path = os.path.join(output_path, f"{video_id}.json")

            with open(json_path, 'w', encoding="utf-8") as json_file:
                json.dump(json_data, json_file, indent=4, ensure_ascii=False)

            print(f"✅ JSON saved: {json_path}")

        self.__unload_model()
        return response