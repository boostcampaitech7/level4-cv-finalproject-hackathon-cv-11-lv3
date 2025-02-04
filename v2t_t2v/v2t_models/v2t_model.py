import torch
import gc
from   video_utils  import (load_video, 
                            extract_audio_np)
from   transformers import (AutoTokenizer,
                            AutoModel)
from   audio_model  import  transcribe_audio

LORA_MODEL_PATH  = "/data/ephemeral/home/lora_epoch/checkpoint-2456"
BASE_MODEL_PATH  = "OpenGVLab/InternVL2_5-1B-MPO"
VIDEO_PATH       = "/data/ephemeral/home/videos_movieclips_461/1ESVngpn1aA.mp4"
NUM_SEGMENTS     = 32

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

GENERATION_CONFIG = {
                    "max_new_tokens": 512,
                    "do_sample": False,
                    "num_beams": 3,
                    "early_stopping": True,     
                    "no_repeat_ngram_size": 3,  
                    "length_penalty": 1.0,
                    }

def analyze_video(video_path = VIDEO_PATH, prompt = PROMPT):
    model              = AutoModel.from_pretrained(LORA_MODEL_PATH,
                                                   torch_dtype       = torch.bfloat16,
                                                   low_cpu_mem_usage = True,
                                                   use_flash_attn    = False,
                                                   trust_remote_code = False
                                                   ).eval().cuda()
    
    tokenizer          = AutoTokenizer.from_pretrained(BASE_MODEL_PATH,
                                                       trust_remote_code = False,
                                                       use_fast          = False)
    
    pix_val, timestamp = load_video(video_path  = video_path,
                                    num_segment = NUM_SEGMENTS
                                    ).to(torch.bfloat16).cuda()
    
    audio_np           = extract_audio_np(video_path)
    transcript         = transcribe_audio(audio_np)
    
    if transcript.strip():
        prompt += f"\nAdditionally, the conversation in the video includes the following dialogue: \"{transcript}\""
    
    video_prefix       = "".join([
                                  f'Frame{i+1} ({timestamp[i]}): <image>\n'
                                  for i in range(pix_val.size(0)) 
                                  ])
    
    query              = video_prefix + prompt
    response, history  = model.chat(tokenizer,
                                    pix_val,
                                    query,
                                    GENERATION_CONFIG,
                                    history        = None,
                                    return_history = True)
    gc.collect()
    torch.cuda.empty_cache()
    del model
    del tokenizer
    return response