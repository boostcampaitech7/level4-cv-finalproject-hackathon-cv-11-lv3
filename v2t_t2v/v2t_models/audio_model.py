import torch
import speech_recognition as sr
import numpy              as np
import gc
import os
from   transformers       import (Wav2Vec2ForCTC,
                                  Wav2Vec2Processor)
from   video_utils        import (extract_auido,
                                  time_count)

AUDIO_MODEL = "facebook/wav2vec2-large-960h-lv60-self"

@time_count
def MODEL_transcribe_audio(audio_np, sr = 16000):
    audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL)
    audio_model     = Wav2Vec2ForCTC.from_pretrained(AUDIO_MODEL).to("cuda")
    
    audio_np        = audio_np.astype(np.float32)
    
    input_values    = audio_processor(audio_np,
                                      sampling_rate  = sr,
                                      return_tensors = "pt"
                                      ).input_values.to("cuda")
    
    with torch.no_grad():
        logits = audio_model(input_values).logits
    
    predicted_ids   = torch.argmax(logits,
                                    dim = -1)
    transcription   = audio_processor.tokenizer.decode(predicted_ids[0].tolist())
    
    del audio_model
    del audio_processor
    
    torch.cuda.synchronize()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return transcription

@time_count
def SR_transcribe_audio(video_path : str):
    recognizer     = sr.Recognizer()
    temp_file_path = extract_auido(video_path = video_path)
    
    with sr.AudioFile(temp_file_path) as source:
        audio_data = recognizer.record(source)  # 오디오 파일 로드
        try:
            transcription = recognizer.recognize_google(audio_data)  # Google STT
            return transcription
        except sr.UnknownValueError:
            return "음성을 인식할 수 없습니다."
        except sr.RequestError:
            return "Google STT API에 접근할 수 없습니다."
    
    os.remove(temp_file_path)
    return transcription