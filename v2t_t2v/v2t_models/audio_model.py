import torch
import numpy as np
import gc
from   transformers import (Wav2Vec2ForCTC,
                            Wav2Vec2Processor)

AUDIO_MODEL = "facebook/wav2vec2-large-960h-lv60-self"

def transcribe_audio(audio_np, sr = 16000):
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