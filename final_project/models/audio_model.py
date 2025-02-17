import torch
import gc
import os
import tempfile
import librosa
import speech_recognition as     sr
import numpy              as     np
from   time               import time
from   transformers       import (Wav2Vec2ForCTC, 
                                  Wav2Vec2Processor)
from   moviepy            import  VideoFileClip

AUDIO_MODEL = "facebook/wav2vec2-large-960h-lv60-self"

class AudioExtractor:
    def __init__(self, 
                 video_path : str, 
                 sr         : int = 16000, 
                 mode       : str = "DL"):
        self.video_path = video_path
        self.sr         = sr
        self.mode       = mode
        self.model      = None
        self.processor  = None

    def __enter__(self):
        self.start_time = time()
        print(f"🚀 [AudioExtractor] 시작됨...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time() - self.start_time
        print(f"⏳ [AudioExtractor] 전체 실행 시간: {elapsed_time:.2f}초")

    def __load_model(self):
        if self.model is None or self.processor is None:
            self.model = Wav2Vec2ForCTC.from_pretrained(AUDIO_MODEL).to("cuda")
            self.processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL)

    def __unload_model(self):
        if self.model is not None:
            torch.cuda.synchronize()
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            torch.cuda.empty_cache()

    def __extract_audio_np(self):
        try:
            print("🚀 [DEBUG] 비디오에서 오디오 추출 중...")
            clip        = VideoFileClip(self.video_path)
            audio_clip  = clip.audio
            audio_array = audio_clip.to_soundarray()
            clip.close()

            if audio_array.ndim == 2:
                audio_array = np.mean(audio_array, axis=1)

            audio_array = librosa.resample(audio_array, orig_sr=audio_clip.fps, target_sr=self.sr)

            audio_array = librosa.util.normalize(audio_array) * 0.95
            print("✅ [DEBUG] 오디오 변환 완료!")
            return audio_array
        except Exception as e:
            raise RuntimeError(f"오디오 추출 실패: {e}")

    def __make_dl_transcription(self):
        self.__load_model()

        audio_np     = self.__extract_audio_np()
        input_values = self.processor(audio_np, sampling_rate=self.sr, return_tensors="pt").input_values.to("cuda")

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.tokenizer.decode(predicted_ids[0].tolist())
        print(transcription)
        self.__unload_model()
        return transcription
    
    def __extract_auido(self):
        with tempfile.NamedTemporaryFile(suffix = ".wav", delete = False) as temp_wav:
            temp_wav_path = temp_wav.name
            
        clip = VideoFileClip(self.video_path)
        
        clip.audio.write_audiofile(temp_wav_path, codec="pcm_s16le")
        clip.audio.close()
        return temp_wav_path
    
    def __make_api_transcription(self):
        recognizer     = sr.Recognizer()
        temp_file_path = self.__extract_auido()
        
        with sr.AudioFile(temp_file_path) as source:
            audio_data = recognizer.record(source)
            try:
                transcription = recognizer.recognize_google(audio_data)
                return transcription
            
            except sr.UnknownValueError:
                return "음성을 인식할 수 없습니다."
            
            except sr.RequestError:
                return "Google STT API에 접근할 수 없습니다."
            
        os.remove(temp_file_path)
        return transcription
    
    def transcript(self):
        if self.mode == "DL":
            return self.__make_dl_transcription()
        
        elif self.mode == "API":
            return self.__make_api_transcription()
        
        else:
            raise ValueError("지원되지 않는 모드입니다. 'API' 혹은 'DL'을 선택하세요")
