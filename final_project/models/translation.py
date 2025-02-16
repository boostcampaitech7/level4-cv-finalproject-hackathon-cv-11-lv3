from   deep_translator import  GoogleTranslator
from   transformers    import (AutoTokenizer,
                               AutoModelForCausalLM,
                               AutoModelForSeq2SeqLM)
from   time            import  time
import gc
import torch

KO_EN_MODEL_NAME =  "Helsinki-NLP/opus-mt-ko-en"
EN_KO_MODEL_NAME =  "nayohan/llama3-instrucTrans-enko-8b"

class Translator:
    def __init__(self,
                 kr2en      : bool = False, 
                 mode       : str  = "API",
                 model_name : str  = KO_EN_MODEL_NAME):
        
        self.kr2en      = kr2en
        self.mode       = mode
        self.prompt     = "당신은 번역기입니다. 한국어를 영어로 번역하세요" if kr2en else "당신은 번역기입니다. 영어를 한국어로 번역하세요"
        self.model_name = KO_EN_MODEL_NAME if self.kr2en else EN_KO_MODEL_NAME
        self.model      = None
        self.tokenizer  = None
    
    def __enter__(self):
        self.start_time = time()
        print(f"🚀 [Translator] 시작됨...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time() - self.start_time
        print(f"⏳ [Translator] 전체 실행 시간: {elapsed_time}초")
    
    def __load_model(self):
        if self.model is None or self.tokenizer is None:
            if self.kr2en:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model     = AutoModelForSeq2SeqLM.from_pretrained(self.model_name,
                                                                       device_map  = "cuda",
                                                                       torch_dtype = torch.bfloat16)            
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model     = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                      device_map  = "cuda",
                                                                      torch_dtype = torch.bfloat16)            
            

    def __unload_model(self):
        if self.model is not None:
            torch.cuda.synchronize()
            del self.model
            del self.tokenizer
            self.model     = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
    
    def __dl_en2kr_translator(self, response : str):
        self.__load_model()
        
        CONVERSATION     =  [
                             {"role" : "system", "content": self.prompt},
                             {"role" : "user",   "content": response}
                            ]
        
        inputs           = self.tokenizer.apply_chat_template(conversation          = CONVERSATION,
                                                              tokenize              = True,
                                                              add_generation_prompt = True,
                                                              return_tensors        = "pt"
                                                              ).to("cuda")
        result           = self.model.generate(inputs,
                                               max_new_tokens = 4096)

        result_text      = self.tokenizer.decode(result[0][len(inputs[0]):], 
                                                 skip_special_tokens=True)
        
        self.__unload_model()
        
        return result_text
    
    def __dl_kr2en_translator(self, response: str):
        self.__load_model()

        inputs = self.tokenizer(response, 
                                return_tensors= "pt", 
                                padding       = True, 
                                truncation    = True).to("cuda")

        result = self.model.generate(**inputs, max_new_tokens=256)

        result_text = self.tokenizer.decode(result[0], skip_special_tokens = True).strip()

        self.__unload_model()

        return result_text
    
    def __api_translator(self, response : str):
        source = "ko" if self.kr2en else "en"
        target = "en" if self.kr2en else "ko"
            
        translator      = GoogleTranslator(source = source,
                                           target = target)
        
        translator_text = translator.translate(text = response)
        return translator_text
    
    def translate(self, 
                  response : str):
        if self.mode == "API":
            return self.__api_translator(response = response)
        elif self.mode == "DL":
            if self.kr2en:
                return self.__dl_kr2en_translator(response = response)
            else:
                return self.__dl_en2kr_translator(response = response)
        else:
            raise ValueError("지원되지 않는 모드입니다. 'API' 혹은 'DL'을 선택하세요")
    