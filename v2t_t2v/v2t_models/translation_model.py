from   deep_translator import  GoogleTranslator
from   transformers    import (AutoTokenizer,
                               AutoModelForCausalLM)
import gc
import torch

EN2KR_PROMPT     =  "당신은 번역기입니다. 영어를 한국어로 문맥에 맞게 자연스럽게 번역하세요"
KR2EN_PROMPT     =  "당신은 번역기입니다. 한국어를 영어로 문맥에 맞게 자연스럽게 번역하세요"
MODEL_NAME       =  "nayohan/llama3-instrucTrans-enko-8b"

def DL_translation(response : str, 
                   kr2en    : bool = False):
    
    if kr2en:
        PROMPT   = KR2EN_PROMPT
    else:
        PROMPT   = EN2KR_PROMPT
        
    tokenizer    =         AutoTokenizer.from_pretrained(MODEL_NAME)
    model        =  AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                         device_map  = "auto",
                                                         torch_dtype = torch.bfloat16)
    
    CONVERSATION =  [
                     {"role" : "system", "content": PROMPT},
                     {"role" : "user", "content": response}
                    ]
    
    inputs       =  tokenizer.apply_chat_template(
                             conversation          = CONVERSATION,
                             tokenize              = True,
                             add_generation_prompt = True,
                             return_tensors        = "pt"
                             ).to("cuda")
    
    outputs      =  model.generate(inputs, max_new_tokens = 4096)
    output_text  =  tokenizer.decode(outputs[0][len(inputs[0]):])
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return output_text
    
def API_translator(response : str,
                   kr2en    : bool = False):
    if kr2en:
        source      = "ko"
        target      = "en"
    else:
        source      = "en"
        target      = "ko"
        
    translator      = GoogleTranslator(source = source,
                                       target = target)
    
    translated_text = translator.translate(text = response)
    return translated_text