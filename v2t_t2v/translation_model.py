import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM
import gc

PROMPT           =  "당신은 번역기입니다. 영어를 한국어로 자연스럽게 번역하세요"
MODEL_NAME       =  "nayohan/llama3-instrucTrans-enko-8b"

def translation(response):
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
    
    