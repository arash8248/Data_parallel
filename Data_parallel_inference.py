
"""
@author: Arash
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor

def inference(prompt, tokenizer, model, device, maxnewtokens):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)    
    with torch.no_grad():
        outputs = model.generate(inputs,max_new_tokens=maxnewtokens)               
    return outputs

def process_prompt(args):
    prompt, model, device, input_idx, maxnewtokens, tokenizer = args    
    print(input_idx)
    output = inference(prompt, tokenizer, model, device, maxnewtokens)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    return input_idx, output

if __name__ == "__main__":
    
    # Prepare the prompts. Provide the prompt body in 'note' and index of each prompt (or file name if you are reading the prompts from a directory in 'label').
    # The label will be used to make the correspondence between inputs and outputs
    # So, the input could also be as:
        #TEXTS = {'note':['What is the capital of france?','Which team won the 1990 work cup?','What is happiness?','How can I fly an airplane?'],'label':['1.txt','2.txt','3.txt','4.txt']}
    TEXTS = {'note':['What is the capital of france?','Which team won the 1990 work cup?','What is happiness?','How can I fly an airplane?'],'label':[1,2,3,4]}
    
    maxnewtokens = 200              
    model_name = "Arash8248/Mistral-7B-Instruct-v0.3-4bit-GPTQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    # Copy the model on each GPU
    models = [AutoModelForCausalLM.from_pretrained(model_name).to(device) for device in devices]    
           
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        # Schedule the parallel data processing
        futures = [executor.submit(process_prompt, (prompt, models[int(device.split(':')[1])], device, idx, maxnewtokens, tokenizer))\
                   for idx, (prompt, device) in enumerate(zip(TEXTS['note'], devices * (len(TEXTS['note']) // len(devices)) + devices[:len(TEXTS['note']) % len(devices)]))]
        # Collect the data
        outputs_associated = {'label':[],'Result':[]}
        outputs_associated['Result'] = [None] * len(TEXTS['note'])
        outputs_associated['label'] = [None] * len(TEXTS['note'])
        for future in futures:
                input_idx, output = future.result()                
                outputs_associated.loc[input_idx,'Result'] = tokenizer.decode(output[0][-maxnewtokens:])
                outputs_associated.loc[input_idx,'label'] = TEXTS.loc[input_idx,'label']
    
