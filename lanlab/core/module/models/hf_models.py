import torch
from flask import Flask, request
import logging
#import torch.multiprocessing as mp
import multiprocess as mp
#from pathos.helpers import mp
import numpy as np
import os
import time

from lanlab.core.module.models.model import Model
from lanlab.core.module.models.openai_models import segment_from_OPENAICompletion
from lanlab.core.structure.sequence import Sequence
from lanlab.core.module.models.model import ModelConfig
import requests

#Set the path to a folder containing the HF models
HFMODELSPATH = "C:\\Users\\nicol\\git\\LLMs"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def default_config():
    """ Default config for the model generation"""
    return {'prompt':None,
            'temperature':0.7,
            'min_tokens':0,
            'max_tokens':8,
            'logprobs':5,
            'stop':[],
            'echo':False,
            'return_logits':False}

def flask_server(port,inp_queue):
    """ Starts a flask server that will handle the requests"""
    app = Flask(__name__)

    @app.route("/completions", methods=['POST'])
    def completions():
        """ Flask route for the completions"""
        global server
        logging.debug('got request')
        config = default_config()
        for k in config:
            try:
                config[k] = request.json[k]
            except KeyError:
                pass
        logging.debug('parsed request')
        parent_conn,child_conn = mp.Pipe()
        inp_queue.put({"config":config,"pipe":child_conn})
        logging.debug('waiting for the model')
        out = parent_conn.recv()#server.ret_queue.get()
        logging.debug('returns')
        return out
    
    app.run(host='0.0.0.0',port=port,threaded=True)
    
class Server:
    """ Server that handles the requests"""
    def __init__(self,model_loader,port):
        self.port = port
        
        self.model_loader = model_loader
        
        self.active = False

        self.batch_size = 64
        self.timeout = 0.5 #In seconds

    def start(self):
        """ Starts the server and creates the process that will handle the requests"""
        self.inp_queue = mp.Queue()
        self.ret_queue = mp.Queue()
        
        
        logging.info('starting flask server')
        self.flask = mp.Process(target=flask_server,args=(self.port,self.inp_queue))
        self.flask.start()
        
        logging.info('starting model hosting process')
        self.process = mp.Process(target=completion_loop,args=(self.model_loader,self.inp_queue,self.ret_queue,self.timeout,self.batch_size))
        self.process.start()
        self.active = True

    def stop(self):
        self.inp_queue.close()
        self.ret_queue.close()
        self.flask.terminate()
        self.process.terminate()
        self.active = False
        
    def __enter__(self):
        if not(self.active):
            self.start()
        
    def __exit__(self,*args,**kwargs):
        if self.active:
            self.stop()

class HFModel:
    pass
        
class HFModelLister(type):
    """ References all HF models where created """
    # This list will store subclasses of A
    subclasses = []

    def __new__(cls, name, bases, dct):
        # Create the new class
        new_class = super().__new__(cls, name, bases, dct)
        # Check if it's a subclass of A (but not A itself)
        if cls._is_subclass_of_HFModel(bases):
            cls.subclasses.append(new_class)
        return new_class
    
    @classmethod
    def _is_subclass_of_HFModel(cls, bases):
        for base in bases:
            # Check for subclasses of A
            if issubclass(base, HFModel) and hasattr(base,'name') and hasattr(base,'engine'):
                return True
        return False

class HFModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.add_key('return_logits',False) #HF models can return logits

class HFModel(Model,metaclass=HFModelLister):
    def __init__(self):
        super().__init__()
        
    @property
    def engine(self):
        return None
    
    @property
    def name(self):
        return None
        
    @property
    def timeout(self):
        return None #No timeout by default for self hosted models

    @property
    def name(self):
        raise NotImplementedError
        
    def host(self,port=None):
        if port is None:
            port = np.random.randint(48750,58750)
        self.port = port
        server = Server(self.init_model,port=port)
        return server

    @property
    def config_class(self):
        return HFModelConfig
    
    def complete(self,sequence,config=None):
        if config is None:
            config = self.config
        prompt = sequence.format(type='completion')
        data = {'prompt':prompt,'logprobs':5,**config.to_dict()}
        answer = requests.post('http://127.0.0.1:'+str(self.port)+'/completions',json=data).json()
        answer['model'] = self.name
        if self['return_logits']:
            answer['logits'] = np.array(answer['choices'][0]['logprobs']['logits'],np.float16)
        segment = segment_from_OPENAICompletion(answer)
        return sequence+segment

    def read(self,sequence,config=None):
        if config is None:
            config = self.config
        prompt = sequence.format(type='completion')
        config['max_tokens'] = 0
        data = {'prompt':prompt,'logprobs':5,'echo':True,**config.to_dict()}
        answer = requests.post('http://127.0.0.1:'+str(self.port)+'/completions',json=data).json()
        answer['model'] = self.name
        segment = segment_from_OPENAICompletion(answer)
        return Sequence(l=[segment])
    
    def init_kv(self,inputs,model):
        """ Initialize the kv dict. Most of the time it is empty but for some models they use custom kv dicts that need to be intialized """
        return None
    
    def init_model(self):
        """ Initialize the model and put the init kv function """ 
        tokenizer,model = self.load_model()
        print('model dtype',model.dtype)
        model.init_kv = self.init_kv #TODO : improve this point as if model already has a init_kv it will conflict
        return tokenizer,model
    

def completion_loop(model_loader,inp_queue,out_queue,timeout,batch_size):
    """ Loop that handles the requests and sends them to the model"""
    logging.info('starting completion loop')
    def load_model():
        logging.info('loading model')
        tokenizer,model = model_loader()
        logging.info('model loaded')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        return model,tokenizer

    last_time = time.perf_counter()
    batches = {}
    
    model,tokenizer = None,None #Lazy loading
    
    def flush(batch,temperature,max_tokens,model,tokenizer):
        """Flushes the batch and sends it to the model and resets the timer"""
        if len(batch)>=1:
            if model is None:
                model,tokenizer = load_model() #Load the model and the tokenizer
            logging.debug('flush batch of size '+str(len(batch)))
            complete(tokenizer,model,batch,temperature,max_tokens)
            logging.debug('completed batch')
        return model,tokenizer

    while True:
        #Get the first item in the queue
        logging.debug('completion loop iter')
        if not(inp_queue.empty()):
            logging.debug('queue not empty')
            top = inp_queue.get(timeout=1)
            logging.debug('got from queue')
            if isinstance(top,dict):
                logging.debug('got completion order')
                #get temperature
                temp = top['config']['temperature']
                max_tokens = top['config']['max_tokens']
                key = (temp,max_tokens)
                if not(key in batches):
                    batches[key] = [top]
                else:
                    batches[key].append(top)
                if len(batches[key])>=batch_size:
                    model,tokenizer = flush(batches[key],temp,max_tokens,model,tokenizer)
                    del batches[key]
                    last_time = time.perf_counter()
        else:
            if time.perf_counter()-last_time>timeout:
                if len(batches)>0:
                    key = list(batches.keys())[0]
                    temp,max_tokens = key
                    model,tokenizer = flush(batches[key],temp,max_tokens,model,tokenizer)
                    del batches[key]
                last_time = time.perf_counter()
            time.sleep(timeout/10)
    
def complete(tokenizer,model,data,temperature,max_tokens):
    """Completes the queries given in data and returns the results with the OPENAI format"""
    configs,pipes = [data_['config'] for data_ in data],[data_['pipe'] for data_ in data]
    prompts = [config['prompt'] for config in configs]
    #Prepare padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    inp_batch = tokenizer(prompts,return_tensors='pt',padding=True)
    
    #temperature = [config['temperature'] for config in configs]
    #min_tokens = [config['min_tokens'] for config in configs]
    #max_tokens = [config['max_tokens'] for config in configs]
    #stop = [config['stop'] if not(config['stop'] is None) else [] for config in configs]
    echo = [config['echo'] for config in configs]
    nb_logprobs = [config['logprobs'] for config in configs]
    return_logits = [config['return_logits'] for config in configs]
    
    #generate the completion with the required parameters
    with torch.no_grad():
        token_ids,logits = generate(model,tokenizer,inp_batch,max_tokens,temperature=temperature)
    results = [dict_to_openai(token_ids[i],logits[i],tokenizer,temperature,return_logits=return_logits[i],nb_logprobs=nb_logprobs[i],inputs_to_remove=inp_batch.input_ids[i] if not echo[i] else None) for i in range(len(configs))]
    for r,p in zip(results,pipes):
        p.send(r)
        p.close()
    
#tokenize the text "Test" with the tokenizer and return the tokens
def tokenize(tokenizer,text):
    """Tokenizes the text with the tokenizer and returns the tokens"""
    return tokenizer(text,return_tensors='pt',padding='longest')

def generate(model,tokenizer,inp_batch,max_tokens,temperature):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_token_ids = inp_batch.input_ids.to(device)
    input_attention_mask = inp_batch.attention_mask.to(device)

    #Start the prompt with BOS
    last_token_ids = input_token_ids[:,0]
    token_ids = [input_token_ids[:,0]]
    
    kv = model.init_kv(last_token_ids[:,None],model)
    logits = []
    attention_mask = torch.cat([input_attention_mask,torch.ones((input_attention_mask.shape[0],max_tokens+1),dtype=torch.int64).to(device)],dim=1)
    for i in range(input_token_ids.shape[1]+max_tokens-1):
        #Forward the model
        out = model(last_token_ids[:,None],return_dict=True,past_key_values=kv,use_cache=True,attention_mask=attention_mask[:,:i+1])
        last_logits = out.logits[:,0,:]
        kv = out.past_key_values
        #For phi-1.5b model that requires a different format for storing kv
        if hasattr(kv,'sequence_len_offset'):
            kv.sequence_len_offset += 1
        logits.append(last_logits)

        #Sample new tokens from logits
        if i < input_token_ids.shape[1]-1:
            last_token_ids = input_token_ids[:,i+1]
        else:
            if temperature > 0:
                probs = (last_logits/temperature).softmax(-1).double()
                last_token_ids = torch.multinomial(probs,num_samples=1,replacement=True)[:,0]
            else:
                last_token_ids = last_logits.argmax(-1)
        token_ids.append(last_token_ids)

    return torch.stack(token_ids,dim=1).cpu(),torch.stack(logits,dim=1).cpu()

def dict_to_openai(token_ids,logits,tokenizer,temperature,return_logits=False,nb_logprobs=5,inputs_to_remove=None):
    """ Returns the data in the OPENAI format"""
    #Compute logp associated with these ids
    if temperature > 0:
        logprobs = (logits/temperature).softmax(-1).log()
        generated_tokens_logp = [lprobs[token_id].item() for lprobs,token_id in zip(logprobs,token_ids[1:])]
        #Top logp computations
        top_logp = []
        for lprobs in logprobs:
            best_token_ids = lprobs.argsort()[-nb_logprobs:]
            top_logp.append({tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens([token_id.item()])[0]]):lprobs[token_id.item()].item() for token_id in best_token_ids})

        #Outputs
        tokens_logprobs = [None] + generated_tokens_logp
        top_logprobs = [None] + top_logp
    else:
        tokens_logprobs = [None]*len(token_ids)
        top_logprobs = [None]*len(token_ids)
    

    #Translate the token ids into text
    logging.debug('Translating tokens into text')
    generated_tokens_raw = [tokenizer.convert_ids_to_tokens([token_id])[0] for token_id in token_ids]
    logging.debug(generated_tokens_raw)
    generated_tokens = [tokenizer.convert_tokens_to_string([t]) for t in generated_tokens_raw]
    logging.debug(generated_tokens)
    generated_sequence = tokenizer.convert_tokens_to_string(generated_tokens_raw)
    logging.debug(generated_sequence)
    logging.debug(inputs_to_remove)
    logging.debug('----------')
    
    #Compute echo -> the bos token as well
    if not(inputs_to_remove is None):
        if tokenizer.bos_token is None:
            tokenizer.bos_token = ''
        generated_tokens_raw = [tokenizer.bos_token]+generated_tokens_raw[inputs_to_remove.shape[0]:]
        logging.debug(generated_tokens_raw)
        generated_tokens = [tokenizer.convert_tokens_to_string([t]) for t in generated_tokens_raw]
        logging.debug(generated_tokens)
        generated_sequence = tokenizer.convert_tokens_to_string(generated_tokens_raw)
        logging.debug(generated_sequence)

        tokens_logprobs = tokens_logprobs[inputs_to_remove.shape[0]:]
        top_logprobs = top_logprobs[inputs_to_remove.shape[0]:]
        
    del generated_tokens[0] #Remove bos
    generated_sequence = generated_sequence[len(tokenizer.bos_token):] #Remove bos
    
    #remove padding tokens
    pad_token = tokenizer.pad_token
    index_pad = [] #Index of the non padded tokens
    for i in range(len(generated_tokens[1:])):
        if generated_tokens[i] == pad_token:
            continue
        index_pad.append(i)
    generated_tokens = [generated_tokens[i] for i in index_pad]
    tokens_logprobs = [tokens_logprobs[i] for i in index_pad]
    top_logprobs = [top_logprobs[i] for i in index_pad]
    logits = logits.numpy().astype(np.float16)[index_pad,:]
    logits = logits.tolist()
    
    #Return the dict
    out =  {'choices':[
        {'text':generated_sequence,
         'logprobs':{
             'tokens': generated_tokens,
             'token_logprobs': tokens_logprobs,
             'top_logprobs': top_logprobs,
         },
         'finish_reason':'Not Implemented'
        }
    ]
    }
    if return_logits:
        out['choices'][0]['logprobs']['logits'] = logits
    return out

    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             LLaMa Models
#
#---------------------------------------------------------------------------------------------------------------
    
    
class LlamaFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map='auto')#,torch_dtype=torch.bfloat16)
        return tokenizer,model

class Llama7B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-7b'
    @property
    def name(self):
        return 'LLA7'
    
class Llama13B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-13b'
    @property
    def name(self):
        return 'LLA13'
    
class Alpaca7B(LlamaFamily):
    @property
    def engine(self):
        return 'alpaca-7b'
    @property
    def name(self):
        return 'ALP7'
    
class Wizard7B(LlamaFamily):
    @property
    def engine(self):
        return 'wizard-7b'
    @property
    def name(self):
        return 'WIZ7'
    
class Vicuna7B_11(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-7b-v1.1'
    @property
    def name(self):
        return 'VIC7_11'
    
class Vicuna7B_13(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-7b-v1.3'
    @property
    def name(self):
        return 'VIC7_13'
    
class Vicuna7B_15(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-7b-v1.5'
    @property
    def name(self):
        return 'VIC7_15'
    
class Vicuna13B_11(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-13b-v1.1'
    @property
    def name(self):
        return 'VIC13_11'
    
class Vicuna13B_13(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-13b-v1.3'
    @property
    def name(self):
        return 'VIC13_13'
    
class Vicuna13B_15(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-13b-v1.5'
    @property
    def name(self):
        return 'VIC13_15'

class Baize7B(LlamaFamily):
    @property
    def engine(self):
        return 'baize-7b'
    @property
    def name(self):
        return 'BAI7'
    
class Guanaco7B(LlamaFamily):
    @property
    def engine(self):
        return 'guanaco-7b'
    @property
    def name(self):
        return 'GUA7'
    
class TinyLlama(LlamaFamily):
    @property
    def engine(self):
        return 'tiny-llama-fast-tokenizer'
    @property
    def name(self):
        return 'TLLA'
    
class Llama2_7B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-2-7b'
    @property
    def name(self):
        return 'LLA2_7'
    
class Llama2_13B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-2-13b'
    @property
    def name(self):
        return 'LLA2_13'
    
class Llama2HF_7B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-2-7b-hf'
    @property
    def name(self):
        return 'LLA2HF_7'
    
class Llama2HF_13B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-2-13b-hf'
    @property
    def name(self):
        return 'LLA2HF_13'
    
"""class Llama2GGUF_7B(LlamaFamily):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),model_file="llama-2-7b-gguf.Q3_K_S.gguf")
        return tokenizer,model
    @property
    def engine(self):
        return 'llama-2-7b-gguf'
    @property
    def name(self):
        return 'LLA2HF_7'
    
class Llama2GGUF_13B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-2-13b-gguf'
    @property
    def name(self):
        return 'LLA2HF_13'"""
    
class Orca2_7B(LlamaFamily):
    @property
    def engine(self):
        return 'Orca-2-7b'
    @property
    def name(self):
        return 'ORC2_7'
    
class Orca2_13B(LlamaFamily):
    @property
    def engine(self):
        return 'Orca-2-13b'
    @property
    def name(self):
        return 'ORC2_13'
    
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Bloom Models
#
#---------------------------------------------------------------------------------------------------------------
    
class BloomFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,BloomForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = BloomForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map='auto')
        return tokenizer,model

    
class Bloom3B(BloomFamily):
    @property
    def engine(self):
        return 'bloom-3b'
    @property
    def name(self):
        return 'BL3'
    
class Bloom7B(BloomFamily):
    @property
    def engine(self):
        return 'bloom-7b'
    @property
    def name(self):
        return 'BL7'
    
class BloomZ3B(BloomFamily):
    @property
    def engine(self):
        return 'bloomz-3b'
    @property
    def name(self):
        return 'BLZ3'
    
class BloomZ7B(BloomFamily):
    @property
    def engine(self):
        return 'bloomz-7b'
    @property
    def name(self):
        return 'BLZ7'
    
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Pythia Models
#
#---------------------------------------------------------------------------------------------------------------

class PythiaFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,GPTNeoXForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = GPTNeoXForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        return tokenizer,model

    
class Pythia3B(PythiaFamily):
    @property
    def engine(self):
        return 'pythia-2.8b'
    @property
    def name(self):
        return 'PY3'
    
class Pythia7B(PythiaFamily):
    @property
    def engine(self):
        return 'pythia-6.9b'
    @property
    def name(self):
        return 'PY7'
    
class Pythia12B(PythiaFamily):
    @property
    def engine(self):
        return 'pythia-12b'
    @property
    def name(self):
        return 'PY12'
    
class Dolly3B(PythiaFamily):
    @property
    def engine(self):
        return 'dolly-v2-3b'
    @property
    def name(self):
        return 'DL3'
    
class Dolly7B(PythiaFamily):
    @property
    def engine(self):
        return 'dolly-v2-7b'
    @property
    def name(self):
        return 'DL7'
    
class Dolly12B(PythiaFamily):
    @property
    def engine(self):
        return 'dolly-v2-12b'
    @property
    def name(self):
        return 'DL12'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             PHI Models
#
#---------------------------------------------------------------------------------------------------------------
    
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
@dataclass
class InferenceParams:
    """Inference parameters passed to model to efficiently calculate
    and store context during inference.
    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/generation.py.
    Args:
        max_sequence_len: Maximum sequence length.
        max_batch_size: Maximum batch size.
        sequence_len_offset: Sequence length offset.
        batch_size_offset: Batch size offset.
        key_value_memory_dict: Key value memory dictionary.
        fused_ft_kernel: Whether to use fused kernel for fast inference.
        lengths_per_sample: Lengths per sample.
    """

    max_sequence_len: int = field(metadata={"help": "Maximum sequence length."})

    max_batch_size: int = field(metadata={"help": "Maximum batch size."})

    sequence_len_offset: int = field(default=0, metadata={"help": "Sequence length offset."})

    batch_size_offset: int = field(default=0, metadata={"help": "Batch size offset."})

    key_value_memory_dict: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Key value memory dictionary."}
    )

    fused_ft_kernel: bool = field(default=False, metadata={"help": "Whether to use fused kernel for fast inference."})

    lengths_per_sample: torch.Tensor = field(default=None, metadata={"help": "Lengths per sample."})
    
class PhiFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine), trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine), trust_remote_code=True)
        return tokenizer,model
    
class Phi1(PhiFamily):
    @property
    def engine(self):
        return 'phi-1'
    @property
    def name(self):
        return 'PHI_1'
    
class Phi15(PhiFamily):
    @property
    def engine(self):
        return 'phi-1_5'
    @property
    def name(self):
        return 'PHI_1.5'
    def init_kv(self,inputs,model):
        return InferenceParams(
                max_batch_size=inputs.shape[0],
                max_sequence_len=model.config.n_positions,
                sequence_len_offset=0,
                batch_size_offset=0,
                fused_ft_kernel=False,
                key_value_memory_dict={},
            )

#-------------------------------------------------------------------------------------------------------------- 
#
#                                             StableLM Models
#
#---------------------------------------------------------------------------------------------------------------
"""
class StableLMFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        return tokenizer,model
    
class StableLM3B(StableLMFamily):
    @property
    def engine(self):
        return 'stablelm-base-alpha-3b'
    @property
    def name(self):
        return 'SLM_3'
    
class StableLM7B(StableLMFamily):
    @property
    def engine(self):
        return 'stablelm-base-alpha-7b'
    @property
    def name(self):
        return 'SLM_7'
    
class StableLMT3B(StableLMFamily):
    @property
    def engine(self):
        return 'stablelm-tuned-alpha-3b'
    @property
    def name(self):
        return 'SLMT_3'
    
class StableLMT7B(StableLMFamily):
    @property
    def engine(self):
        return 'stablelm-tuned-alpha-7b'
    @property
    def name(self):
        return 'SLMT_7'
"""
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Cerebras Models
#
#---------------------------------------------------------------------------------------------------------------
    
class CerebrasFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,GPT2LMHeadModel
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        return tokenizer,model
    
class Cerebras111M(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-111m'
    @property
    def name(self):
        return 'CER_111M'
    
class Cerebras256M(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-256m'
    @property
    def name(self):
        return 'CER_256M'
    
class Cerebras590M(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-590m'
    @property
    def name(self):
        return 'CER_590M'
    
class Cerebras1B(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-1.3b'
    @property
    def name(self):
        return 'CER_1'
    
class Cerebras3B(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-2.7b'
    @property
    def name(self):
        return 'CER_3'
    
class Cerebras7B(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-6.7b'
    @property
    def name(self):
        return 'CER_7'
    
class Cerebras13B(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-13b'
    @property
    def name(self):
        return 'CER_13'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             QWEN Models
#
#---------------------------------------------------------------------------------------------------------------
    
class QWENFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine),trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),trust_remote_code=True)
        return tokenizer,model
    
class QWEN2B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen-1_8B'
    @property
    def name(self):
        return 'QWE_2'
    
class QWEN7B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen-7B'
    @property
    def name(self):
        return 'QWE_7'
    
class QWEN14B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen-14B'
    @property
    def name(self):
        return 'QWE_14'
    
class QWEN72B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen-72B'
    @property
    def name(self):
        return 'QWE_72'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             YI Models
#
#---------------------------------------------------------------------------------------------------------------

class YiFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        return tokenizer,model
    
class Yi6B(YiFamily):
    @property
    def engine(self):
        return 'Yi-6B'
    @property
    def name(self):
        return 'YI_6'
    
class Yi34B(YiFamily):
    @property
    def engine(self):
        return 'Yi-34B'
    @property
    def name(self):
        return 'YI_34'

"""
class Yi6B_GGUF(YiFamily):
    @property
    def engine(self):
        return 'Yi-6B-GGUF'
    @property
    def name(self):
        return 'YI_6_GGUF'
    
class Yi34B_GGUF(YiFamily):
    @property
    def engine(self):
        return 'Yi-34B-GGUF'
    @property
    def name(self):
        return 'YI_34_GGUF'
"""
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Mistral Models
#
#---------------------------------------------------------------------------------------------------------------

class MistralFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        return tokenizer,model
    
class Mistral7B(MistralFamily):
    @property
    def engine(self):
        return 'Mistral-7B-v0.1'
    @property
    def name(self):
        return 'MIS_7'
    
class Mistral7BI(MistralFamily):
    @property
    def engine(self):
        return 'Mistral-7B-Instruct-v0.1'
    @property
    def name(self):
        return 'MISI_7'
    
class Zephyr7BA(MistralFamily):
    @property
    def engine(self):
        return 'zephyr-7b-alpha'
    @property
    def name(self):
        return 'ZPHA_7'
    
class Zephyr7BB(MistralFamily):
    @property
    def engine(self):
        return 'zephyr-7b-beta'
    @property
    def name(self):
        return 'ZPHB_7'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             T5 Models
#
#---------------------------------------------------------------------------------------------------------------
"""
class T5Family(HFModel):
    def load_model(self):
        from transformers import T5Tokenizer,T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = T5ForConditionalGeneration.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        return tokenizer,model
    
class T5Small(T5Family):
    @property
    def engine(self):
        return 't5-small'
    @property
    def name(self):
        return 'T5S'
    
class T5Base(T5Family):
    @property
    def engine(self):
        return 't5-base'
    @property
    def name(self):
        return 'T5B'
    
class T5Large(T5Family):
    @property
    def engine(self):
        return 't5-large'
    @property
    def name(self):
        return 'T5L'
    
class T5XL(T5Family):
    @property
    def engine(self):
        return 't5-3b'
    @property
    def name(self):
        return 'T5XL'
    
class T5XXL(T5Family):
    @property
    def engine(self):
        return 't5-11b'
    @property
    def name(self):
        return 'T5XXL'
    
class T5Small11(T5Family):
    @property
    def engine(self):
        return 't5-v1_1-small'
    @property
    def name(self):
        return 'T5S11'
    
class T5Base11(T5Family):
    @property
    def engine(self):
        return 't5-v1_1-base'
    @property
    def name(self):
        return 'T5B11'
    
class T5Large11(T5Family):
    @property
    def engine(self):
        return 't5-v1_1-large'
    @property
    def name(self):
        return 'T5L11'
    
class T5XL11(T5Family):
    @property
    def engine(self):
        return 't5-v1_1-xl'
    @property
    def name(self):
        return 'T5XL11'
    
class T5XXL11(T5Family):
    @property
    def engine(self):
        return 't5-v1_1-xl'
    @property
    def name(self):
        return 'T5XXL11'
    
class FlanT5Small(T5Family):
    @property
    def engine(self):
        return 'flan-t5-small'
    @property
    def name(self):
        return 'FT5S'
    
class FlanT5Base(T5Family):
    @property
    def engine(self):
        return 'flan-t5-base'
    @property
    def name(self):
        return 'FT5B'
    
class FlanT5Large(T5Family):
    @property
    def engine(self):
        return 'flan-t5-large'
    @property
    def name(self):
        return 'FT5L'
    
class FlanT5XL(T5Family):
    @property
    def engine(self):
        return 'flan-t5-xl'
    @property
    def name(self):
        return 'FT5XL'
    
class FlanT5XXL(T5Family):
    @property
    def engine(self):
        return 'flan-t5-xxl'
    @property
    def name(self):
        return 'FT5XXL'
"""
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             FUYU Models
#
#---------------------------------------------------------------------------------------------------------------

class FUYUFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        return tokenizer,model
    
class FUYU8B(FUYUFamily):
    @property
    def engine(self):
        return 'fuyu-8b'
    @property
    def name(self):
        return 'FUYU8B'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Falcon Models
#
#---------------------------------------------------------------------------------------------------------------

class FalconFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine))#,trust_remote_code=True)
        return tokenizer,model
    
class FalconRW1B(FalconFamily):
    @property
    def engine(self):
        return 'falcon-rw-1b'
    @property
    def name(self):
        return 'FALRW1B'
    
class FalconRW7B(FalconFamily):
    @property
    def engine(self):
        return 'falcon-rw-7b'
    @property
    def name(self):
        return 'FALRW7B'
    
class Falcon7B(FalconFamily):
    @property
    def engine(self):
        return 'falcon-7b'
    @property
    def name(self):
        return 'FAL7B'
    
class Falcon7BI(FalconFamily):
    @property
    def engine(self):
        return 'falcon-7b-instruct'
    @property
    def name(self):
        return 'FAL7BI'
    
class Falcon40B(FalconFamily):
    @property
    def engine(self):
        return 'falcon-40b'
    @property
    def name(self):
        return 'FAL40B'
    
class Falcon40BI(FalconFamily):
    @property
    def engine(self):
        return 'falcon-40b-instruct'
    @property
    def name(self):
        return 'FAL40BI'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             OPT Models
#
#---------------------------------------------------------------------------------------------------------------

class OPTFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        return tokenizer,model
    
class OPT125M(OPTFamily):
    @property
    def engine(self):
        return 'opt-125m'
    @property
    def name(self):
        return 'OPT125M'
    
class OPT350M(OPTFamily):
    @property
    def engine(self):
        return 'opt-350m'
    @property
    def name(self):
        return 'OPT350M'
    
class OPT1B(OPTFamily):
    @property
    def engine(self):
        return 'opt-1.3b'
    @property
    def name(self):
        return 'OPT1B'
    
class OPT3B(OPTFamily):
    @property
    def engine(self):
        return 'opt-2.7b'
    @property
    def name(self):
        return 'OPT3B'
    
class OPT7B(OPTFamily):
    @property
    def engine(self):
        return 'opt-6.7b'
    @property
    def name(self):
        return 'OPT7B'
    
class OPT13B(OPTFamily):
    @property
    def engine(self):
        return 'opt-13b'
    @property
    def name(self):
        return 'OPT13B'
    
class OPT30B(OPTFamily):
    @property
    def engine(self):
        return 'opt-30b'
    @property
    def name(self):
        return 'OPT30B'
    
class OPT66B(OPTFamily):
    @property
    def engine(self):
        return 'opt-66b'
    @property
    def name(self):
        return 'OPT66B'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             OpenChat Models
#
#---------------------------------------------------------------------------------------------------------------
    
class OpenChatFamily(MistralFamily):
    pass
    
class OpenChat2(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_v2'
    @property
    def name(self):
        return 'OC2'
    
class OpenChat2W(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_v2_w'
    @property
    def name(self):
        return 'OC2W'
    
class OpenChat31(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_v3.1'
    @property
    def name(self):
        return 'OC31'
    
class OpenChat32(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_v3.2'
    @property
    def name(self):
        return 'OC32'
    
class OpenChat32Super(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_v3.2_super'
    @property
    def name(self):
        return 'OC32S'
    
class OpenChat35(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_3.5'
    @property
    def name(self):
        return 'OC35'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Berkeley-Nest Models
#
#---------------------------------------------------------------------------------------------------------------
class BerkeleyNestFamily(OpenChatFamily):
    pass
    
class Starling7BAlpha(BerkeleyNestFamily):
    @property
    def engine(self):
        return 'Starling-LM-7B-alpha'
    @property
    def name(self):
        return 'STL7BA'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             NeuralChat Models
#
#---------------------------------------------------------------------------------------------------------------
class NeuralChatFamily(MistralFamily):
    pass
    
class NeuralChat3_7B(NeuralChatFamily):
    @property
    def engine(self):
        return 'neural-chat-7b-v3'
    @property
    def name(self):
        return 'NC3_7B'
    
class NeuralChat31_7B(NeuralChatFamily):
    @property
    def engine(self):
        return 'neural-chat-7b-v3-1'
    @property
    def name(self):
        return 'NC31_7B'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             CausalLM Models
#
#---------------------------------------------------------------------------------------------------------------
class CausalLMFamily(LlamaFamily):
    pass
    
class CausalLM7B(CausalLMFamily):
    @property
    def engine(self):
        return 'causallm-7b'
    @property
    def name(self):
        return 'CLM_7B'
    
class CausalLM14B(CausalLMFamily):
    @property
    def engine(self):
        return 'causallm-14b'
    @property
    def name(self):
        return 'CLM_14B'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             TigerBot Models
#
#---------------------------------------------------------------------------------------------------------------
class TigerBotFamily(LlamaFamily):
    pass

class TigerBotBase_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-base'
    @property
    def name(self):
        return 'TBB_7B'
    
class TigerBotBasev1_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-base-v1'
    @property
    def name(self):
        return 'TBBV1_7B'
    
class TigerBotBasev2_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-base-v2'
    @property
    def name(self):
        return 'TBBV2_7B'
    
class TigerBotSFTv1_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-sft-v1'
    @property
    def name(self):
        return 'TBFV1_7B'
    
class TigerBotSFTv2_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-sft-v2'
    @property
    def name(self):
        return 'TBFV2_7B'
    
class TigerBotChat_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-chat'
    @property
    def name(self):
        return 'TBC_7B'
    
"""class TigerBotChat4B_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-chat-4bit'
    @property
    def name(self):
        return 'TBC4B_7B'
    
class TigerBotChat8B_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-chat-8bit'
    @property
    def name(self):
        return 'TBC8B_7B'"""
    
class TigerBotBasev1_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-base-v1'
    @property
    def name(self):
        return 'TBBV1_13B'
    
class TigerBotBasev2_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-base-v2'
    @property
    def name(self):
        return 'TBBV2_13B'
    
class TigerBotChatv1_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-chat-v1'
    @property
    def name(self):
        return 'TBCV1_13B'

class TigerBotChatv2_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-chat-v2'
    @property
    def name(self):
        return 'TBCV2_13B'
    
class TigerBotChatv3_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-chat-v3'
    @property
    def name(self):
        return 'TBCV3_13B'
    
class TigerBotChatv4_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-chat-v4'
    @property
    def name(self):
        return 'TBCV4_13B'
    
"""class TigerBotChatv4_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-chat-4bit-exl2'
    @property
    def name(self):
        return 'TBC4B_13B'"""
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             OpenHermes Models
#
#---------------------------------------------------------------------------------------------------------------
class OpenHermes7B(LlamaFamily):
    @property
    def engine(self):
        return 'OpenHermes-7B'
    @property
    def name(self):
        return 'OH_7B'

class OpenHermes13B(LlamaFamily):
    @property
    def engine(self):
        return 'OpenHermes-13B'
    @property
    def name(self):
        return 'OH_13B'
    
class OpenHermes2_7B(MistralFamily):
    @property
    def engine(self):
        return 'OpenHermes-2-Mistral-7B'
    @property
    def name(self):
        return 'OH2_7B'
    
class OpenHermes25_7B(MistralFamily):
    @property
    def engine(self):
        return 'OpenHermes-2.5-Mistral-7B'
    @property
    def name(self):
        return 'OH25_7B'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             NexusFlow Models
#
#---------------------------------------------------------------------------------------------------------------
class NexusFlowFamily(LlamaFamily):
    pass 

class NexusRaven13B(NexusFlowFamily):
    @property
    def engine(self):
        return 'NexusRaven-13B'
    @property
    def name(self):
        return 'NR_13B'
    
class NexusRavenv2_13B(NexusFlowFamily):
    @property
    def engine(self):
        return 'NexusRaven-V2-13B'
    @property
    def name(self):
        return 'NR2_13B'
   
    
def get_hf_model_classes():
    return HFModel.subclasses
