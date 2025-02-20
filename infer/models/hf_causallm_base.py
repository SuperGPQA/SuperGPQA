from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from config.config_wrapper import config_wrapper


def load_model(model_name, model_args, use_accel=False):
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    model_components = {}
    if use_accel:
        model_components['use_accel'] = True
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
        model_components['model_name'] = model_name
    else:
        model_components['use_accel'] = False
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
        model_components['model_name'] = model_name
    return model_components

def infer(prompts, historys, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    model_name = kwargs.get('model_name', None)
    use_accel = kwargs.get('use_accel', False)
    
    if isinstance(prompts[0], str):
        prompts = prompts
    else:
        raise ValueError("Invalid prompts format")
    if use_accel:
        stop_token_ids=[tokenizer.eos_token_id]
        sampling_params = SamplingParams(max_tokens=config_wrapper.max_tokens, stop_token_ids=stop_token_ids, temperature=config_wrapper.temperatrue)
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
        responses = []
        for output in outputs:
            response = output.outputs[0].text
            responses.append(response)
    else:
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=config_wrapper.max_tokens, do_sample=False)
        responses = []
        for i, prompt in enumerate(prompts):
            response = tokenizer.decode(outputs[i, len(inputs['input_ids'][i]):], skip_special_tokens=True)
            responses.append(response)
    return responses

if __name__ == '__main__':
    prompts = [
        '''Can you tell me a story about a time-traveling cat?''',
        '''What happened when a group of friends found a mysterious treasure map in their attic?''',
    ]
    model_args = {
        'model_path_or_name': 'Qwen/Qwen2.5-0.5B',
        'model_type': 'local',
        'tp': 1
    }
    model_components = load_model("Qwen2.5-0.5B", model_args, use_accel=True)
    # model_components = {"model": None, "chat_template": get_chat_template_from_config('')}
    responses = infer(prompts, **model_components)
    for response in responses:
        print(response)
