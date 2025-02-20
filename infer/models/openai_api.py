from openai import OpenAI
from utils.vl_utils import make_interleave_content
from utils.build_conversation import build_conversation
from config.config_wrapper import config_wrapper


def load_model(model_name="GPT4", base_url="", api_key="", model="", call_type='api_chat'):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    model_components['call_type'] = call_type
    return model_components

def request(messages, timeout=6000, max_tokens=2000, base_url="", api_key="", model="", model_name=None):
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages = messages,
        stream=False,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=config_wrapper.temperatrue,
    )
    return response

def request_to_base_model(prompt, timeout=6000, max_tokens=2000, base_url="", api_key="", model="", model_name=None):
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        timeout=timeout
    )
    print(response)
    
    return response

def request_with_images(texts_or_image_paths, timeout=60, max_tokens=2000, base_url="", api_key="", model="gpt-4o", model_name=None):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": make_interleave_content(texts_or_image_paths),
            }  
        ],  
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response

def infer(prompts, historys, **kwargs):
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)
    call_type = kwargs.get('call_type', 'api_chat')
    try:
        if call_type == 'api_chat':
            if isinstance(prompts, list):
                if len(prompts) > 1:
                    print(f'[Warning] infer/models/openai_api.py: Multiple prompts detected, only the first one will be processed')
                prompts = prompts[0]
                historys = historys[0]
            if isinstance(prompts, dict) and 'images' in prompts:
                prompts, images = prompts['prompt'], prompts['images']
                images = ["<|image|>" + image for image in images]
                response = request_with_images([prompts, *images], max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name).choices[0].message.content
            else:
                messages = build_conversation(historys, prompts)
                # print(messages)
                if model_name == 'DeepSeek-V3':
                    response = request(messages, max_tokens=8192, base_url=base_url, api_key=api_key, model=model, model_name=model_name).choices[0].message.content
                else:
                    response = request(messages, max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name).choices[0].message.content
        elif call_type == 'api_base':
            response = request_to_base_model(prompts, max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name).choices[0].text
        else:
            raise ValueError(f'Invalid call_type: {call_type}')
    except Exception as e:
        response = {"error": str(e)}
    # print(response)
    return [response]

    

