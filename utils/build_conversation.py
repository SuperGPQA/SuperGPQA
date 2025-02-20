def build_conversation(history={}, prompt='', system_prompt=''):
    conversation = []
    if system_prompt:
        conversation.append({'role': 'system', 'content': system_prompt})
    for idx, message in history.items():
        conversation.append({'role': 'user', 'content': message['prompt']})
        conversation.append({'role': 'assistant', 'content': message['response']})
    conversation.append({'role': 'user', 'content': prompt})
    return conversation

# TODO: multimodal history
def build_qwen2vl_conversation(content=[], system_prompt=''):
    conversation = []
    if system_prompt:
        conversation.append({'role': 'system', 'content': system_prompt})
    # for idx, message in history.items():
    #     conversation.append({'role': 'user', 'content': message['prompt']})
    #     conversation.append({'role': 'assistant', 'content': message['response']})
    conversation.append({'role': 'user', 'content': content})
    return conversation

if __name__ == '__main__':
    history = {}
    system_prompt = 'You are a helpful assistant'
    print(build_conversation(history, 'What is the capital of Belgium?', system_prompt))
    
