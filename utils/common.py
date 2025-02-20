import os
import json
import yaml
from config.config_wrapper import get_config_wrapper

def read_yaml(config='default'):
    if os.path.exists(f'config/prompt/{config}.yaml'):
        yaml_file = f'config/prompt/{config}.yaml'
    else:
        yaml_file = config
    with open(yaml_file, 'r', encoding='utf-8') as yaml_file:
        return yaml.safe_load(yaml_file)

def write_jsonl_lines(file, data):
    config_wrapper = get_config_wrapper()
    if config_wrapper.save_prompt:
        json.dump(data, file, ensure_ascii=False)
    else:
        data.pop(config_wrapper.prompt_key)
        json.dump(data, file, ensure_ascii=False)
    file.write('\n')
    file.flush()

def print_info(info):
    config_wrapper = get_config_wrapper()
    print('-'*100)
    print("[INFO] model_name:", info['model_name'])
    print("[INFO] splits:", info['splits'])
    print("[INFO] modes:", info['modes'])
    print("[INFO] output_dir:", info['output_dir'])
    print("[INFO] Infer Limit:", "No limit" if info['infer_limit'] is None else info['infer_limit'])
    print("[INFO] Number of Workers:", info['num_workers'])
    print("[INFO] Batch Size:", info['batch_size'])
    print("[INFO] Temperatrue:", config_wrapper.temperatrue)
    print("[INFO] Use Accel:", info['use_accel'])
    print("[INFO] Index:", info['index'])
    print("[INFO] World Size:", info['world_size'])
    print('-'*100)

def read_json_or_jsonl(data_path, split='', mapping_key=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        file_path = base_path
    else:
        raise FileNotFoundError("No JSON or JSONL file found.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.json'):
            data = json.load(file)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in file]
            # data = []
            # for line in file:
            #     print(line)
            #     data.append(json.loads(line))
    
    if mapping_key:
        return {item[mapping_key]: item for item in data if mapping_key in item}
    else:
        return data

def read_json_or_jsonl_with_idx(data_path, split='', idx=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        file_path = base_path
    else:
        raise FileNotFoundError("No JSON or JSONL file found.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.json'):
            data = json.load(file)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in file]
    
    if idx is not None:
        try:
            return next(item for item in data if item.get('idx') == idx)
        except StopIteration:
            raise ValueError(f"No entry found for idx {idx}")
    else:
        return data