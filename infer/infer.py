import json
import sys
import argparse
import os
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from queue import Queue

from tenacity import RetryError

from data_loader import load_data
from models import load_model, infer
from utils.common import write_jsonl_lines, print_info
from post_process.custom_post_processor import PostProcessorRegistry
from config.config_wrapper import initialize_config, get_config_wrapper

STATUS_DICT = {
    'to_skip': ['completed'],
    'to_resume': ['processing', 'error', 'max_rounds', 'resume'],
    'to_write': ['completed', 'error', 'max_rounds']
}

def check_response(data):
    response_key = config_wrapper.response_key
    error_key = config_wrapper.error_key
    if response_key in data and data[response_key] and (isinstance(data[response_key], str) 
                                or (isinstance(data[response_key], dict) and error_key not in data[response_key])):
        return True
    return False

# TODO: Add timestamp to ensure the check is correct
def check_status(data, status_list):
    status_key = config_wrapper.status_key
    if status_key not in data or (status_key in data and data[status_key] in status_list):
        return True
    return False

def check_completed(output_file, processor):
    completed = {}
    processing = {}
    no_response_id = []
    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                if check_response(data):
                    if processor and check_status(data, STATUS_DICT['to_skip']):
                        completed[config_wrapper.get_id(data)] = data
                    elif processor and check_status(data, STATUS_DICT['to_resume']):
                        processing[config_wrapper.get_id(data)] = data
                    elif not processor:
                        completed[config_wrapper.get_id(data)] = data
                else:
                    no_response_id.append(config_wrapper.get_id(data))
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    return completed, processing, no_response_id

def finalize_output(temp_output_file_path, output_file_path, processor):
    with open(temp_output_file_path, 'r', encoding='utf-8') as temp_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in temp_file:
            data = json.loads(line)
            if check_status(data, STATUS_DICT['to_write']):
                write_jsonl_lines(output_file, data)

    os.remove(temp_output_file_path)
    _, _, no_response_id = check_completed(output_file_path, processor)
    if len(no_response_id) > 0:
        print(f"Failed to get response for the following IDs: {no_response_id}", file=sys.stderr)
        print(f"Failed to get response for {len(no_response_id)} questions in {config_wrapper.mode} mode.", file=sys.stderr)
    

def infer_batch(model_components, model_name, batch):
    results = []
    prompts, historys = [sample[config_wrapper.prompt_key] for sample in batch], [sample.get(config_wrapper.history_key, {}) for sample in batch]
    try:
        responses = infer(model_name)(prompts, historys, **model_components)
        for sample, response in zip(batch, responses):
            sample[config_wrapper.response_key] = response
            results.append(sample)
    except RetryError as e:
        last_attempt = e.last_attempt
        if last_attempt:
            exception = last_attempt.exception()
            if exception:
                # print(f"Error processing {prompts}: {str(exception)}", file=sys.stderr)
                print(f"Error: {str(exception)}")
                for sample in batch:
                    sample[config_wrapper.response_key] = {"error": str(exception)}
                    results.append(sample)
    except Exception as e:
        # print(f"Error processing {prompts}: {str(e)}", file=sys.stderr)
        print(f"Error: {str(e)}")
        for sample in batch:
            sample[config_wrapper.response_key] = {"error": str(e)}
            results.append(sample)
    return results

def main(model_name='gpt4o', splits=[], modes=[], output_dir='results', infer_limit=None, num_workers=1, batch_size=1, index=0, world_size=1, use_accel=False):
    info = {
        'model_name': model_name,
        'splits': splits,
        'modes': modes,
        'output_dir': output_dir,
        'infer_limit': infer_limit,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'use_accel': use_accel,
        'index': index,
        'world_size': world_size
    }
    print_info(info)
    model_components = None
    
    os.makedirs(output_dir, exist_ok=True)
    for split in splits:
        for mode in modes:
            processor = PostProcessorRegistry.get_processor(mode)
            config_wrapper.mode = mode
            config_wrapper.split = split
            if index == 0 and world_size == 1:
                output_file_path = f'{output_dir}/{model_name}_{split}_{mode}.jsonl'
            else:
                output_file_path = f'{output_dir}/{model_name}_{split}_{mode}_{index}_{world_size}.jsonl'
            other_output_file_path = [path for path in glob.glob(f'{output_dir}/{model_name}_{split}_{mode}*.jsonl') if path != output_file_path]

            temp_output_file_path = f'{output_file_path}.tmp'
            temp_other_output_file_path = [f'{path}.tmp' for path in other_output_file_path]

            completed, processing, _ = check_completed(output_file_path, processor)
            temp_completed, temp_processing, _ = check_completed(temp_output_file_path, processor)

            merged_completed = {**temp_completed, **completed}
            merged_processing = {**temp_processing, **processing}
            
            for other_path in other_output_file_path:
                other_completed, other_processing, _ = check_completed(other_path, processor)
                merged_completed.update(other_completed)
                merged_processing.update(other_processing)
            
            for temp_other_path in temp_other_output_file_path:
                temp_other_completed, temp_other_processing, _ = check_completed(temp_other_path, processor)
                merged_completed.update(temp_other_completed)
                merged_processing.update(temp_other_processing)

            print(f'Found {len(merged_completed)} completed inferences for {split} {mode} mode.')
            print(f'Found {len(merged_processing)} processing inferences for {split} {mode} mode.')
            # print(merged_completed.keys())
            
            infer_count = 0

            with open(temp_output_file_path, 'w', encoding='utf-8') as temp_file:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    batch_queue = Queue()
                    batch = []

                    def process_batch(batch):
                        futures.append(executor.submit(infer_batch, model_components, model_name, batch))

                    for idx, (prompt, sample) in tqdm(enumerate(load_data(split=split, mode=mode)), desc=f'Processing {split} {mode} data'):
                        if idx % world_size != index:
                            continue
                            
                        sample[config_wrapper.prompt_key] = prompt
                        if config_wrapper.get_id(sample) in merged_completed:
                            sample = merged_completed[config_wrapper.get_id(sample)]
                            write_jsonl_lines(temp_file, sample)
                            continue
                        if prompt == '':
                            sample[config_wrapper.response_key] = ''
                            write_jsonl_lines(temp_file, sample)
                            continue
                        if config_wrapper.get_id(sample) in merged_processing:
                            sample = merged_processing[config_wrapper.get_id(sample)]
                            sample[config_wrapper.status_key] = 'resume'
                            write_jsonl_lines(temp_file, sample)
                        
                        if infer_limit is not None and infer_count >= infer_limit:
                            break
                        
                        batch.append(sample)
                        infer_count += 1
                        
                        if len(batch) == batch_size:
                            batch_queue.put(batch)
                            batch = []
                        
                        if infer_limit is not None and infer_count >= infer_limit:
                            break
                    
                    if batch:
                        batch_queue.put(batch)

                    while not batch_queue.empty():
                        if model_components is None:
                            model_components = load_model(model_name, use_accel)
                        process_batch(batch_queue.get())

                    def process_results(futures):
                        batch_to_return = []
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {split} {mode} results'):
                            results = future.result()
                            if processor:
                                results_to_save, results_to_return = processor.process(results)
                                for result in results_to_save:
                                    write_jsonl_lines(temp_file, result)
                                batch_to_return.extend(results_to_return)
                            else:
                                for result in results:
                                    write_jsonl_lines(temp_file, result)
                        return batch_to_return

                    batch_to_return = process_results(futures)
                    futures = []
                    
                    while batch_to_return:
                        while batch_to_return:
                            new_batch = list(batch_to_return[:min(batch_size, len(batch_to_return))])
                            batch_to_return = list(batch_to_return[min(batch_size, len(batch_to_return)):])
                            process_batch(new_batch)
                        batch_to_return = process_results(futures)
                        futures = []

            finalize_output(temp_output_file_path, output_file_path, processor)
            
        print(f'Inference for {split} completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference and save results.')
    parser.add_argument('--model_name', type=str, default='', help='Model name to use')
    parser.add_argument('--config', type=str, default='config/config_default.yaml', help='Config file to use')
    parser.add_argument('--split', nargs='+', default=[], help='Data split to use')
    parser.add_argument('--mode', nargs='+', default=[], help='Modes to use for data loading, separated by space')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to write results')
    parser.add_argument('--infer_limit', type=int, help='Limit the number of inferences per run, default is no limit', default=None)
    parser.add_argument('--num_workers', type=int, default=1, help='Number of concurrent workers for inference, currently only used for API')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference, currently only used for local model processing')
    parser.add_argument('--use_accel', action='store_true', help='Use inference acceleration framework for inference, LLM-->vLLM')
    parser.add_argument('--save_prompt', action='store_true', help='Save prompt to output file')
    parser.add_argument('--index', type=int, default=0, help='Index of the current worker')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of workers')
    args = parser.parse_args()
    initialize_config(args.config)
    config_wrapper = get_config_wrapper()
    main(model_name=args.model_name, splits=args.split, modes=args.mode, output_dir=args.output_dir, infer_limit=args.infer_limit, num_workers=args.num_workers, batch_size=args.batch_size, index=args.index, world_size=args.world_size, use_accel=args.use_accel)
