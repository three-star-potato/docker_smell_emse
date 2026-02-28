import json
import requests
import random
import re
from tqdm import tqdm
import fnmatch
import os
import sys
import signal
import time
from datetime import datetime

def save_time_records(time_records, filename, mode='w'):
    """Save time records to file"""
    if not time_records:
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Determine file format
    if filename.endswith('.json'):
        with open(filename, mode, encoding='utf-8') as f:
            if mode == 'a' and os.path.exists(filename) and os.path.getsize(filename) > 0:
                # Read existing data and append
                try:
                    f.seek(0)
                    existing_data = json.load(f)
                    existing_data.extend(time_records)
                    f.seek(0)
                    f.truncate()
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Error reading existing JSON file: {e}, creating new file")
                    json.dump(time_records, f, indent=2, ensure_ascii=False)
            else:
                json.dump(time_records, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Time records saved: {filename}")

def generate_summary_report(time_log_file, output_file=None):
    """Generate repair time summary report"""
    if not os.path.exists(time_log_file):
        print(f"Time log file not found: {time_log_file}")
        return
    
    if time_log_file.endswith('.json'):
        with open(time_log_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
    else:
        records = []
    
    if not records:
        print("No records found in time log")
        return
    
    # Analyze data
    successful_repairs = [r for r in records if r.get('status') == 'success']
    failed_repairs = [r for r in records if r.get('status') in ['error', 'timeout']]
    
    summary = {
        'total_files': len(records),
        'successful_repairs': len(successful_repairs),
        'failed_repairs': len(failed_repairs),
        'avg_repair_time': round(sum(r.get('repair_time_seconds', 0) for r in records) / len(records), 2) if records else 0,
        'total_processing_time': round(sum(r.get('repair_time_seconds', 0) for r in records), 2),
        'timestamp': datetime.now().isoformat()
    }
    
    # Print summary
    print("\n" + "="*50)
    print("Repair Time Summary Report")
    print("="*50)
    print(f"Total files processed: {summary['total_files']}")
    print(f"Successful repairs: {summary['successful_repairs']}")
    print(f"Failed repairs: {summary['failed_repairs']}")
    print(f"Average repair time: {summary['avg_repair_time']} seconds")
    print(f"Total processing time: {summary['total_processing_time']} seconds")
    
    # Save summary report
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary report saved to: {output_file}")
    
    return summary

def send_message_and_get_response(message, model_name, no_think=False, use_openai_api=False, use_cpu=False):
    """Send message and get response, supports OpenAI API and local Ollama API"""
    if use_openai_api:
        return _call_openai_api(message, model_name)
    else:
        return _call_ollama_api(message, model_name, no_think, use_cpu)

def _call_openai_api(message, model_name):
    """Call OpenAI-compatible API (Bailian)"""
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key="fake_api",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        messages = [{"role": "user", "content": message}]
        
        # Set timeout
        def timeout_handler(signum, frame):
            raise TimeoutError("Request timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(180)
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False,
                temperature=0.3,
                max_tokens=4096
            )
            signal.alarm(0)  # Reset timeout
            
            message_content = completion.choices[0].message.content
            
            # Extract Dockerfile content
            dockerfile_pattern = re.compile(r'```dockerfile(.*?)```', re.DOTALL | re.IGNORECASE)
            match = dockerfile_pattern.search(message_content)
            if match:
                dockerfile_content = match.group(1).strip()
                return dockerfile_content
            else:
                print("No Dockerfile found in the response")
                return None
                
        except TimeoutError:
            print("Request timed out after 180 seconds")
            return None
        except Exception as e:
            print(f"OpenAI API call error: {str(e)}")
            return None
            
    except ImportError:
        print("OpenAI library not installed, please run: pip install openai")
        return None
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")
        return None

def _call_ollama_api(message, model_name, no_think=False, use_cpu=False):
    """Call local Ollama API"""
    url = "http://localhost:11434/api/chat"
    
    # Model differentiation control
    if no_think:
        if "qwen3" in model_name.lower():
            message = f"/no_think\n\n{message}"
    
    messages = [{"role": "user", "content": message}]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 4096
        }
    }
    
    # Add CPU running option
    if use_cpu:
        payload["options"]["num_gpu"] = 0  # Force CPU usage
        print("🔧 Running model in CPU mode")

    try:
        # Set timeout
        def timeout_handler(signum, frame):
            raise requests.exceptions.Timeout("Request timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(180)

        response = requests.post(url, json=payload)
        signal.alarm(0)  # Reset timeout

        if response.status_code == 200:
            result = response.json()
            message_content = result['message']['content']
            
            # Extract Dockerfile content
            dockerfile_pattern = re.compile(r'```dockerfile(.*?)```', re.DOTALL | re.IGNORECASE)
            match = dockerfile_pattern.search(message_content)
            if match:
                dockerfile_content = match.group(1).strip()
                return dockerfile_content
            else:
                print("No Dockerfile found in the response")
                return None
        else:
            print(f"API returned error status code: {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        print("Request timed out after 180 seconds")
        return None
    except requests.exceptions.RequestException as e:
        print("Error:", str(e))
        return None
    except Exception as e:
        print("Unexpected error:", str(e))
        return None

def process_dockerfiles(json_path, root_folder, mode_name, mode_dir, time_log_file=None, no_think=False, use_openai_api=False, use_cpu=False):
    """Process Dockerfiles and record time"""
    if not os.path.exists(mode_dir):
        os.makedirs(mode_dir)
    
    # Time record data structure
    time_records = []
    
    # Read data from the specified JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        data_json = json.load(file)
    
    # Iterate over each Dockerfile, read its content, modify it, and save to a new file
    for dockerfile in tqdm(sorted(data_json, key=lambda x: x['dockerfile_path'])):
        dockerfile_path = dockerfile["dockerfile_path"]
        issues = dockerfile["issues"]
        
        # Record start time
        start_time = time.time()
        
        with open(dockerfile_path, 'r', encoding='utf-8') as file:
            original_content = file.read()
        
        modified_filepath = dockerfile_path.replace(root_folder, mode_dir)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(modified_filepath), exist_ok=True)
        
        if not issues:
            with open(modified_filepath, 'w', encoding='utf-8') as file:
                file.write(original_content)
            print(f"{modified_filepath} Skipping with perfect.")
            
            # Record skip information
            end_time = time.time()
            repair_time = end_time - start_time
            time_record = {
                'dockerfile': dockerfile_path,
                'repaired_file': modified_filepath,
                'repair_time_seconds': round(repair_time, 2),
                'status': 'skipped',
                'reason': 'no_issues',
                'timestamp': datetime.now().isoformat()
            }
            time_records.append(time_record)
            continue
        
        if os.path.exists(modified_filepath):
            print(f"Modified Dockerfile '{modified_filepath}' already exists. Skipping.")
            
            # Record skip information
            end_time = time.time()
            repair_time = end_time - start_time
            time_record = {
                'dockerfile': dockerfile_path,
                'repaired_file': modified_filepath,
                'repair_time_seconds': round(repair_time, 2),
                'status': 'skipped',
                'reason': 'already_exists',
                'timestamp': datetime.now().isoformat()
            }
            time_records.append(time_record)
            continue
        
        dockerfile_issue_str = "\n".join(dockerfile["issues"])

        # Construct strict prompt
        prompt = (
            f"Original Dockerfile:\n```dockerfile\n{original_content}\n```\n\n"
            f"Smells need to fix:\n{dockerfile_issue_str}\n\n"
            "Return ONLY the modified Dockerfile that:\n"
            "1. Is directly buildable with `docker build`\n"
            "2. Preserves all original functionality\n"
            "3. NO new features added\n\n"
            "4. CRITICAL: ALL package versions MUST be preserved exactly as in original (apt-get, apk, yum, etc.)\n"
                "- If original has versions, keep them unchanged\n"
                "- If original has NO versions, do NOT add versions\n"
            "5. Format:\n```dockerfile\n...\n```"
        )

        modified_content = send_message_and_get_response(prompt, mode_name, no_think, use_openai_api, use_cpu)
        
        # Record end time
        end_time = time.time()
        repair_time = end_time - start_time
        
        if modified_content:
            with open(modified_filepath, 'w', encoding='utf-8') as file:
                file.write(modified_content)
            
            # Record success information
            time_record = {
                'dockerfile': dockerfile_path,
                'repaired_file': modified_filepath,
                'repair_time_seconds': round(repair_time, 2),
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'model': mode_name,
                'no_think': no_think,
                'api_type': 'openai' if use_openai_api else 'ollama',
                'use_cpu': use_cpu
            }
            time_records.append(time_record)
            
            print(f"✅ LLM repair executed successfully in {repair_time:.2f}s: {dockerfile_path}")
        else:
            print(f"Failed to modify Dockerfile '{dockerfile_path}'. Saved original as '{modified_filepath}'")
            with open(modified_filepath, 'w', encoding='utf-8') as file:
                file.write(original_content)
            
            # Record failure information
            time_record = {
                'dockerfile': dockerfile_path,
                'repaired_file': modified_filepath,
                'repair_time_seconds': round(repair_time, 2),
                'status': 'error',
                'reason': 'llm_failed',
                'timestamp': datetime.now().isoformat(),
                'model': mode_name,
                'no_think': no_think,
                'api_type': 'openai' if use_openai_api else 'ollama',
                'use_cpu': use_cpu
            }
            time_records.append(time_record)
    
    # Save time records
    if time_log_file:
        save_time_records(time_records, time_log_file)
    
    print("All Dockerfiles processed.")
    return time_records

def remove_comments_in_lines(folder_path):
    """Remove comments in Dockerfiles"""
    # Traverse all files in specified folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        # Only process files starting with Dockerfile
       
        print(f"Processing file: {filename}")
            # Read file content
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # Process file content, remove comments within each line
        new_lines = []
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
                # Find comment symbol '#' position
            comment_index = line.find('#')
            if comment_index != -1:
                line = line[:comment_index].rstrip()  # Remove content after comment
            new_lines.append(line + '\n')  # Add newline to maintain original format
            
            # Write processed content back to file
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
            
    print(f"Completed: {folder_path}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python your_script.py json_path root_folder mode_name mode_dir [time_log_dir] [--no-think] [--use-openai-api] [--use-cpu]")
        print("\nParameter description:")
        print("  json_path: JSON file path")
        print("  root_folder: Original Dockerfile root directory")
        print("  mode_name: Model name")
        print("  mode_dir: Output directory")
        print("  time_log_dir: Time log directory (optional)")
        print("  --no-think: Enable no-think mode (only effective for Qwen)")
        print("  --use-openai-api: Use OpenAI-compatible API (Bailian)")
        print("  --use-cpu: Use CPU to run model (only effective for Ollama)")
        sys.exit(1)
    
    json_path = sys.argv[1]
    root_folder = sys.argv[2]
    mode_name = sys.argv[3]
    mode_dir = sys.argv[4]
    
    # Set time log directory
    time_log_dir = 'time/star/hd_llm'
    if len(sys.argv) > 5 and not sys.argv[5].startswith('--'):
        time_log_dir = sys.argv[5]
    
    # Create time log directory
    os.makedirs(time_log_dir, exist_ok=True)
    
    # Check for flags
    no_think = '--no-think' in sys.argv
    use_openai_api = '--use-openai-api' in sys.argv
    use_cpu = '--use-cpu' in sys.argv
    
    # Generate time log filename (based on model name and mode)
    model_safe_name = mode_name.replace(':', '_').replace('/', '_')
    think_suffix = '_nothink' if no_think else ''
    api_suffix = '_openai' if use_openai_api else ''
    cpu_suffix = '_cpu' if use_cpu else ''
    time_log_file = os.path.join(time_log_dir, f'hd_llm_repair_{model_safe_name}{think_suffix}{api_suffix}{cpu_suffix}.json')
    
    print(f"Configuration information:")
    print(f"  JSON path: {json_path}")
    print(f"  Root directory: {root_folder}")
    print(f"  Model: {mode_name}")
    print(f"  Output directory: {mode_dir}")
    print(f"  Time log: {time_log_file}")
    print(f"  No-think mode: {no_think}")
    print(f"  OpenAI API: {use_openai_api}")
    print(f"  CPU mode: {use_cpu}")
    
    # Execute repair
    repair_times = process_dockerfiles(json_path, root_folder, mode_name, mode_dir, time_log_file, no_think, use_openai_api, use_cpu)
    
    # Remove comments
    # remove_comments_in_lines(mode_dir)
    
    # Generate summary report
    summary_file = os.path.join(time_log_dir, f'summary_hd_llm_repair_{model_safe_name}{think_suffix}{api_suffix}{cpu_suffix}.json')
    generate_summary_report(time_log_file, summary_file)
    
    print(f"\nAll processing completed! Time records saved in: {time_log_dir}")

if __name__ == "__main__":
    main()

    # python repair_methods/hd_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3:32b" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_32b_hd_LLM_nothink" --no-think
    # python repair_methods/hd_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3:8b" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_8b_hd_LLM_nothink" --no-think
# python repair_methods/hd_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3-235b-a22b-instruct-2507" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_235b_hd_LLM_1" --use-openai-api
# python repair_methods/hd_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3:0.6b" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_06b_hd_LLM_nothink" --no-think