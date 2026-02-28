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

def send_message_and_get_response(message, model_name, no_think=False, use_openai_api=False):
    """Send message and get response, supports OpenAI API and local Ollama API"""
    if use_openai_api:
        return _call_openai_api(message, model_name)
    else:
        return _call_ollama_api(message, model_name, no_think)

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

def _call_ollama_api(message, model_name, no_think=False):
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
            "num_predict": 8192
        }
    }

    try:
        # Set timeout
        def timeout_handler(signum, frame):
            raise requests.exceptions.Timeout("Request timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(180)

        response = requests.post(url, json=payload)
        signal.alarm(0)

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

def remove_comments_in_lines(folder_path):
    """Remove comments in Dockerfiles"""
    # Traverse all files in specified folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        print(f"Processing file: {filename}")
        
        # Read file content
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Process file content, remove comments within each line
        new_lines = []
        for line in lines:
            line = line.rstrip()
            comment_index = line.find('#')
            if comment_index != -1:
                line = line[:comment_index].rstrip()
            new_lines.append(line + '\n')
            
        # Write processed content back to file
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
            
    print(f"Completed: {folder_path}")

def parse_log_content(log_content):
    """Parse log content containing <phase>, <path>, and <error> tags"""
    # Use regular expressions to extract each part
    phase_pattern = r'<phase>(.*?)<phase>'
    path_pattern = r'<path>(.*?)<path>'
    error_pattern = r'<error>(.*?)<error>'

    phase = re.search(phase_pattern, log_content)
    path = re.search(path_pattern, log_content)
    error = re.search(error_pattern, log_content)

    # Extract matched content
    phase_content = phase.group(1) if phase else None
    path_content = path.group(1) if path else None
    error_content = error.group(1) if error else None

    return {
        'phase': phase_content,
        'path': path_content,
        'error': error_content
    }

def extract_original_path(full_path):
    """Extract format: dataset_fast/subsequent_path/final_filename"""
    parts = full_path.split('/')
    try:
        # Find index position of 'dataset_fast'
        start_idx = parts.index('dataset_fast')
        # Get first two directories and final filename
        original_path = '/'.join([
            parts[start_idx],       # dataset_fast
            parts[start_idx+1],     # ctf_dockerfile
            parts[-1]               # final filename
        ])
        return original_path
    except (ValueError, IndexError):
        return None

def process_dockerfiles(unbuild_path, model_name="qwen3:32b", no_think=False, use_openai_api=False):
    """Process Dockerfile repairs, supports OpenAI API"""
    
    with open(unbuild_path, 'r', encoding='utf-8') as file:
        unbuild_content = file.readlines()
    
    # Create output directory
    output_dir = "build_repair_result"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Record processing results
    repair_records = []
    
    for line in tqdm(unbuild_content, desc="Repairing build-failed Dockerfiles"):
        line = line.strip()
        if not line:
            continue
            
        parsed_log = parse_log_content(line)
        original_path = extract_original_path(parsed_log['path'])
        repair_path = parsed_log['path']
        
        if not original_path or not os.path.exists(original_path):
            print(f"Original file does not exist: {original_path}")
            continue
            
        if not os.path.exists(repair_path):
            print(f"Repair file does not exist: {repair_path}")
            continue
        
        # Read file contents
        with open(original_path, 'r', encoding='utf-8') as file:
            original_content = file.read()
        with open(repair_path, 'r', encoding='utf-8') as file:
            repair_content = file.read()
        
        # Determine failure stage
        last_step = parsed_log['phase'] if parsed_log['phase'] else "beginning"
        
        # Generate output path
        relative_path = os.path.relpath(repair_path, "repair_result")
        modified_filepath = os.path.join(output_dir, relative_path)
        
        # Check if target file already exists
        if os.path.exists(modified_filepath):
            print(f"File already exists, skipping: {modified_filepath}")
            repair_records.append({
                'original_path': original_path,
                'repair_path': repair_path,
                'output_path': modified_filepath,
                'status': 'skipped',
                'reason': 'already_exists',
                'timestamp': datetime.now().isoformat()
            })
            continue

        # Create output directory
        os.makedirs(os.path.dirname(modified_filepath), exist_ok=True)
        
        # Construct prompt
        prompt = (
            f"## Dockerfile Repair Analysis\n"
            f"**Original Dockerfile**:\n```dockerfile\n{original_content}\n```\n\n"
            f"**Repaired Dockerfile**:\n```dockerfile\n{repair_content}\n```\n"
            f"**Build Error**: `{parsed_log['error']}` (Failed at: {last_step})\n\n"
            "## Requirements\n"
            "Generate a corrected Dockerfile that:\n"
            "1. Retains ALL original functionality\n"
            "2. Fixes the build error while preserving docker smell repairs\n"
            "3. NO unrelated changes or new features\n"
            "4. Format:\n```dockerfile\n...\n```"
        )
        
        # Call LLM for repair
        start_time = time.time()
        modified_content = send_message_and_get_response(prompt, model_name, no_think, use_openai_api)
        repair_time = time.time() - start_time
        
        # Save result
        if modified_content:
            with open(modified_filepath, 'w', encoding='utf-8') as file:
                file.write(modified_content)
            
            status = 'success'
            reason = 'LLM repair successful'
            print(f"✅ Repair successful: {repair_path} -> {modified_filepath} ({repair_time:.2f}s)")
        else:
            # If LLM repair fails, save original repair content
            with open(modified_filepath, 'w', encoding='utf-8') as file:
                file.write(repair_content)
            
            status = 'failed'
            reason = 'LLM repair failed, saved original repair'
            print(f"❌ Repair failed: {repair_path} -> {modified_filepath} ({repair_time:.2f}s)")
        
        # Record processing result
        repair_records.append({
            'original_path': original_path,
            'repair_path': repair_path,
            'output_path': modified_filepath,
            'status': status,
            'reason': reason,
            'repair_time_seconds': round(repair_time, 2),
            'build_error': parsed_log['error'],
            'failed_at': last_step,
            'model': model_name,
            'no_think': no_think,
            'api_type': 'openai' if use_openai_api else 'ollama',
            'timestamp': datetime.now().isoformat()
        })
    
    # Save processing records
    if repair_records:
        record_file = f"build_repair_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(repair_records, f, indent=2, ensure_ascii=False)
        print(f"📊 Repair records saved: {record_file}")
    
    # Generate statistical summary
    generate_repair_summary(repair_records, output_dir)
    
    # Remove comments
    # remove_comments_in_lines(os.path.dirname(modified_filepath))
    
    print("All Dockerfiles processing completed.")

def generate_repair_summary(repair_records, output_dir):
    """Generate repair statistical summary"""
    if not repair_records:
        return
    
    successful_repairs = [r for r in repair_records if r['status'] == 'success']
    failed_repairs = [r for r in repair_records if r['status'] == 'failed']
    skipped_repairs = [r for r in repair_records if r['status'] == 'skipped']
    
    summary = {
        'total_files': len(repair_records),
        'successful_repairs': len(successful_repairs),
        'failed_repairs': len(failed_repairs),
        'skipped_repairs': len(skipped_repairs),
        'success_rate': round(len(successful_repairs) / len(repair_records) * 100, 2) if repair_records else 0,
        'avg_repair_time': round(sum(r.get('repair_time_seconds', 0) for r in repair_records) / len(repair_records), 2) if repair_records else 0,
        'total_processing_time': round(sum(r.get('repair_time_seconds', 0) for r in repair_records), 2),
        'output_directory': output_dir,
        'timestamp': datetime.now().isoformat()
    }
    
    # Print summary
    print(f"\n📊 Repair Statistical Summary:")
    print(f"   Total files processed: {summary['total_files']}")
    print(f"   Successful repairs: {summary['successful_repairs']} ({summary['success_rate']}%)")
    print(f"   Failed repairs: {summary['failed_repairs']}")
    print(f"   Skipped repairs: {summary['skipped_repairs']}")
    print(f"   Average repair time: {summary['avg_repair_time']} seconds")
    print(f"   Total processing time: {summary['total_processing_time']} seconds")
    print(f"   Output directory: {output_dir}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "repair_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📄 Statistical summary saved: {summary_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_repair.py unbuild_path [model_name] [--no-think] [--use-openai-api]")
        print("\nParameter description:")
        print("  unbuild_path: File path containing build failure logs")
        print("  model_name: Model name (default: qwen3:32b)")
        print("  --no-think: Enable no-think mode (only effective for Qwen)")
        print("  --use-openai-api: Use OpenAI-compatible API (Bailian)")
        sys.exit(1)
    
    unbuild_path = sys.argv[1]
    
    # Default model name
    # model_name = "qwen3:32b"
    no_think = False
    use_openai_api = False
    
    # Parse parameters
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--no-think":
            no_think = True
        elif sys.argv[i] == "--use-openai-api":
            use_openai_api = True
        elif not sys.argv[i].startswith("--"):
            model_name = sys.argv[i]
    
    print(f"🔧 Configuration information:")
    print(f"  Build failure log: {unbuild_path}")
    print(f"  Model name: {model_name}")
    print(f"  No-think mode: {no_think}")
    print(f"  OpenAI API: {use_openai_api}")
    
    # Execute repair
    process_dockerfiles(unbuild_path, model_name, no_think, use_openai_api)

if __name__ == "__main__":
    main()

# python repair_methods/build_repair.py evaluate_result/star/qwen3_235b_hd_LLM/_unbuild.txt "qwen3-235b-a22b-instruct-2507" --use-openai-api