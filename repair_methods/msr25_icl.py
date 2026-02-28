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




from openai import OpenAI

def send_message_and_get_response(message, model_name, no_think=False, use_openai_api=False):
    # Determine which API to use
    if use_openai_api:
        # Use OpenAI-compatible API (Bailian)
        return _call_openai_api(message, model_name)
    else:
        # Use local Ollama API
        return _call_ollama_api(message, model_name, no_think)

def _call_openai_api(message, model_name):
    """Call OpenAI-compatible API (Bailian)"""
    try:
        client = OpenAI(
            api_key="you_api",
            # api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        
        messages = [
            {"role": "system", "content": "You are an expert in Docker and software refactoring. You're provided a Dockerfile and must recommend improvements strictly from a predefined list of refactorings the output should be JSON of the Original Problems, Refactorings you are willing to apply and the new Refactored Dockerfile (the dockerfile should be in plain text, no array and correct syntax ready to be built): ."},
            {"role": "user", "content": message}
        ]
        
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
            print("Raw response:", message_content)
            
            # Use unified response parsing function
            return _parse_response_content(message_content)
            
        except TimeoutError:
            print("Request timed out after 180 seconds")
            return None
        except Exception as e:
            print(f"OpenAI API call error: {str(e)}")
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
    
    messages = [
        {"role": "system", "content": "You are an expert in Docker and software refactoring. You're provided a Dockerfile and must recommend improvements strictly from a predefined list of refactorings the output should be JSON of the Original Problems, Refactorings you are willing to apply and the new Refactored Dockerfile (the dockerfile should be in plain text, no array and correct syntax ready to be built): ."},
        {"role": "user", "content": message}
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 4096
        }
    }

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
            print("Raw response:", message_content)
            
            # Use unified response parsing function
            return _parse_response_content(message_content)
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

def _parse_response_content(message_content):
    """Unified response content parsing function"""
    # Method 1: Extract JSON code block
    try:
        data = json.loads(message_content)
        dockerfile_content = data.get("Refactored Dockerfile")
        if dockerfile_content:
            print("Method 1 success: extracted JSON code block")
            return dockerfile_content
    except json.JSONDecodeError as e:
        print(f"JSON code block parsing error: {e}")
    
    # Method 2: Parse format with title and code block
    # Match format: **Refactored Dockerfile:** followed by ```dockerfile code block
    pattern_title_codeblock = r'\*\*Refactored Dockerfile:\*\*\s*```(?:dockerfile)?\s*(.*?)\s*```'
    match = re.search(pattern_title_codeblock, message_content, re.DOTALL)
    if match:
        dockerfile_content = match.group(1).strip()
        print("Method 2 success: parsed format with title and code block")
        return dockerfile_content
    
    # Method 3: Flexible regular expression matching
    patterns = [
        r'"Refactored Dockerfile":\s*"((?:[^"\\]|\\.)*)"\s*}',
        r'"Refactored Dockerfile"\s*:\s*"([^"]*)"\s*}',
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, message_content, re.DOTALL)
        if match:
            dockerfile_content = match.group(1)
            # Handle escape characters
            dockerfile_content = dockerfile_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            print(f"Method 3.{i+1} success: regular expression match")
            return dockerfile_content
    
    # Method 4: If all methods fail, return original content for subsequent processing
    print("All parsing methods failed, returning raw response content")
    return message_content

def process_dockerfiles(json_path, root_folder, mode_name, mode_dir, time_log_file=None, no_think=False, use_openai=False):
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
        
        # Record start time
        start_time = time.time()
        
        with open(dockerfile_path, 'r', encoding='utf-8') as file:
            original_content = file.read()
        
        modified_filepath = dockerfile_path.replace(root_folder, mode_dir)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(modified_filepath), exist_ok=True)
        
        # Remove logic to skip files with no issues, process all files
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

        prompt = f"""
 You are an expert in Docker and software refactoring. You're provided with a Dockerfile. Your task is to analyze this Dockerfile in detail and perform the following steps:

1. **Identify Problems:** Identify any problems or inefficiencies in the Dockerfile that could be optimized. Focus on issues related to build efficiency, image size, understandability and maintainability.

2. **Refactoring Techniques:** After identifying the problems, go through the following list of refactoring techniques and determine which ones could be applied to solve the identified problems:
   - Replace ADD with COPY Instruction: Replace `ADD` with `COPY` for non-URL sources to improve clarity and reduce build context size. if there is no add used don't opt for this refactoring
   - Inline Run Instructions: Combine adjacent `RUN` commands using `&&` to reduce the number of layers and optimize the image size.
   - Multistage builds to separate the build environment from the runtime environment, reducing the final image size.
   - Update Base-Image-TAG:  when the base image tag is 'latest'. Specify an explicit version in the `FROM` statement instead of using `latest`.
   - Update Base Image: verify if the current base image is oversized, or a more specific image is available (based on the packages and dependencies used in the Dockerfile). Update the `FROM` statement to a more suitable base image and perform necessary changes if needed to ensure compatibility.
   - Rename Image: when stages' names are missing or could be better for clarity and understandability add meaningful names or rename using `AS` in multi-stage builds
   - Add ARG instruction: Introduce ARG instructions to definie build time variables to customize and parameterize the build process without hardcoding values.
   - Introduce environment variables for configuration instead of hard-coded values.
   - Inline stage: if multi-staging is used in the dockerfile, verify its worthiness and if it does not reduce complexity or if all intermediate artifacts are needed in the final image. Remove multi-stage building.
   - Sort Instructions: Rearrange instructions to optimize layer caching.

3. **Apply Refactorings:** Based on the assessment in step 2, refactor the Dockerfile. Implement the selected refactoring techniques, ensuring that each change maintains or improves the functionality and performance of the Dockerfile. Provide a detailed explanation for each refactoring applied, including how it addresses the issues identified in step 1.

##Dockerfile for Analysis:
{original_content}

4. **Output Format:** Your response should be structured as follows:
   - **Original Problems:** Identify the current problems.
   - **Refactoring you will to apply:** Describe each refactoring technique applied and its rationale. (if any; otherwise leave empty)
   - **Refactored Dockerfile:** Provide the complete refactored Dockerfile that can be copied and used immeditly into the project, that incorporates all the changes.(if any; otherwise keep same dockerfile)

Ensure that the final refactored Dockerfile is fully functional, ready to be built, and optimized according to the refactoring techniques listed. The response should be clear, concise, and directly applicable to the provided Dockerfile.
"""

        modified_content = send_message_and_get_response(prompt, mode_name, no_think, use_openai)
        
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
                'no_think': no_think
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
                'no_think': no_think
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
        print("Usage: python your_script.py json_path root_folder mode_name mode_dir [time_log_dir] [--no-think]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    root_folder = sys.argv[2]
    mode_name = sys.argv[3]
    mode_dir = sys.argv[4]
    
    # Set time log directory
    time_log_dir = 'time/star/msricl'
    if len(sys.argv) > 5 and not sys.argv[5].startswith('--'):
        time_log_dir = sys.argv[5]
    
    # Create time log directory
    os.makedirs(time_log_dir, exist_ok=True)
    
    # Generate time log filename (based on model name and mode)
    model_safe_name = mode_name.replace(':', '_').replace('/', '_')
    think_suffix = '_nothink' if '--no-think' in sys.argv else ''
    time_log_file = os.path.join(time_log_dir, f'hd_llm_repair_{model_safe_name}{think_suffix}.json')
    
    # Check for no_think flag
    no_think = '--no-think' in sys.argv
    use_openai = '--use-openai-api' in sys.argv
    
    # Execute repair
    repair_times = process_dockerfiles(json_path, root_folder, mode_name, mode_dir, time_log_file, no_think, use_openai)
    
    # Remove comments
    # remove_comments_in_lines(mode_dir)
    
    # Generate summary report
    summary_file = os.path.join(time_log_dir, f'summary_hd_llm_repair_{model_safe_name}{think_suffix}.json')
    generate_summary_report(time_log_file, summary_file)
    
    print(f"\nAll processing completed! Time records saved in: {time_log_dir}")

if __name__ == "__main__":
    main()




# python repair_methods/msr25_icl.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3-235b-a22b-instruct-2507" "repair_result/dataset_fast/star1000+_dockerfile/msr25_icl_qwen3_235b" --use-openai-api
# python repair_methods/msr25_icl.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "deepseek-r1-0528" "repair_result/dataset_fast/star1000+_dockerfile/msr25_icl_ds_671b" --use-openai-api