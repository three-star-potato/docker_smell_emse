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
    """保存时间记录到文件"""
    if not time_records:
        return
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 确定文件格式
    if filename.endswith('.json'):
        with open(filename, mode, encoding='utf-8') as f:
            if mode == 'a' and os.path.exists(filename) and os.path.getsize(filename) > 0:
                # 读取现有数据并追加
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
    """生成修复时间摘要报告"""
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
    
    # 分析数据
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
    
    # 打印摘要
    print("\n" + "="*50)
    print("修复时间摘要报告")
    print("="*50)
    print(f"总处理文件数: {summary['total_files']}")
    print(f"成功修复: {summary['successful_repairs']}")
    print(f"修复失败: {summary['failed_repairs']}")
    print(f"平均修复时间: {summary['avg_repair_time']}秒")
    print(f"总处理时间: {summary['total_processing_time']}秒")
    
    # 保存摘要报告
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"摘要报告已保存至: {output_file}")
    
    return summary




from openai import OpenAI

def send_message_and_get_response(message, model_name, no_think=False, use_openai_api=False):
    # 判断使用哪种API
    if use_openai_api:
        # 使用OpenAI兼容API（百炼）
        return _call_openai_api(message, model_name)
    else:
        # 使用本地Ollama API
        return _call_ollama_api(message, model_name, no_think)

def _call_openai_api(message, model_name):
    """调用OpenAI兼容API（百炼）"""
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
        
        # 设置超时
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
            signal.alarm(0)  # 重置超时
            
            message_content = completion.choices[0].message.content
            print("原始响应:", message_content)
            
            # 使用统一的响应解析函数
            return _parse_response_content(message_content)
            
        except TimeoutError:
            print("Request timed out after 180 seconds")
            return None
        except Exception as e:
            print(f"OpenAI API调用错误: {str(e)}")
            return None
            
    except Exception as e:
        print(f"初始化OpenAI客户端错误: {str(e)}")
        return None

def _call_ollama_api(message, model_name, no_think=False):
    """调用本地Ollama API"""
    url = "http://localhost:11434/api/chat"
    
    # 模型差异化控制
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
        # 设置超时
        def timeout_handler(signum, frame):
            raise requests.exceptions.Timeout("Request timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(180)

        response = requests.post(url, json=payload)
        signal.alarm(0)  # 重置超时

        if response.status_code == 200:
            result = response.json()
            message_content = result['message']['content']
            print("原始响应:", message_content)
            
            # 使用统一的响应解析函数
            return _parse_response_content(message_content)
        else:
            print(f"API返回错误状态码: {response.status_code}")
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
    """统一的响应内容解析函数"""
    # 方法1：提取JSON代码块
    try:
        data = json.loads(message_content)
        dockerfile_content = data.get("Refactored Dockerfile")
        if dockerfile_content:
            print("方法1成功：提取JSON代码块")
            return dockerfile_content
    except json.JSONDecodeError as e:
        print(f"JSON代码块解析错误: {e}")
    
    # 方法2：解析带标题和代码块的格式
    # 匹配格式：**Refactored Dockerfile:** 后跟 ```dockerfile 代码块
    pattern_title_codeblock = r'\*\*Refactored Dockerfile:\*\*\s*```(?:dockerfile)?\s*(.*?)\s*```'
    match = re.search(pattern_title_codeblock, message_content, re.DOTALL)
    if match:
        dockerfile_content = match.group(1).strip()
        print("方法2成功：解析带标题的代码块格式")
        return dockerfile_content
    
    # 方法3：宽松的正则表达式匹配
    patterns = [
        r'"Refactored Dockerfile":\s*"((?:[^"\\]|\\.)*)"\s*}',
        r'"Refactored Dockerfile"\s*:\s*"([^"]*)"\s*}',
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, message_content, re.DOTALL)
        if match:
            dockerfile_content = match.group(1)
            # 处理转义字符
            dockerfile_content = dockerfile_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            print(f"方法3.{i+1}成功：正则表达式匹配")
            return dockerfile_content
    
    # 方法4：如果所有方法都失败，返回原始内容让后续处理
    print("所有解析方法都失败，返回原始响应内容")
    return message_content

def process_dockerfiles(json_path, root_folder, mode_name, mode_dir, time_log_file=None, no_think=False,use_openai=False):
    """处理Dockerfiles并记录时间"""
    if not os.path.exists(mode_dir):    
        os.makedirs(mode_dir)
    
    # 时间记录数据结构
    time_records = []
    
    # Read data from the specified JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        data_json = json.load(file)
    
    # Iterate over each Dockerfile, read its content, modify it, and save to a new file
    for dockerfile in tqdm(sorted(data_json, key=lambda x: x['dockerfile_path'])):
        dockerfile_path = dockerfile["dockerfile_path"]
        
        # 记录开始时间
        start_time = time.time()
        
        with open(dockerfile_path, 'r', encoding='utf-8') as file:
            original_content = file.read()
        
        modified_filepath = dockerfile_path.replace(root_folder, mode_dir)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(modified_filepath), exist_ok=True)
        
        # 移除跳过无issues文件的逻辑，所有文件都处理
        if os.path.exists(modified_filepath):
            print(f"Modified Dockerfile '{modified_filepath}' already exists. Skipping.")
            
            # 记录跳过信息
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

        modified_content = send_message_and_get_response(prompt, mode_name, no_think,use_openai)
        
        # 记录结束时间
        end_time = time.time()
        repair_time = end_time - start_time
        
        if modified_content:
            with open(modified_filepath, 'w', encoding='utf-8') as file:
                file.write(modified_content)
            
            # 记录成功信息
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
            
            # 记录失败信息
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
    
    # 保存时间记录
    if time_log_file:
        save_time_records(time_records, time_log_file)
    
    print("All Dockerfiles processed.")
    return time_records

def remove_comments_in_lines(folder_path):
    """移除Dockerfile中的注释"""
    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        # 只处理以 Dockerfile 开头的文件
       
        print(f"处理文件: {filename}")
            # 读取文件内容
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # 处理文件内容，去除每行内的注释
        new_lines = []
        for line in lines:
            # 去除行尾的空白字符
            line = line.rstrip()
                # 查找注释符号 '#' 的位置
            comment_index = line.find('#')
            if comment_index != -1:
                line = line[:comment_index].rstrip()  # 去除注释部分后的内容
            new_lines.append(line + '\n')  # 添加换行符保持原有格式
            
            # 将处理后的内容写回文件
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
            
    print(f"已完成: {folder_path}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python your_script.py json_path root_folder mode_name mode_dir [time_log_dir] [--no-think]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    root_folder = sys.argv[2]
    mode_name = sys.argv[3]
    mode_dir = sys.argv[4]
    
    # 设置时间记录目录
    time_log_dir = 'time/star/msricl'
    if len(sys.argv) > 5 and not sys.argv[5].startswith('--'):
        time_log_dir = sys.argv[5]
    
    # 创建时间记录目录
    os.makedirs(time_log_dir, exist_ok=True)
    
    # 生成时间记录文件名（基于模型名称和模式）
    model_safe_name = mode_name.replace(':', '_').replace('/', '_')
    think_suffix = '_nothink' if '--no-think' in sys.argv else ''
    time_log_file = os.path.join(time_log_dir, f'hd_llm_repair_{model_safe_name}{think_suffix}.json')
    
    # Check for no_think flag
    no_think = '--no-think' in sys.argv
    use_openai = '--use-openai-api' in sys.argv
    
    # 执行修复
    repair_times = process_dockerfiles(json_path, root_folder, mode_name, mode_dir, time_log_file, no_think,use_openai)
    
    # 移除注释
    # remove_comments_in_lines(mode_dir)
    
    # 生成摘要报告
    summary_file = os.path.join(time_log_dir, f'summary_hd_llm_repair_{model_safe_name}{think_suffix}.json')
    generate_summary_report(time_log_file, summary_file)
    
    print(f"\n所有处理完成！时间记录保存在: {time_log_dir}")

if __name__ == "__main__":
    main()




# python repair_methods/msr25_icl.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3-235b-a22b-instruct-2507" "repair_result/dataset_fast/star1000+_dockerfile/msr25_icl_qwen3_235b" --use-openai-api
# python repair_methods/msr25_icl.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "deepseek-r1-0528" "repair_result/dataset_fast/star1000+_dockerfile/msr25_icl_ds_671b" --use-openai-api