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

def send_message_and_get_response(message, model_name, no_think=False, use_openai_api=False, use_cpu=False):
    """发送消息并获取响应，支持OpenAI API和本地Ollama API"""
    if use_openai_api:
        return _call_openai_api(message, model_name)
    else:
        return _call_ollama_api(message, model_name, no_think, use_cpu)

def _call_openai_api(message, model_name):
    """调用OpenAI兼容API（百炼）"""
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key="fake_api",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        messages = [{"role": "user", "content": message}]
        
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
            
            # 提取Dockerfile内容
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
            print(f"OpenAI API调用错误: {str(e)}")
            return None
            
    except ImportError:
        print("OpenAI库未安装，请运行: pip install openai")
        return None
    except Exception as e:
        print(f"初始化OpenAI客户端错误: {str(e)}")
        return None

def _call_ollama_api(message, model_name, no_think=False, use_cpu=False):
    """调用本地Ollama API"""
    url = "http://localhost:11434/api/chat"
    
    # 模型差异化控制
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
    
    # 添加CPU运行选项
    if use_cpu:
        payload["options"]["num_gpu"] = 0  # 强制使用CPU
        print("🔧 使用CPU模式运行模型")

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
            
            # 提取Dockerfile内容
            dockerfile_pattern = re.compile(r'```dockerfile(.*?)```', re.DOTALL | re.IGNORECASE)
            match = dockerfile_pattern.search(message_content)
            if match:
                dockerfile_content = match.group(1).strip()
                return dockerfile_content
            else:
                print("No Dockerfile found in the response")
                return None
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

def process_dockerfiles(json_path, root_folder, mode_name, mode_dir, time_log_file=None, no_think=False, use_openai_api=False, use_cpu=False):
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
        issues = dockerfile["issues"]
        
        # 记录开始时间
        start_time = time.time()
        
        with open(dockerfile_path, 'r', encoding='utf-8') as file:
            original_content = file.read()
        
        modified_filepath = dockerfile_path.replace(root_folder, mode_dir)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(modified_filepath), exist_ok=True)
        
        if not issues:
            with open(modified_filepath, 'w', encoding='utf-8') as file:
                file.write(original_content)
            print(f"{modified_filepath} Skipping with perfect.")
            
            # 记录跳过信息
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
        
        dockerfile_issue_str = "\n".join(dockerfile["issues"])

        # 构造严格prompt
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
            
            # 记录失败信息
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
        print("Usage: python your_script.py json_path root_folder mode_name mode_dir [time_log_dir] [--no-think] [--use-openai-api] [--use-cpu]")
        print("\n参数说明:")
        print("  json_path: JSON文件路径")
        print("  root_folder: 原始Dockerfile根目录")
        print("  mode_name: 模型名称")
        print("  mode_dir: 输出目录")
        print("  time_log_dir: 时间记录目录（可选）")
        print("  --no-think: 启用无思考模式（仅对Qwen有效）")
        print("  --use-openai-api: 使用OpenAI兼容API（百炼）")
        print("  --use-cpu: 使用CPU运行模型（仅对Ollama有效）")
        sys.exit(1)
    
    json_path = sys.argv[1]
    root_folder = sys.argv[2]
    mode_name = sys.argv[3]
    mode_dir = sys.argv[4]
    
    # 设置时间记录目录
    time_log_dir = 'time/star/hd_llm'
    if len(sys.argv) > 5 and not sys.argv[5].startswith('--'):
        time_log_dir = sys.argv[5]
    
    # 创建时间记录目录
    os.makedirs(time_log_dir, exist_ok=True)
    
    # Check for flags
    no_think = '--no-think' in sys.argv
    use_openai_api = '--use-openai-api' in sys.argv
    use_cpu = '--use-cpu' in sys.argv
    
    # 生成时间记录文件名（基于模型名称和模式）
    model_safe_name = mode_name.replace(':', '_').replace('/', '_')
    think_suffix = '_nothink' if no_think else ''
    api_suffix = '_openai' if use_openai_api else ''
    cpu_suffix = '_cpu' if use_cpu else ''
    time_log_file = os.path.join(time_log_dir, f'hd_llm_repair_{model_safe_name}{think_suffix}{api_suffix}{cpu_suffix}.json')
    
    print(f"配置信息:")
    print(f"  JSON路径: {json_path}")
    print(f"  根目录: {root_folder}")
    print(f"  模型: {mode_name}")
    print(f"  输出目录: {mode_dir}")
    print(f"  时间记录: {time_log_file}")
    print(f"  无思考模式: {no_think}")
    print(f"  OpenAI API: {use_openai_api}")
    print(f"  CPU模式: {use_cpu}")
    
    # 执行修复
    repair_times = process_dockerfiles(json_path, root_folder, mode_name, mode_dir, time_log_file, no_think, use_openai_api, use_cpu)
    
    # 移除注释
    # remove_comments_in_lines(mode_dir)
    
    # 生成摘要报告
    summary_file = os.path.join(time_log_dir, f'summary_hd_llm_repair_{model_safe_name}{think_suffix}{api_suffix}{cpu_suffix}.json')
    generate_summary_report(time_log_file, summary_file)
    
    print(f"\n所有处理完成！时间记录保存在: {time_log_dir}")

if __name__ == "__main__":
    main()

    # python repair_methods/hd_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3:32b" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_32b_hd_LLM_nothink" --no-think
    # python repair_methods/hd_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3:8b" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_8b_hd_LLM_nothink" --no-think
# python repair_methods/hd_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3-235b-a22b-instruct-2507" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_235b_hd_LLM_1" --use-openai-api
# python repair_methods/hd_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3:0.6b" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_06b_hd_LLM_nothink" --no-think