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
    """发送消息并获取响应，支持OpenAI API和本地Ollama API"""
    if use_openai_api:
        return _call_openai_api(message, model_name)
    else:
        return _call_ollama_api(message, model_name, no_think)

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

def _call_ollama_api(message, model_name, no_think=False):
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
            "num_predict": 8192
        }
    }

    try:
        # 设置超时
        def timeout_handler(signum, frame):
            raise requests.exceptions.Timeout("Request timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(180)

        response = requests.post(url, json=payload)
        signal.alarm(0)

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

def remove_comments_in_lines(folder_path):
    """移除Dockerfile中的注释"""
    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        print(f"处理文件: {filename}")
        
        # 读取文件内容
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # 处理文件内容，去除每行内的注释
        new_lines = []
        for line in lines:
            line = line.rstrip()
            comment_index = line.find('#')
            if comment_index != -1:
                line = line[:comment_index].rstrip()
            new_lines.append(line + '\n')
            
        # 将处理后的内容写回文件
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
            
    print(f"已完成: {folder_path}")

def parse_log_content(log_content):
    """解析包含<phase>、<path>和<error>标签的日志内容"""
    # 使用正则表达式提取各部分内容
    phase_pattern = r'<phase>(.*?)<phase>'
    path_pattern = r'<path>(.*?)<path>'
    error_pattern = r'<error>(.*?)<error>'

    phase = re.search(phase_pattern, log_content)
    path = re.search(path_pattern, log_content)
    error = re.search(error_pattern, log_content)

    # 提取匹配到的内容
    phase_content = phase.group(1) if phase else None
    path_content = path.group(1) if path else None
    error_content = error.group(1) if error else None

    return {
        'phase': phase_content,
        'path': path_content,
        'error': error_content
    }

def extract_original_path(full_path):
    """提取格式：dataset_fast/随后的路径/最后文件名"""
    parts = full_path.split('/')
    try:
        # 找到'dataset_fast'的索引位置
        start_idx = parts.index('dataset_fast')
        # 获取前两个目录和最后文件名
        original_path = '/'.join([
            parts[start_idx],       # dataset_fast
            parts[start_idx+1],     # ctf_dockerfile
            parts[-1]               # 最后文件名
        ])
        return original_path
    except (ValueError, IndexError):
        return None

def process_dockerfiles(unbuild_path, model_name="qwen3:32b", no_think=False, use_openai_api=False):
    """处理Dockerfiles修复，支持OpenAI API"""
    
    with open(unbuild_path, 'r', encoding='utf-8') as file:
        unbuild_content = file.readlines()
    
    # 创建输出目录
    output_dir = "build_repair_result"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录处理结果
    repair_records = []
    
    for line in tqdm(unbuild_content, desc="修复构建失败的Dockerfiles"):
        line = line.strip()
        if not line:
            continue
            
        parsed_log = parse_log_content(line)
        original_path = extract_original_path(parsed_log['path'])
        repair_path = parsed_log['path']
        
        if not original_path or not os.path.exists(original_path):
            print(f"原始文件不存在: {original_path}")
            continue
            
        if not os.path.exists(repair_path):
            print(f"修复文件不存在: {repair_path}")
            continue
        
        # 读取文件内容
        with open(original_path, 'r', encoding='utf-8') as file:
            original_content = file.read()
        with open(repair_path, 'r', encoding='utf-8') as file:
            repair_content = file.read()
        
        # 确定失败阶段
        last_step = parsed_log['phase'] if parsed_log['phase'] else "beginning"
        
        # 生成输出路径
        relative_path = os.path.relpath(repair_path, "repair_result")
        modified_filepath = os.path.join(output_dir, relative_path)
        
        # 检查目标文件是否已存在
        if os.path.exists(modified_filepath):
            print(f"文件已存在，跳过处理: {modified_filepath}")
            repair_records.append({
                'original_path': original_path,
                'repair_path': repair_path,
                'output_path': modified_filepath,
                'status': 'skipped',
                'reason': 'already_exists',
                'timestamp': datetime.now().isoformat()
            })
            continue

        # 创建输出目录
        os.makedirs(os.path.dirname(modified_filepath), exist_ok=True)
        
        # 构造prompt
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
        
        # 调用LLM进行修复
        start_time = time.time()
        modified_content = send_message_and_get_response(prompt, model_name, no_think, use_openai_api)
        repair_time = time.time() - start_time
        
        # 保存结果
        if modified_content:
            with open(modified_filepath, 'w', encoding='utf-8') as file:
                file.write(modified_content)
            
            status = 'success'
            reason = 'LLM repair successful'
            print(f"✅ 修复成功: {repair_path} -> {modified_filepath} ({repair_time:.2f}s)")
        else:
            # 如果LLM修复失败，保存原始修复内容
            with open(modified_filepath, 'w', encoding='utf-8') as file:
                file.write(repair_content)
            
            status = 'failed'
            reason = 'LLM repair failed, saved original repair'
            print(f"❌ 修复失败: {repair_path} -> {modified_filepath} ({repair_time:.2f}s)")
        
        # 记录处理结果
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
    
    # 保存处理记录
    if repair_records:
        record_file = f"build_repair_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(repair_records, f, indent=2, ensure_ascii=False)
        print(f"📊 修复记录已保存: {record_file}")
    
    # 生成统计摘要
    generate_repair_summary(repair_records, output_dir)
    
    # 移除注释
    # remove_comments_in_lines(os.path.dirname(modified_filepath))
    
    print("所有Dockerfiles处理完成。")

def generate_repair_summary(repair_records, output_dir):
    """生成修复统计摘要"""
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
    
    # 打印摘要
    print(f"\n📊 修复统计摘要:")
    print(f"   总处理文件数: {summary['total_files']}")
    print(f"   成功修复: {summary['successful_repairs']} ({summary['success_rate']}%)")
    print(f"   修复失败: {summary['failed_repairs']}")
    print(f"   跳过修复: {summary['skipped_repairs']}")
    print(f"   平均修复时间: {summary['avg_repair_time']}秒")
    print(f"   总处理时间: {summary['total_processing_time']}秒")
    print(f"   输出目录: {output_dir}")
    
    # 保存摘要
    summary_file = os.path.join(output_dir, "repair_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📄 统计摘要已保存: {summary_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_repair.py unbuild_path [model_name] [--no-think] [--use-openai-api]")
        print("\n参数说明:")
        print("  unbuild_path: 包含构建失败日志的文件路径")
        print("  model_name: 模型名称 (默认: qwen3:32b)")
        print("  --no-think: 启用无思考模式（仅对Qwen有效）")
        print("  --use-openai-api: 使用OpenAI兼容API（百炼）")
        sys.exit(1)
    
    unbuild_path = sys.argv[1]
    
    # 默认模型名称
    # model_name = "qwen3:32b"
    no_think = False
    use_openai_api = False
    
    # 解析参数
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--no-think":
            no_think = True
        elif sys.argv[i] == "--use-openai-api":
            use_openai_api = True
        elif not sys.argv[i].startswith("--"):
            model_name = sys.argv[i]
    
    print(f"🔧 配置信息:")
    print(f"  构建失败日志: {unbuild_path}")
    print(f"  模型名称: {model_name}")
    print(f"  无思考模式: {no_think}")
    print(f"  OpenAI API: {use_openai_api}")
    
    # 执行修复
    process_dockerfiles(unbuild_path, model_name, no_think, use_openai_api)

if __name__ == "__main__":
    main()


# python repair_methods/build_repair.py evaluate_result/star/qwen3_235b_hd_LLM/_unbuild.txt "qwen3-235b-a22b-instruct-2507" --use-openai-api