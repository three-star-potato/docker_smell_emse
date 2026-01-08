import os
import subprocess
import json
from tqdm import tqdm
import sys
count=0
def find_dockerfiles(root_folder):
    dockerfiles = []
    
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            dockerfiles.append(os.path.join(root, filename))
    # print(1,dockerfiles)
    
    return dockerfiles


def run_hadolint(dockerfile_path):
    command = f"docker run --rm -i hadolint/hadolint < {dockerfile_path}"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout.strip()
        if not output:
            return []  # 如果没有输出，返回空列表
        issues = [line.strip() for line in output.splitlines() if line.strip()]
        return issues
    except subprocess.CalledProcessError as e:

        issues = [line.strip() for line in e.output.splitlines() if line.strip()]
        if all(line.startswith('-:') for line in issues):
                return issues
        else:
            print(f"Error running command: {command}")
            global count 
            count=count+1
            print(count)
            return issues  # 计算一下有多少格式错误

def analyze_dockerfiles(root_folder):
    dockerfiles = find_dockerfiles(root_folder)
    # print(dockerfiles)
    results = []
    
    for dockerfile in tqdm(dockerfiles):
        issues = run_hadolint(dockerfile)
        result_entry = {
                "dockerfile_path": dockerfile,
                "issues": issues
            }
        results.append(result_entry)
    return results

def write_json_output(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_script.py root_folder output_file")
        sys.exit(1)
    
    root_folder = sys.argv[1]
    output_file = sys.argv[2]
    
    # 检查输出文件是否已存在  
    if os.path.exists(output_file):  
        print(f"The output file '{output_file}' already exists. Skipping analysis and writing.")  
    else:  
        results = analyze_dockerfiles(root_folder)  
        write_json_output(results, output_file)



#  python evaluate/hadolint.py 'dataset_fast/star1000+_dockerfile' 'evaluate_result/dataset_fast_star1000+_dockerfile.json'
# python evaluate/hadolint.py 'repair_result/dataset_fast/star1000+_dockerfile/parfum' 'evaluate_result/dataset_fast_star1000+_dockerfile_parfum.json'
# python evaluate/hadolint.py 'repair_result/dataset_fast/star1000+_dockerfile/dockercleaner' 'evaluate_result/dataset_fast_star1000+_dockerfile_dockercleaner.json'
# python evaluate/hadolint.py 'repair_result/dataset_fast/star1000+_dockerfile/qwen3_235b_hd_LLM' 'evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_LLM.json'
#  python evaluate/hadolint.py 'repair_methods/Distillation/cross_validation_simple/repaired' 'evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_06b_finetune.json'
# python evaluate/hadolint.py 'repair_result/dataset_fast/star1000+_dockerfile/qwen3_8b_hd_LLM_nothink' 'evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_8b_hd_LLM_nothink.json'


# python evaluate/hadolint.py 'repair_result/dataset_fast/star1000+_dockerfile/msr25_icl_qwen3_235b' 'evaluate_result/dataset_fast_star1000+_dockerfile_msr25_icl_qwen3_235b.json'

# python evaluate/hadolint.py 'repair_result/dataset_fast/star1000+_dockerfile/qwen3_235b_hd_tool_LLM' 'evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_tool_LLM.json'

#  python evaluate/hadolint.py 'repair_result/dataset_fast/star1000+_dockerfile/qwen3_32b_hd_LLM_nothink' 'evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_32b_hd_LLM_nothink.json'
# python evaluate/hadolint.py 'repair_result/dataset_fast/star1000+_dockerfile/qwen3_14b_hd_LLM_nothink' 'evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_14b_hd_LLM_nothink.json'
# python evaluate/hadolint.py 'repair_result/dataset_fast/star1000+_dockerfile/dsr1_32b_hd_LLM' 'evaluate_result/dataset_fast_star1000+_dockerfile_dsr1_32b_hd_LLM.json'
# python evaluate/hadolint.py 'repair_result/dataset_fast/star1000+_dockerfile/dsr1_14b_hd_LLM' 'evaluate_result/dataset_fast_star1000+_dockerfile_dsr1_14b_hd_LLM.json'