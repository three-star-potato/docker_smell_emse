import os
import subprocess
import fnmatch
import docker
import time
from tqdm import tqdm
import sys
import os
import shutil
import fcntl
import errno
import select
import argparse
# 添加上级目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import PATHS, LOG_CONFIG



def load_last_processed_file(last_processed_file):
    try:
        with open(last_processed_file, "r") as f:
            last_processed_path = f.read().strip()
        return last_processed_path
    except FileNotFoundError:
        return None



def build_image(docker_name, dockerfile_path, context_path, error_log_file):
    os.chdir(context_path)
    start_time = time.time()
    last_status_time = start_time
    timeout = 600  # 10 minutes total timeout
    output_timeout = 600  # 10 minutes timeout if step pause
    last_output_time = time.time()
    # 记录最后一步的输出
    last_output_line = ""
    def log_error(message,last_output_line):
        print(message)
        with open(error_log_file, "a") as f:
            f.write(f"<phase>{last_output_line}<phase>: {message}\n")

    temp_dir = os.path.dirname(error_log_file)  # 获取日志文件所在目录
    temp_dockerfile_path = os.path.join(temp_dir, "dockerfile")
    
    with open(dockerfile_path, 'r') as src, open(temp_dockerfile_path, 'w') as dst:
        for line in src:
            # # 处理GitHub URL替换
            # if 'https://github.com' in line:
            #     dst.write(line.replace(
            #         'https://github.com',
            #         'https://gh-proxy.com/https://github.com/'
            #     ))
            # # 在npm install命令之前添加registry设置
            # elif any(cmd in line for cmd in ['npm', 'yarn install', 'pnpm install']):
            #     # 检查是否已经安装了Node.js（通过查找之前的Node.js安装步骤）
            #     dst.write('# 设置npm镜像源以避免超时\n')
            #     dst.write('RUN npm config set registry https://registry.npmmirror.com/\n')
            #     dst.write(line)
            # else:
            #     dst.write(line)
            dst.write(line)
        
    try:
        # Disable BuildKit to get traditional output format
        env = os.environ.copy()
        env["DOCKER_BUILDKIT"] = "0"
        
        process = subprocess.Popen(
                        [
                "docker", "build",
                "-t", docker_name,
                "-f", temp_dockerfile_path,  # Dockerfile 的绝对路径
                "."
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env
        )


        while True:
            
            # 非阻塞读取实现
            rlist, _, _ = select.select([process.stdout], [], [], 0.1)  # 0.1秒超时
            if process.stdout in rlist:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    output = output.strip()
                    print(output) 
                    if "Step" in output:
                    # if  output:
                        print(output)  # Real-time printing
                        last_output_line=output
                        last_output_time = time.time()
           
            # Print status every 10 seconds (without affecting real-time output)
            current_time = time.time()
            if current_time - last_status_time >= 10:
                elapsed = current_time - start_time
                print(f"\n[Status Check] Elapsed: {elapsed:.1f}s | Last output: {current_time - last_output_time:.1f}s ago")
                last_status_time = current_time

            # Timeout check
            if current_time - start_time > timeout:
                process.terminate()
                log_error(f"<path>{dockerfile_path}<path><error>build timed out (exceeded {timeout//60} minutes)<error>",last_output_line)
                return False
                
            if current_time - last_output_time > output_timeout:
                process.terminate()
                log_error(f"<path>{dockerfile_path}<path><error>build terminated - no output for {output_timeout} seconds<error>",last_output_line)
                return False
            # 检查最终结果
        if process.returncode == 0:
            print(f"Image {docker_name} built successfully")
            return True
        else:
            log_error(f"<path>{dockerfile_path}<path><error>build failed with exit code: {process.returncode}<error>",last_output_line)
            return False

    except Exception as ex:
        log_error(f"Build exception: {str(ex)}")
        return False
def get_image_size(image_name):
    client = docker.from_env()
    try:
        image = client.images.get(image_name)
        size = image.attrs['Size']
        return size
    except docker.errors.ImageNotFound as e:
        print(f"Image {image_name} not found: {e}")
        return None
    except docker.errors.APIError as e:
        print(f"API error occurred: {e}")
        return None

def delete_image(image_name):
    try:
        subprocess.run(["docker", "rmi", image_name], check=True)
        print(f"Deleted image: {image_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while deleting image {image_name}: {e}")


def process_dockerfiles(project_path, file_folder, evaluate_result_path, context_folder,image_name):
    # 确保输出目录存在
    full_evaluate_path = os.path.join(project_path, evaluate_result_path)
    os.makedirs(full_evaluate_path, exist_ok=True)
    print(full_evaluate_path)

    # 设置日志文件路径
    error_log_file = os.path.join(project_path, evaluate_result_path + '_unbuild.txt')
    image_sizes_log_file = os.path.join(project_path, evaluate_result_path + '_image_sizes.log')
    last_processed_file = os.path.join(project_path, evaluate_result_path + '_last_build.txt')

    # 获取Dockerfile列表
    dockerfiles = sorted(set(os.listdir(os.path.join(project_path, file_folder))))
    print(f"Found {len(dockerfiles)} Dockerfiles to process")
    print(dockerfiles)

    # 加载上次处理的位置
    last_processed_path = load_last_processed_file(last_processed_file)
    start_index = 0
    
    if last_processed_path:
        try:
            start_index = dockerfiles.index(last_processed_path)
            dockerfiles = dockerfiles[start_index:]
        except ValueError:
            print(f"Last processed file not found, starting from beginning")
            start_index = 0

    print(f"Resuming from index: {start_index}")

    exist_dockerfile_images=set()
    if os.path.exists(image_sizes_log_file):
        with open(image_sizes_log_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                        # 提取Dockerfile路径
                exist_dockerfile_image = line.split(':')[0].strip()
                exist_dockerfile_images.add(exist_dockerfile_image)
    for index, dockerfile_name in enumerate(tqdm(dockerfiles)):
        try:
            # 构建完整路径

            dockerfile_path = os.path.join(project_path, file_folder, dockerfile_name)
            print(dockerfile_name,123)
            if dockerfile_name in exist_dockerfile_images:
                print(dockerfile_name)
                continue
            context_path = os.path.join(project_path, context_folder, 
                                      os.path.dirname(dockerfile_name.replace("__", "/"))+"/",)
            
            docker_name = f'{image_name}-{start_index + index}'
            print(f"\nProcessing {docker_name}:")
            print(f"Dockerfile: {dockerfile_path}")
            print(f"Context: {context_path}")

            # 构建镜像
            if not build_image(docker_name, dockerfile_path, context_path, error_log_file):
                continue

            # 记录镜像大小
            image_size = get_image_size(docker_name)
            if image_size:
                with open(image_sizes_log_file, "a") as f:
                    f.write(f"{dockerfile_name}: {image_size}\n")

            time.sleep(3)  # 间隔防止资源冲突
            
            # 清理镜像
            delete_image(docker_name)

        except Exception as e:
            error_msg = f"Error processing {dockerfile_name}: {str(e)}"
            print(error_msg)
            with open(error_log_file, "a") as f:
                f.write(f"{time.ctime()}: {error_msg}\n")
        finally:
            # 更新最后处理记录
            with open(last_processed_file, "w") as f:
                f.write(dockerfile_name)
    print("\n[状态] 所有镜像构建完成，开始清理悬空镜像...")
    subprocess.run(["docker", "image", "prune", "-f"], check=True)
    print("悬空镜像清理完成！")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量构建 Docker 镜像")
    
    # 定义位置参数（按顺序解析）
    parser.add_argument("file_folder", help="Dockerfile 存储目录（如 dataset_fast/star1000+_dockerfile/）")
    parser.add_argument("evaluate_result_path", help="评估结果目录（如 evaluate_result/ctf/dockerfile/）")
    parser.add_argument("context_folder", help="构建上下文目录（如 dataset_fast/star1000+_context/）")
    parser.add_argument("image_name", help="构建镜像名称（如 star或ctf）")
    
    args = parser.parse_args()
    
    process_dockerfiles(
        project_path=PATHS["project_root"],
        file_folder=args.file_folder,
        evaluate_result_path=args.evaluate_result_path,
        context_folder=args.context_folder,
        image_name=args.image_name
    )


# python evaluate/build.py "repair_result/dataset_fast/star1000+_dockerfile/parfum" "evaluate_result/star/parfum/" "dataset_fast/star1000+_context/" "star"
# python evaluate/build.py "repair_result/dataset_fast/star1000+_dockerfile/dockercleaner" "evaluate_result/star/dockercleaner/" "dataset_fast/star1000+_context/" "star"
# python evaluate/build.py "repair_result/dataset_fast/star1000+_dockerfile/msr25_icl_qwen3_235b/" "evaluate_result/star/msr25_icl_qwen3_235b/" "dataset_fast/star1000+_context/" "star"
# python evaluate/build.py "repair_result/dataset_fast/star1000+_dockerfile/qwen3_235b_hd_LLM/" "evaluate_result/star/qwen3_235b_hd_LLM/" "dataset_fast/star1000+_context/" "star"



# python evaluate/build.py "build_repair_result/dataset_fast/star1000+_dockerfile/parfum" "evaluate_result/star/parfum_correct/" "dataset_fast/star1000+_context/" "star"
# python evaluate/build.py "build_repair_result/dataset_fast/star1000+_dockerfile/qwen3_235b_hd_LLM" "evaluate_result/star/qwen3_235b_hd_LLM_correct/" "dataset_fast/star1000+_context/" "star"







