import os
import subprocess
import docker
import time
from tqdm import tqdm
import sys
import shutil
# 添加上级目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import PATHS, DOCKER_CONFIG, LOG_CONFIG

def find_root_dockerfiles(root_folder):
    """只查找根目录下的Dockerfile"""
    dockerfiles = []
    with open(PATHS["has_dockerfile_file"]) as f:
        for line in f:
            repo_address = line.strip()  # 例如 "4x99/code6"
            if not repo_address:
                continue
                
            # 正确的路径拼接方式
            repo_parts = repo_address.split('/')
            if len(repo_parts) != 2:
                print(f"警告：仓库地址格式不正确 {repo_address}")
                continue
                
            username, repo_name = repo_parts
            repo_address_dir = os.path.join(root_folder, username, repo_name)
            if not os.path.exists(repo_address_dir):
                print(f"错误：目录不存在 {repo_address_dir}")
            else:
                for filename in os.listdir(repo_address_dir):
                    if filename.lower() == 'dockerfile':
                        dockerfile_path = os.path.join(repo_address_dir, filename)
                        if os.path.isfile(dockerfile_path):
                            dockerfiles.append(dockerfile_path)
    
    return dockerfiles

def load_last_processed_file(last_processed_file):
    try:
        with open(last_processed_file, "r") as f:
            last_processed_path = f.read().strip()
        return last_processed_path
    except FileNotFoundError:
        return None

def load_existing_image_sizes(image_sizes_log_file):
    """加载已处理的镜像记录"""
    existing_records = set()
    if os.path.exists(image_sizes_log_file):
        try:
            with open(image_sizes_log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        # 提取Dockerfile路径（冒号前的部分）
                        dockerfile_path = line.split(':', 1)[0].strip()
                        existing_records.add(dockerfile_path)
            print(f"📖 已加载 {len(existing_records)} 个已处理的镜像记录")
        except Exception as e:
            print(f"⚠️ 读取镜像大小日志文件失败: {e}")
    return existing_records

def build_image(docker_name, directory, build_docker_path, error_log_file):
    os.chdir(directory)
    start_time = time.time()
    last_status_time = start_time
    timeout = DOCKER_CONFIG['timeout']
    output_timeout = DOCKER_CONFIG['output_timeout']
    last_output_time = time.time()

    def log_error(message):
        print(message)
        with open(error_log_file, "a") as f:
            f.write(f"{time.ctime()}: {message}\n")

    try:
        # 禁用BuildKit以获取传统输出格式
        env = os.environ.copy()
        env["DOCKER_BUILDKIT"] = "0"
        
        process = subprocess.Popen(
            ["docker", "build", "-t", docker_name, "-f", build_docker_path, "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        # 设置非阻塞读取
        import fcntl
        fl = fcntl.fcntl(process.stdout, fcntl.F_GETFL)
        fcntl.fcntl(process.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        while True:
            current_time = time.time()
            
            # 检查进程是否已经结束
            if process.poll() is not None:
                break

            # 非阻塞读取输出
            try:
                output = process.stdout.readline()
                if output:
                    output = output.strip()
                    if output and"Step" in output:  # 只打印非空输出
                        print(output)
                        last_output_time = current_time
            except (IOError, OSError):
                # 没有数据可读时继续
                pass

            # 每10秒打印状态
            if current_time - last_status_time >= 10:
                elapsed = current_time - start_time
                print(f"\n[状态检查] 已运行: {elapsed:.1f}s, 镜像: {docker_name}")
                last_status_time = current_time

            # 超时检查
            if current_time - start_time > timeout:
                process.terminate()
                log_error(f"构建超时（超过{timeout//60}分钟）")
                return False
                
            if current_time - last_output_time > output_timeout:
                process.terminate()
                log_error(f"构建终止 - {output_timeout}秒无输出")
                return False

            # 短暂休眠避免CPU占用过高
            time.sleep(0.1)

        # 读取剩余的输出
        try:
            remaining_output = process.stdout.read()
            if remaining_output:
                print(remaining_output.strip())
        except (IOError, OSError):
            pass

        # 检查最终结果
        if process.returncode == 0:
            print(f"✅ 镜像 {docker_name} 构建成功")
            return True
        else:
            log_error(f"构建失败，退出码: {process.returncode}")
            return False

    except Exception as ex:
        log_error(f"构建异常: {str(ex)}")
        return False

def get_image_size(image_name):
    client = docker.from_env()
    try:
        image = client.images.get(image_name)
        size = image.attrs['Size']
        return size
    except docker.errors.ImageNotFound:
        print(f"镜像 {image_name} 未找到")
        return None
    except docker.errors.APIError as e:
        print(f"API错误: {e}")
        return None

def delete_image(image_name):
    """删除指定镜像"""
    try:
        result = subprocess.run(["docker", "rmi", image_name], check=True, capture_output=True, text=True)
        print(f"✅ 已删除目标镜像: {image_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 删除目标镜像 {image_name} 失败: {e}")
        # 尝试强制删除
        try:
            subprocess.run(["docker", "rmi", "-f", image_name], check=True)
            print(f"✅ 强制删除目标镜像成功: {image_name}")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ 强制删除目标镜像也失败: {e2}")
            return False

def cleanup_docker_system():
    """清理Docker系统：删除所有未使用的镜像、容器、网络等"""
    try:
        print("🧹 开始清理Docker系统...")
        
        # 记录清理前的磁盘使用情况
        result_before = subprocess.run(["docker", "system", "df"], capture_output=True, text=True)
        print("清理前磁盘使用情况:")
        print(result_before.stdout)
        
        # 清理所有未使用的资源（镜像、容器、网络、构建缓存）
        result = subprocess.run(["docker", "system", "prune", "-a", "-f"], 
                              check=True, capture_output=True, text=True)
        
        print("✅ Docker系统清理完成")
        print("清理输出:", result.stdout)
        
        # 记录清理后的磁盘使用情况
        result_after = subprocess.run(["docker", "system", "df"], capture_output=True, text=True)
        print("清理后磁盘使用情况:")
        print(result_after.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker系统清理失败: {e}")
        return False

def cleanup_dangling_images():
    """只删除悬虚镜像（构建过程中产生的中间层）"""
    try:
        print("🧹 清理悬虚镜像...")
        result = subprocess.run(["docker", "image", "prune", "-f"], 
                              check=True, capture_output=True, text=True)
        print("✅ 悬虚镜像清理完成")
        print("清理输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 悬虚镜像清理失败: {e}")
        return False

def get_disk_usage():
    """获取Docker磁盘使用情况"""
    try:
        result = subprocess.run(["docker", "system", "df"], capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return "无法获取磁盘使用情况"

def delete_failed_dockerfiles(dataset_fast_dir, image_sizes_log_file):
    """删除构建失败的Dockerfile"""
    # 获取所有Dockerfile
    print(dataset_fast_dir)
    
    all_dockerfiles = os.listdir(dataset_fast_dir)
    # 获取成功的Dockerfile
    success_dockerfiles = set()
    if os.path.exists(image_sizes_log_file):
        with open(image_sizes_log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    dockerfile_path = line.split(':', 1)[0].strip()
                    rel_dockerfile_path='__'.join(os.path.relpath(dockerfile_path, PATHS['root_folder']).split(os.sep))
                    success_dockerfiles.add(rel_dockerfile_path)
    # print(success_dockerfiles)
    
    # 删除失败的Dockerfile
    deleted_count = 0
    for dockerfile_path in all_dockerfiles:
        if dockerfile_path not in success_dockerfiles:

            print(dockerfile_path)
            try:
                os.remove(os.path.join(dataset_fast_dir,dockerfile_path))
                print(f"🗑️ 删除失败的Dockerfile: {dockerfile_path}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ 删除失败: {dockerfile_path}, 错误: {e}")
    
    print(f"✅ 已删除 {deleted_count} 个构建失败的Dockerfile")
    return success_dockerfiles

def main():
    # 创建必要的目录
    os.makedirs(PATHS['dataset_fast_dir'], exist_ok=True)
    os.makedirs(PATHS['dataset_fast_build_dir'], exist_ok=True)

    # 构建完整的日志文件路径
    error_log_file = os.path.join(PATHS['dataset_fast_build_dir'], LOG_CONFIG['error_log'])
    image_sizes_log_file = os.path.join(PATHS['dataset_fast_build_dir'], LOG_CONFIG['image_sizes_log'])
    last_processed_file = os.path.join(PATHS['dataset_fast_build_dir'], LOG_CONFIG['last_processed_log'])

    print("=" * 60)
    print("Docker镜像批量构建工具")
    print("=" * 60)
    print(f"项目根目录: {PATHS['project_root']}")
    print(f"数据目录: {PATHS['root_folder']}")
    print(f"输出目录: {PATHS['dataset_fast_dir']}")
    print(f"仓库目录文件: {PATHS['has_dockerfile_file']}")
    print(f"构建配置: 超时{DOCKER_CONFIG['timeout']}秒, 清理间隔{DOCKER_CONFIG['cleanup_interval']}个镜像")
    print("=" * 60)

    # 只查找根目录下的Dockerfile
    dockerfiles = find_root_dockerfiles(PATHS['root_folder'])
    print(f"找到 {len(dockerfiles)} 个根目录Dockerfile")

    # 加载上次的处理进度和已处理的镜像记录
    last_processed_path = load_last_processed_file(last_processed_file)
    existing_records = load_existing_image_sizes(image_sizes_log_file)

    if last_processed_path:
        try:
            start_index = sorted(dockerfiles).index(last_processed_path)
            dockerfiles = sorted(dockerfiles)[start_index:]
            print(f"从上次的进度继续，开始索引: {start_index}")
        except ValueError:
            print("上次处理的文件未找到，从头开始")
            dockerfiles = sorted(dockerfiles)
            start_index = 0
    else:
        dockerfiles = sorted(dockerfiles)
        start_index = 0

    # 显示初始磁盘使用情况
    print("初始Docker磁盘使用情况:")
    print(get_disk_usage())

    # 处理每个Dockerfile
    processed_count = 0
    skipped_count = 0
    
    for index, dockerfile_path in enumerate(tqdm(dockerfiles, desc="构建Docker镜像")):
        try:
            # 检查是否已经处理过
            if dockerfile_path in existing_records:
                print(f"⏭️  跳过已处理的镜像: {dockerfile_path}")
                skipped_count += 1
                
                # 仍然更新处理进度文件
                with open(last_processed_file, "w") as f:
                    f.write(dockerfile_path)
                continue
            
            dockerfile_dir = os.path.dirname(dockerfile_path)
            docker_name = f'star-{start_index + index}'
            
            print(f"\n{'='*60}")
            print(f"处理 {index+1}/{len(dockerfiles)}: {docker_name}")
            print(f"Dockerfile路径: {dockerfile_path}")
            print(f"构建目录: {dockerfile_dir}")
            print(f"{'='*60}")

            # 构建Docker镜像
            is_build = build_image(docker_name, dockerfile_dir, dockerfile_path, error_log_file)
            if not is_build:
                # 构建失败时也尝试清理悬虚镜像
                cleanup_dangling_images()
                continue
            
            # 获取镜像大小
            image_size = get_image_size(docker_name)
            if image_size is not None:
                print(f"📊 镜像大小: {image_size} bytes ({image_size/1024/1024:.2f} MB)")
                with open(image_sizes_log_file, 'a') as f:
                    f.write(f"{dockerfile_path}: {image_size}\n")
                
                # 备份Dockerfile
                relative_path = os.path.relpath(dockerfile_path, PATHS['root_folder'])
                safe_filename = relative_path.replace(os.sep, '__')
                target_path = os.path.join(PATHS['dataset_fast_dir'], safe_filename)
                shutil.copy2(dockerfile_path, target_path)
            
            # 等待后删除目标镜像
            time.sleep(DOCKER_CONFIG['sleep_after_build'])
            delete_image(docker_name)
            
            # 每次构建后都清理悬虚镜像（中间层）
            cleanup_dangling_images()
            
            processed_count += 1
            
            # 定期完整系统清理
            if processed_count % DOCKER_CONFIG['cleanup_interval'] == 0:
                print(f"\n🎯 已处理 {processed_count} 个镜像，进行完整系统清理...")
                cleanup_docker_system()
                
        except Exception as e:
            error_message = f"处理 {dockerfile_path} 时出错: {e}"
            print(error_message)
            with open(error_log_file, "a") as f:
                f.write(f"{error_message}\n")
        finally:
            # 更新处理进度
            with open(last_processed_file, "w") as f:
                f.write(dockerfile_path)

    # 最终清理
    print("\n🎉 所有镜像处理完成，进行最终清理...")
    cleanup_docker_system()

    print(f"\n📊 处理统计:")
    print(f"✅ 成功处理: {processed_count} 个镜像")
    print(f"⏭️  跳过已处理: {skipped_count} 个镜像")
    # 在main函数开始处添加
    dockerfiles_num=delete_failed_dockerfiles(PATHS['dataset_fast_dir'], image_sizes_log_file)
    print(f"📁 总计Dockerfile: {len(dockerfiles_num)} 个")
    
    print("\n最终Docker磁盘使用情况:")
    print(get_disk_usage())
    print("🎊 任务完成！")

if __name__ == "__main__":
    main()