import os
import json
import subprocess
import time
from tqdm import tqdm
#用于获取筛选后的地址的项目

with open('dataset_build/star1000+_repos_with_dockerfile.txt','r',encoding='utf-8') as file:
     found_repos = file.readlines()


docker_context = 'dataset/star1000+_context'

# 定义多个Git加速源
GIT_MIRRORS = [
    'https://ghproxy.net/https://github.com/',

]

for i, found_repo in enumerate(tqdm(sorted(set(found_repos)))):
    # time.sleep(1)
    try:
        found_repo=found_repo.strip()
        parts = found_repo.split('/')
        username = parts[0]
        repo_name = parts[1]
    
    # 构建目标路径
        target_dir = os.path.join(docker_context, username, repo_name)
        if os.path.exists(target_dir):
            print(f'Skipping {found_repo} because {target_dir} already exists')
            continue

    # 使用Git克隆到目标路径
        time.sleep(3)  # 避免GitHub 限制
        # 每20个仓库切换一次加速源
        current_mirror = GIT_MIRRORS[(i // 10) % len(GIT_MIRRORS)]
        repository_url = current_mirror + found_repo.strip()
        # repository_url='https://js-github.pages.dev/https://github.com/'+found_repo.strip()#这个是使用的文件加速器，后续要删掉
        new_repository_url=repository_url
        os.environ['GIT_ASKPASS'] = '/bin/true' 
        try:
            subprocess.run(['git', 'clone', '--depth', '1', new_repository_url, target_dir], check=True)
            print(f'Successfully cloned {repository_url} to {target_dir}')
        except subprocess.CalledProcessError as e:
    # 打开一个文件以追加模式写入错误信息
            with open('dataset_build/star1000+_git_error.log', 'a') as f:
                f.write(f"Git clone failed with error: {e}\n")
            
            print(f"Git clone failed with error: {e}")  # 可选：在控制台打印错误信息
                # subprocess.run(['git', 'clone', new_repository_url, target_dir])

    except Exception as e:
        print(f'Error   {e}')
    

print(' finished.')