import json
import requests
from tqdm import tqdm
import time
import os

def get_repo_address(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    repo_address = set()
    for item in data['items']:
        repo_address.add(item['name'])
    return repo_address

def check_for_dockerfile_recursive(repo_owner, repo_name, path="", depth=0):
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}'
    access_token = 'github_api'
    headers = {
        'Authorization': f'token {access_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    time.sleep(1)  # 避免GitHub API速率限制
    print(url)

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        content = response.json()

        for item in content:
            if item['type'] == 'file' and item['name'].lower() == 'dockerfile':
                return True
            elif item['type'] == 'dir' and depth > 0:
                if check_for_dockerfile_recursive(repo_owner, repo_name, item['path'], depth - 1):
                    return True

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        raise  # 重新抛出异常，让外层处理

    return False

def write_repositories_with_dockerfile(repo_set, output_file, output_file_without_dockerfile, output_file_failed):
    # 加载已经确认有Dockerfile的仓库
    existing_with_docker = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                existing_with_docker.add(line.strip())
    except FileNotFoundError:
        pass
    
    # 加载已经确认没有Dockerfile的仓库
    existing_without_docker = set()
    try:
        with open(output_file_without_dockerfile, 'r', encoding='utf-8') as f:
            for line in f:
                existing_without_docker.add(line.strip())
    except FileNotFoundError:
        pass
    
    # 加载之前失败的仓库
    failed_repos = set()
    try:
        with open(output_file_failed, 'r', encoding='utf-8') as f:
            for line in f:
                failed_repos.add(line.strip())
    except FileNotFoundError:
        pass
    
    # 计算需要实际检查的仓库数量
    repos_to_check = [repo for repo in repo_set 
                     if repo not in existing_with_docker 
                     and repo not in existing_without_docker]
    
    # 添加之前失败的仓库到检查列表
    repos_to_check.extend(failed_repos)
    repos_to_check = list(set(repos_to_check))  # 去重
    
    total_to_check = len(repos_to_check)
    print(f"\nTotal repositories to check: {total_to_check}")
    print(f"Already confirmed with Dockerfile: {len(existing_with_docker)}")
    print(f"Already confirmed without Dockerfile: {len(existing_without_docker)}")
    print(f"Previously failed checks: {len(failed_repos)}")
    
    # 准备写入新发现的仓库
    docker_addresses = set()
    no_docker_addresses = set()
    new_failed_repos = set()

    # 清空失败文件，重新记录
    open(output_file_failed, 'w').close()

    with open(output_file, 'a', encoding='utf-8') as f_success, \
         open(output_file_without_dockerfile, 'a', encoding='utf-8') as f_without_dockerfile, \
         open(output_file_failed, 'a', encoding='utf-8') as f_failed:
        
        # 使用tqdm显示进度，并设置总数
        progress_bar = tqdm(repos_to_check, desc="Checking repositories", unit="repo")
        for repo in progress_bar:
            try:
                owner, name = repo.split('/')
                if check_for_dockerfile_recursive(owner, name):
                    f_success.write(repo + '\n')
                    f_success.flush()
                    docker_addresses.add(repo)
                    progress_bar.set_postfix({
                        'Found': len(docker_addresses), 
                        'Not found': len(no_docker_addresses),
                        'Failed': len(new_failed_repos)
                    })
                else:
                    f_without_dockerfile.write(repo + '\n')
                    f_without_dockerfile.flush()
                    no_docker_addresses.add(repo)
                    progress_bar.set_postfix({
                        'Found': len(docker_addresses), 
                        'Not found': len(no_docker_addresses),
                        'Failed': len(new_failed_repos)
                    })
            except Exception as e:
                f_failed.write(repo + '\n')
                f_failed.flush()
                new_failed_repos.add(repo)
                progress_bar.set_postfix({
                    'Found': len(docker_addresses), 
                    'Not found': len(no_docker_addresses),
                    'Failed': len(new_failed_repos)
                })
                print(f"\nFailed to check {repo}: {str(e)}")
    
    return docker_addresses, no_docker_addresses, new_failed_repos

# 主程序
print("Starting Dockerfile detection...")
ctf = get_repo_address('dataset_build/star1000+.json')
print(f"\nTotal repositories in input: {len(ctf)}")

output_file = 'dataset_build/star1000+_repos_with_dockerfile.txt'
output_file_without_dockerfile = 'dataset_build/star1000+_repos_without_dockerfile.txt'
output_file_failed = 'dataset_build/star1000+_repos_failed_checks.txt'

# 确保输出目录存在
os.makedirs('dataset_build', exist_ok=True)

docker_addresses, no_docker_addresses, failed_repos = write_repositories_with_dockerfile(
    ctf, output_file, output_file_without_dockerfile, output_file_failed
)

print(f"\nSummary:")
print(f"New repositories with Dockerfile found: {len(docker_addresses)}")
print(f"New repositories without Dockerfile: {len(no_docker_addresses)}")
print(f"Repositories with API check failures: {len(failed_repos)}")
print(f"Already confirmed with Dockerfile: {len(ctf) - len(docker_addresses) - len(no_docker_addresses) - len(failed_repos)}")
print(f"\nRepositories with Dockerfile written to {output_file}")
print(f"Repositories without Dockerfile written to {output_file_without_dockerfile}")
print(f"Repositories with check failures written to {output_file_failed}")