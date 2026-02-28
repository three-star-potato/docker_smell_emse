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

    time.sleep(1)  # Avoid GitHub API rate limiting
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
        raise  # Re-raise the exception for outer handling

    return False

def write_repositories_with_dockerfile(repo_set, output_file, output_file_without_dockerfile, output_file_failed):
    # Load repositories already confirmed to have Dockerfile
    existing_with_docker = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                existing_with_docker.add(line.strip())
    except FileNotFoundError:
        pass
    
    # Load repositories already confirmed to not have Dockerfile
    existing_without_docker = set()
    try:
        with open(output_file_without_dockerfile, 'r', encoding='utf-8') as f:
            for line in f:
                existing_without_docker.add(line.strip())
    except FileNotFoundError:
        pass
    
    # Load previously failed repositories
    failed_repos = set()
    try:
        with open(output_file_failed, 'r', encoding='utf-8') as f:
            for line in f:
                failed_repos.add(line.strip())
    except FileNotFoundError:
        pass
    
    # Calculate the number of repositories that actually need checking
    repos_to_check = [repo for repo in repo_set 
                     if repo not in existing_with_docker 
                     and repo not in existing_without_docker]
    
    # Add previously failed repositories to the check list
    repos_to_check.extend(failed_repos)
    repos_to_check = list(set(repos_to_check))  # Remove duplicates
    
    total_to_check = len(repos_to_check)
    print(f"\nTotal repositories to check: {total_to_check}")
    print(f"Already confirmed with Dockerfile: {len(existing_with_docker)}")
    print(f"Already confirmed without Dockerfile: {len(existing_without_docker)}")
    print(f"Previously failed checks: {len(failed_repos)}")
    
    # Prepare to write newly discovered repositories
    docker_addresses = set()
    no_docker_addresses = set()
    new_failed_repos = set()

    # Clear the failure file and re-record
    open(output_file_failed, 'w').close()

    with open(output_file, 'a', encoding='utf-8') as f_success, \
         open(output_file_without_dockerfile, 'a', encoding='utf-8') as f_without_dockerfile, \
         open(output_file_failed, 'a', encoding='utf-8') as f_failed:
        
        # Use tqdm to show progress, setting the total number
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

# Main program
print("Starting Dockerfile detection...")
ctf = get_repo_address('dataset_build/star1000+.json')
print(f"\nTotal repositories in input: {len(ctf)}")

output_file = 'dataset_build/star1000+_repos_with_dockerfile.txt'
output_file_without_dockerfile = 'dataset_build/star1000+_repos_without_dockerfile.txt'
output_file_failed = 'dataset_build/star1000+_repos_failed_checks.txt'

# Ensure output directory exists
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