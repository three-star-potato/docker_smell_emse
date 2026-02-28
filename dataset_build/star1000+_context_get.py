import os
import json
import subprocess
import time
from tqdm import tqdm
# Used to retrieve projects from filtered addresses

with open('dataset_build/star1000+_repos_with_dockerfile.txt', 'r', encoding='utf-8') as file:
    found_repos = file.readlines()

docker_context = 'dataset/star1000+_context'

# Define multiple Git mirror sources
GIT_MIRRORS = [
    'https://ghproxy.net/https://github.com/',
]

for i, found_repo in enumerate(tqdm(sorted(set(found_repos)))):
    # time.sleep(1)
    try:
        found_repo = found_repo.strip()
        parts = found_repo.split('/')
        username = parts[0]
        repo_name = parts[1]

        # Construct target path
        target_dir = os.path.join(docker_context, username, repo_name)
        if os.path.exists(target_dir):
            print(f'Skipping {found_repo} because {target_dir} already exists')
            continue

        # Use Git to clone to target path
        time.sleep(3)  # Avoid GitHub rate limiting
        # Switch mirror source every 20 repositories
        current_mirror = GIT_MIRRORS[(i // 10) % len(GIT_MIRRORS)]
        repository_url = current_mirror + found_repo.strip()
        new_repository_url = repository_url
        os.environ['GIT_ASKPASS'] = '/bin/true'
        try:
            subprocess.run(['git', 'clone', '--depth', '1', new_repository_url, target_dir], check=True)
            print(f'Successfully cloned {repository_url} to {target_dir}')
        except subprocess.CalledProcessError as e:
            # Open a file to append error messages
            with open('dataset_build/star1000+_git_error.log', 'a') as f:
                f.write(f"Git clone failed with error: {e}\n")

            print(f"Git clone failed with error: {e}")  # Optional: print error message to console
            # subprocess.run(['git', 'clone', new_repository_url, target_dir])

    except Exception as e:
        print(f'Error   {e}')

print('Finished.')