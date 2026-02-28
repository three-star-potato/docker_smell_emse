import os
import shutil
from pathlib import Path
import sys
# Add parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import PATHS, LOG_CONFIG


# Configuration parameters
input_log = os.path.join(PATHS['dataset_fast_build_dir'], LOG_CONFIG['image_sizes_log'])
src_base = PATHS['root_folder']
dst_base = os.path.join(PATHS['project_root'], 'dataset_fast', 'star1000+_context')

def extract_repo_names(log_file):
    """Extract repository names from log file"""
    repo_names = set()
    base_path = PATHS['root_folder']
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            # Extract Dockerfile path
            dockerfile_path = line.split(':')[0].strip()
            # Ensure path starts with base_path
            if dockerfile_path.startswith(base_path):
                # Get relative path and split to get two-level directory
                rel_path = os.path.relpath(dockerfile_path, base_path)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    repo_name = f"{parts[0]}/{parts[1]}"
                    repo_names.add(repo_name)
    return sorted(repo_names)

def safe_copy(src, dst):
    """Safely copy files, skip non-existent files or links"""
    try:
        if os.path.lexists(src):  # Check if file or link exists
            if os.path.islink(src):  # If symbolic link, copy the link itself
                linkto = os.readlink(src)
                # If target file already exists, delete it first
                if os.path.exists(dst) or os.path.islink(dst):
                    os.remove(dst)
                os.symlink(linkto, dst)
            else:
                shutil.copy2(src, dst)
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"Warning: Skipped {src} (reason: {str(e)})")

def copy_repo(src, dst):
    """Recursively copy directory, skip missing files and .git"""
    if not os.path.exists(src):
        print(f"Warning: Source path does not exist - {src}")
        return
        
    os.makedirs(dst, exist_ok=True)
    
    try:
        for item in os.listdir(src):
            src_path = os.path.join(src, item)
            dst_path = os.path.join(dst, item)
            
            # Skip .git directory
            if item == '.git':
                continue
                
            # Skip some common large or unnecessary files
            if item in ['__pycache__', '.pytest_cache', 'node_modules', '.DS_Store']:
                continue
                
            if os.path.isdir(src_path):
                copy_repo(src_path, dst_path)
            else:
                safe_copy(src_path, dst_path)
    except PermissionError as e:
        print(f"Permission denied accessing {src}: {e}")


def copy_repos_from_log():
    """Main function: Extract repositories from log file and copy to two locations"""
    # Check if log file exists
    if not os.path.exists(input_log):
        print(f"Error: Log file not found - {input_log}")
        return
    
    # Extract repository names from log file
    print("Extracting repository names from log file...")
    repos = extract_repo_names(input_log)
    print(f"Found {len(repos)} repositories in log file")
    
    if not repos:
        print("No repositories found to copy.")
        return
    
    # Create target directory
    Path(dst_base).mkdir(parents=True, exist_ok=True)
    
    # Ensure adalflow directory exists
    adalflow_repos_dir = os.path.join(PATHS['adalflow_root'], 'repos')
    Path(adalflow_repos_dir).mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    adalflow_copied_count = 0
    adalflow_skipped_count = 0
    
    print("Starting to copy repositories...")
    for i, repo_name in enumerate(repos, 1):
        src_repo = os.path.join(src_base, repo_name)
        dst_repo = os.path.join(dst_base, repo_name)
        
        if os.path.exists(src_repo):
            print(f"[{i}/{len(repos)}] Copying {repo_name}")
            
            # Copy to main target directory
            try:
                copy_repo(src_repo, dst_repo)
                copied_count += 1
            except Exception as e:
                print(f"Error copying to main destination {src_repo}: {e}")
                skipped_count += 1
    
        else:
            print(f"[{i}/{len(repos)}] Warning: Source repo not found - {src_repo}")
            skipped_count += 1
            adalflow_skipped_count += 1
    
    print(f"\nCopy completed!")
    print("=" * 50)
    print("Main destination summary:")
    print(f"  Successfully copied: {copied_count} repositories")
    print(f"  Skipped (not found/error): {skipped_count} repositories")
    print("=" * 50)
    print("Adalflow destination summary:")
    print(f"  Successfully copied: {adalflow_copied_count} repositories")
    print(f"  Skipped (not found/error): {adalflow_skipped_count} repositories")
    print("=" * 50)
    print(f"Total processed: {len(repos)} repositories")
    print(f"Main destination: {dst_base}")
    print(f"Adalflow destination: {adalflow_repos_dir}")

if __name__ == "__main__":
    copy_repos_from_log()