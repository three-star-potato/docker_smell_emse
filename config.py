# config.py
import os


PROJECT_ROOT="/you_path/docker_smell"

PATHS = {
    'project_root': PROJECT_ROOT,
    "has_dockerfile_file":os.path.join(PROJECT_ROOT, 'dataset_build','star1000+_repos_with_dockerfile.txt'),
    'root_folder': os.path.join(PROJECT_ROOT, 'dataset', 'star1000+_context'),
    'dataset_fast_dir': os.path.join(PROJECT_ROOT, 'dataset_fast', 'star1000+_dockerfile'),
    'dataset_fast_build_dir': os.path.join(PROJECT_ROOT, 'dataset_fast_build'),
}


DOCKER_CONFIG = {
    'timeout': 600,  
    'output_timeout': 60,  
    'sleep_after_build': 3,  
    'cleanup_interval': 10,  
}

LOG_CONFIG = {
    'error_log': 'star1000+_dockerfile_unbuild.txt',
    'image_sizes_log': 'star1000+_dockerfile_image_sizes.log', 
    'last_processed_log': 'star1000+_dockerfile_last_build.txt',
}