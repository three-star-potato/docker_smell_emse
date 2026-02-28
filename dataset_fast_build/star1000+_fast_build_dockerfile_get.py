import os
import subprocess
import docker
import time
from tqdm import tqdm
import sys
import shutil
# Add parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import PATHS, DOCKER_CONFIG, LOG_CONFIG

def find_root_dockerfiles(root_folder):
    """Find Dockerfiles only in the root directory"""
    dockerfiles = []
    with open(PATHS["has_dockerfile_file"]) as f:
        for line in f:
            repo_address = line.strip()  # e.g., "4x99/code6"
            if not repo_address:
                continue
                
            # Correct way to join paths
            repo_parts = repo_address.split('/')
            if len(repo_parts) != 2:
                print(f"Warning: Incorrect repository address format {repo_address}")
                continue
                
            username, repo_name = repo_parts
            repo_address_dir = os.path.join(root_folder, username, repo_name)
            if not os.path.exists(repo_address_dir):
                print(f"Error: Directory does not exist {repo_address_dir}")
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
    """Load already processed image records"""
    existing_records = set()
    if os.path.exists(image_sizes_log_file):
        try:
            with open(image_sizes_log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        # Extract Dockerfile path (part before the colon)
                        dockerfile_path = line.split(':', 1)[0].strip()
                        existing_records.add(dockerfile_path)
            print(f"📖 Loaded {len(existing_records)} processed image records")
        except Exception as e:
            print(f"⚠️ Failed to read image sizes log file: {e}")
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
        # Disable BuildKit to get traditional output format
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

        # Set non-blocking read
        import fcntl
        fl = fcntl.fcntl(process.stdout, fcntl.F_GETFL)
        fcntl.fcntl(process.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        while True:
            current_time = time.time()
            
            # Check if the process has already ended
            if process.poll() is not None:
                break

            # Non-blocking read output
            try:
                output = process.stdout.readline()
                if output:
                    output = output.strip()
                    if output and "Step" in output:  # Only print non-empty output
                        print(output)
                        last_output_time = current_time
            except (IOError, OSError):
                # Continue when no data to read
                pass

            # Print status every 10 seconds
            if current_time - last_status_time >= 10:
                elapsed = current_time - start_time
                print(f"\n[Status Check] Running: {elapsed:.1f}s, Image: {docker_name}")
                last_status_time = current_time

            # Timeout check
            if current_time - start_time > timeout:
                process.terminate()
                log_error(f"Build timeout (exceeded {timeout//60} minutes)")
                return False
                
            if current_time - last_output_time > output_timeout:
                process.terminate()
                log_error(f"Build terminated - no output for {output_timeout} seconds")
                return False

            # Brief sleep to avoid high CPU usage
            time.sleep(0.1)

        # Read remaining output
        try:
            remaining_output = process.stdout.read()
            if remaining_output:
                print(remaining_output.strip())
        except (IOError, OSError):
            pass

        # Check final result
        if process.returncode == 0:
            print(f"✅ Image {docker_name} built successfully")
            return True
        else:
            log_error(f"Build failed, exit code: {process.returncode}")
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
    except docker.errors.ImageNotFound:
        print(f"Image {image_name} not found")
        return None
    except docker.errors.APIError as e:
        print(f"API error: {e}")
        return None

def delete_image(image_name):
    """Delete specified image"""
    try:
        result = subprocess.run(["docker", "rmi", image_name], check=True, capture_output=True, text=True)
        print(f"✅ Deleted target image: {image_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to delete target image {image_name}: {e}")
        # Try force delete
        try:
            subprocess.run(["docker", "rmi", "-f", image_name], check=True)
            print(f"✅ Force deleted target image successfully: {image_name}")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ Force delete also failed: {e2}")
            return False

def cleanup_docker_system():
    """Clean Docker system: delete all unused images, containers, networks, etc."""
    try:
        print("🧹 Starting Docker system cleanup...")
        
        # Record disk usage before cleanup
        result_before = subprocess.run(["docker", "system", "df"], capture_output=True, text=True)
        print("Disk usage before cleanup:")
        print(result_before.stdout)
        
        # Clean all unused resources (images, containers, networks, build cache)
        result = subprocess.run(["docker", "system", "prune", "-a", "-f"], 
                              check=True, capture_output=True, text=True)
        
        print("✅ Docker system cleanup completed")
        print("Cleanup output:", result.stdout)
        
        # Record disk usage after cleanup
        result_after = subprocess.run(["docker", "system", "df"], capture_output=True, text=True)
        print("Disk usage after cleanup:")
        print(result_after.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker system cleanup failed: {e}")
        return False

def cleanup_dangling_images():
    """Delete only dangling images (intermediate layers generated during build)"""
    try:
        print("🧹 Cleaning dangling images...")
        result = subprocess.run(["docker", "image", "prune", "-f"], 
                              check=True, capture_output=True, text=True)
        print("✅ Dangling image cleanup completed")
        print("Cleanup output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Dangling image cleanup failed: {e}")
        return False

def get_disk_usage():
    """Get Docker disk usage"""
    try:
        result = subprocess.run(["docker", "system", "df"], capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return "Unable to get disk usage"

def delete_failed_dockerfiles(dataset_fast_dir, image_sizes_log_file):
    """Delete Dockerfiles that failed to build"""
    # Get all Dockerfiles
    print(dataset_fast_dir)
    
    all_dockerfiles = os.listdir(dataset_fast_dir)
    # Get successful Dockerfiles
    success_dockerfiles = set()
    if os.path.exists(image_sizes_log_file):
        with open(image_sizes_log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    dockerfile_path = line.split(':', 1)[0].strip()
                    rel_dockerfile_path = '__'.join(os.path.relpath(dockerfile_path, PATHS['root_folder']).split(os.sep))
                    success_dockerfiles.add(rel_dockerfile_path)
    # print(success_dockerfiles)
    
    # Delete failed Dockerfiles
    deleted_count = 0
    for dockerfile_path in all_dockerfiles:
        if dockerfile_path not in success_dockerfiles:

            print(dockerfile_path)
            try:
                os.remove(os.path.join(dataset_fast_dir, dockerfile_path))
                print(f"🗑️ Deleted failed Dockerfile: {dockerfile_path}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Deletion failed: {dockerfile_path}, error: {e}")
    
    print(f"✅ Deleted {deleted_count} failed Dockerfiles")
    return success_dockerfiles

def main():
    # Create necessary directories
    os.makedirs(PATHS['dataset_fast_dir'], exist_ok=True)
    os.makedirs(PATHS['dataset_fast_build_dir'], exist_ok=True)

    # Build complete log file paths
    error_log_file = os.path.join(PATHS['dataset_fast_build_dir'], LOG_CONFIG['error_log'])
    image_sizes_log_file = os.path.join(PATHS['dataset_fast_build_dir'], LOG_CONFIG['image_sizes_log'])
    last_processed_file = os.path.join(PATHS['dataset_fast_build_dir'], LOG_CONFIG['last_processed_log'])

    print("=" * 60)
    print("Docker Image Batch Build Tool")
    print("=" * 60)
    print(f"Project root directory: {PATHS['project_root']}")
    print(f"Data directory: {PATHS['root_folder']}")
    print(f"Output directory: {PATHS['dataset_fast_dir']}")
    print(f"Repository directory file: {PATHS['has_dockerfile_file']}")
    print(f"Build configuration: Timeout {DOCKER_CONFIG['timeout']}s, Cleanup interval {DOCKER_CONFIG['cleanup_interval']} images")
    print("=" * 60)

    # Find Dockerfiles only in root directory
    dockerfiles = find_root_dockerfiles(PATHS['root_folder'])
    print(f"Found {len(dockerfiles)} root directory Dockerfiles")

    # Load last processing progress and existing image records
    last_processed_path = load_last_processed_file(last_processed_file)
    existing_records = load_existing_image_sizes(image_sizes_log_file)

    if last_processed_path:
        try:
            start_index = sorted(dockerfiles).index(last_processed_path)
            dockerfiles = sorted(dockerfiles)[start_index:]
            print(f"Continuing from last progress, starting index: {start_index}")
        except ValueError:
            print("Last processed file not found, starting from beginning")
            dockerfiles = sorted(dockerfiles)
            start_index = 0
    else:
        dockerfiles = sorted(dockerfiles)
        start_index = 0

    # Show initial disk usage
    print("Initial Docker disk usage:")
    print(get_disk_usage())

    # Process each Dockerfile
    processed_count = 0
    skipped_count = 0
    
    for index, dockerfile_path in enumerate(tqdm(dockerfiles, desc="Building Docker images")):
        try:
            # Check if already processed
            if dockerfile_path in existing_records:
                print(f"⏭️  Skipping already processed image: {dockerfile_path}")
                skipped_count += 1
                
                # Still update processing progress file
                with open(last_processed_file, "w") as f:
                    f.write(dockerfile_path)
                continue
            
            dockerfile_dir = os.path.dirname(dockerfile_path)
            docker_name = f'star-{start_index + index}'
            
            print(f"\n{'='*60}")
            print(f"Processing {index+1}/{len(dockerfiles)}: {docker_name}")
            print(f"Dockerfile path: {dockerfile_path}")
            print(f"Build directory: {dockerfile_dir}")
            print(f"{'='*60}")

            # Build Docker image
            is_build = build_image(docker_name, dockerfile_dir, dockerfile_path, error_log_file)
            if not is_build:
                # Also try to clean dangling images when build fails
                cleanup_dangling_images()
                continue
            
            # Get image size
            image_size = get_image_size(docker_name)
            if image_size is not None:
                print(f"📊 Image size: {image_size} bytes ({image_size/1024/1024:.2f} MB)")
                with open(image_sizes_log_file, 'a') as f:
                    f.write(f"{dockerfile_path}: {image_size}\n")
                
                # Backup Dockerfile
                relative_path = os.path.relpath(dockerfile_path, PATHS['root_folder'])
                safe_filename = relative_path.replace(os.sep, '__')
                target_path = os.path.join(PATHS['dataset_fast_dir'], safe_filename)
                shutil.copy2(dockerfile_path, target_path)
            
            # Wait then delete target image
            time.sleep(DOCKER_CONFIG['sleep_after_build'])
            delete_image(docker_name)
            
            # Clean dangling images (intermediate layers) after each build
            cleanup_dangling_images()
            
            processed_count += 1
            
            # Regular complete system cleanup
            if processed_count % DOCKER_CONFIG['cleanup_interval'] == 0:
                print(f"\n🎯 Processed {processed_count} images, performing complete system cleanup...")
                cleanup_docker_system()
                
        except Exception as e:
            error_message = f"Error processing {dockerfile_path}: {e}"
            print(error_message)
            with open(error_log_file, "a") as f:
                f.write(f"{error_message}\n")
        finally:
            # Update processing progress
            with open(last_processed_file, "w") as f:
                f.write(dockerfile_path)

    # Final cleanup
    print("\n🎉 All image processing completed, performing final cleanup...")
    cleanup_docker_system()

    print(f"\n📊 Processing statistics:")
    print(f"✅ Successfully processed: {processed_count} images")
    print(f"⏭️  Skipped already processed: {skipped_count} images")
    # Add at the beginning of the main function
    dockerfiles_num = delete_failed_dockerfiles(PATHS['dataset_fast_dir'], image_sizes_log_file)
    print(f"📁 Total Dockerfiles: {len(dockerfiles_num)}")
    
    print("\nFinal Docker disk usage:")
    print(get_disk_usage())
    print("🎊 Task completed!")

if __name__ == "__main__":
    main()