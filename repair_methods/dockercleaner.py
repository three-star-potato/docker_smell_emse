import os
import shutil
from tqdm import tqdm
import subprocess
import time
import json
from datetime import datetime

def find_dockerfiles(root_folder):
    """Find all Dockerfiles"""
    dockerfiles = []
    for root, _, files in os.walk(root_folder):
        for filename in files:
            if "dockerfile" in filename.lower() or filename.lower().endswith("dockerfile"):
                dockerfiles.append(os.path.join(root, filename))
    return dockerfiles

def process_dockerfiles(input_root, output_root, time_log_file=None):
    """Process Dockerfiles and record time"""
    dockerfiles = find_dockerfiles(input_root)
    clean_dir = os.path.join(output_root, input_root, 'dockercleaner')

    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
    
    # Time record data structure
    time_records = []

    for dockerfile in tqdm(sorted(dockerfiles), desc="Processing Dockerfiles"):
        dockerclean_filepath = dockerfile.replace(input_root, clean_dir)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(dockerclean_filepath), exist_ok=True)
        
        # If file already exists, skip
        if os.path.exists(dockerclean_filepath):
            print(f"Skipping existing file: {dockerclean_filepath}")
            continue

        # Record start time
        start_time = time.time()
        
        fix_flag = "--fix" 
        command = f"~/.local/bin/dockercleaner -i {dockerfile} -o {dockerclean_filepath} {fix_flag}"

        try:
            result = subprocess.run(command, shell=True, check=True, 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  timeout=300)  # 5 minute timeout
            
            # Record end time
            end_time = time.time()
            repair_time = end_time - start_time
            
            # Record success information
            time_record = {
                'dockerfile': dockerfile,
                'repaired_file': dockerclean_filepath,
                'repair_time_seconds': round(repair_time, 2),
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'command': command
            }
            time_records.append(time_record)
            
            print(f"✅ Dockercleaner command executed successfully in {repair_time:.2f}s: {dockerfile}")

        except subprocess.CalledProcessError as e:
            end_time = time.time()
            repair_time = end_time - start_time
            
            # Record error information
            time_record = {
                'dockerfile': dockerfile,
                'repaired_file': dockerclean_filepath,
                'repair_time_seconds': round(repair_time, 2),
                'status': 'error',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat(),
                'command': command
            }
            time_records.append(time_record)
            
            print(f"❌ Error executing dockercleaner command for {dockerfile}: {e}")
            shutil.copy2(dockerfile, dockerclean_filepath)
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            repair_time = end_time - start_time
            
            time_record = {
                'dockerfile': dockerfile,
                'repaired_file': dockerclean_filepath,
                'repair_time_seconds': round(repair_time, 2),
                'status': 'timeout',
                'error_message': 'Command timed out after 300 seconds',
                'timestamp': datetime.now().isoformat(),
                'command': command
            }
            time_records.append(time_record)
            
            print(f"⏰ Command timed out for {dockerfile}")
            shutil.copy2(dockerfile, dockerclean_filepath)
    
    # Save time records
    if time_log_file:
        save_time_records(time_records, time_log_file)
    
    return time_records

def save_time_records(time_records, filename, mode='w'):
    """Save time records to file"""
    if not time_records:
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Determine file format
    if filename.endswith('.json'):
        with open(filename, mode, encoding='utf-8') as f:
            if mode == 'a' and os.path.exists(filename) and os.path.getsize(filename) > 0:
                # Read existing data and append
                try:
                    f.seek(0)
                    existing_data = json.load(f)
                    existing_data.extend(time_records)
                    f.seek(0)
                    f.truncate()
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Error reading existing JSON file: {e}, creating new file")
                    json.dump(time_records, f, indent=2, ensure_ascii=False)
            else:
                json.dump(time_records, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Time records saved: {filename}")

def generate_summary_report(time_log_file, output_file=None):
    """Generate repair time summary report"""
    if not os.path.exists(time_log_file):
        print(f"Time log file not found: {time_log_file}")
        return
    
    if time_log_file.endswith('.json'):
        with open(time_log_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
    else:
        records = []
    
    if not records:
        print("No records found in time log")
        return
    
    # Analyze data
    successful_repairs = [r for r in records if r.get('status') == 'success']
    failed_repairs = [r for r in records if r.get('status') in ['error', 'timeout']]
    
    summary = {
        'total_files': len(records),
        'successful_repairs': len(successful_repairs),
        'failed_repairs': len(failed_repairs),
        'avg_repair_time': round(sum(r.get('repair_time_seconds', 0) for r in records) / len(records), 2) if records else 0,
        'total_processing_time': round(sum(r.get('repair_time_seconds', 0) for r in records), 2),
        'timestamp': datetime.now().isoformat()
    }
    
    # Print summary
    print("\n" + "="*50)
    print("Repair Time Summary Report")
    print("="*50)
    print(f"Total files processed: {summary['total_files']}")
    print(f"Successful repairs: {summary['successful_repairs']}")
    print(f"Failed repairs: {summary['failed_repairs']}")
    print(f"Average repair time: {summary['avg_repair_time']} seconds")
    print(f"Total processing time: {summary['total_processing_time']} seconds")
    
    # Save summary report
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary report saved to: {output_file}")
    
    return summary

def main():
    """Main execution function"""
    # Main execution
    output_root = 'repair_result'
    
    # Process star1000+ dockerfiles
    print("\nProcessing star1000+ Dockerfiles...")
    star_input = 'dataset_fast/star1000+_dockerfile'
    
    # Set time record directory for star
    star_time_dir = 'time/star/dockercleaner'
    os.makedirs(star_time_dir, exist_ok=True)
    star_time_log = os.path.join(star_time_dir, 'star_dockercleaner_times.json')
    
    # Execute repair
    star_repair_times = process_dockerfiles(star_input, output_root, star_time_log)
    
    # Generate summary report
    star_summary_file = os.path.join(star_time_dir, 'summary_star_dockercleaner.json')
    generate_summary_report(star_time_log, star_summary_file)
    
    print(f"\nAll processing completed! Time records saved in: {star_time_dir}")

if __name__ == "__main__":
    main()