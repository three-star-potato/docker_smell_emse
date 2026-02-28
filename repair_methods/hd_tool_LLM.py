import json
import requests
import re
from tqdm import tqdm
import os
import sys
import signal
import time
from datetime import datetime
import shutil
import subprocess

class UnionCoverageRouter:
    """Union coverage routing system"""
    
    def __init__(self):
        # Define DL rules that tools can solve mapping
        self.tool_capabilities = self._initialize_tool_capabilities()
    
    def _initialize_tool_capabilities(self):
        """Initialize tool capability mapping"""
        return {
            'parfum': [
                'DL3002', 'DL3004', 'DL3006', 'DL3008', 'DL3009', 'DL3013', 
                'DL3015', 'DL3016', 'DL3018', 'DL3019', 'DL3020', 'DL3027', 
                'DL3028', 'DL3029', 'DL3030', 'DL3032', 'DL3033', 'DL3034', 
                'DL3036', 'DL3037', 'DL3038', 'DL3040', 'DL3041', 'DL3042', 
                'DL3059', 'DL3060', 'DL3062', 'DL4001', 'SC2035', 'SC2086'
            ]
        }
    
    def parse_issue(self, issue_line):
        """Parse Hadolint format issue line"""
        groups = issue_line.split(' ')
        HD_number = groups[1].strip()
        dl_number = None
        
        if HD_number.startswith('DL') or HD_number.startswith('SC'):
            dl_number = HD_number
        return dl_number
    
    def calculate_coverage(self, issues):
        """Calculate various coverages"""
        if not issues:
            return {
                'parfum': {'count': 0, 'percentage': 0, 'covered': []},
                'union': {'count': 0, 'percentage': 0, 'covered': []},
                'total_issues': 0
            }
        
        # Extract DL numbers
        dl_numbers = []
        for issue in issues:
            dl_number = self.parse_issue(issue)
            if dl_number:
                dl_numbers.append(dl_number)
  
        total_issues = len(dl_numbers)
        
        # Calculate each tool's coverage
        parfum_coverage = self._calculate_single_coverage(dl_numbers, 'parfum')
        
        # Calculate union coverage (now only parfum)
        hd_tool_LLM = self._calculate_hd_tool_LLM(dl_numbers)
        
        return {
            'parfum': {
                'count': parfum_coverage['count'],
                'percentage': (parfum_coverage['count'] / total_issues * 100) if total_issues > 0 else 0,
                'covered': parfum_coverage['covered']
            },
            'union': {
                'count': hd_tool_LLM['count'],
                'percentage': (hd_tool_LLM['count'] / total_issues * 100) if total_issues > 0 else 0,
                'covered': hd_tool_LLM['covered']
            },
            'total_issues': total_issues
        }
    
    def _calculate_single_coverage(self, dl_numbers, tool):
        """Calculate single tool coverage"""
        covered = []
        for dl_number in dl_numbers:
            if dl_number in self.tool_capabilities[tool]:
                covered.append(dl_number)
        
        return {'count': len(covered), 'covered': covered}
    
    def _calculate_hd_tool_LLM(self, dl_numbers):
        """Calculate union coverage (now only parfum)"""
        covered = []
        for dl_number in dl_numbers:
            if dl_number in self.tool_capabilities['parfum']:
                covered.append(dl_number)
        
        return {'count': len(covered), 'covered': covered}
    
    def select_repair_strategy(self, issues):
        """Select optimal repair strategy based on union coverage"""
        if not issues:
            return {
                'strategy': 'skip',
                'primary_tool': None,
                'reason': 'No issues detected',
                'coverage': {'total_issues': 0}
            }
        
        # Calculate various coverages
        coverage = self.calculate_coverage(issues)
        
        print(f"🔍 Coverage analysis (total {coverage['total_issues']} issues):")
        print(f"   - Parfum alone: {coverage['parfum']['percentage']:6.1f}% ({coverage['parfum']['count']:2d})")
        print(f"   - Union coverage: {coverage['union']['percentage']:6.1f}% ({coverage['union']['count']:2d})")
        
        # Simplified logic: decision based on parfum coverage
        union_coverage = coverage['union']['percentage']
        parfum_coverage = coverage['parfum']['percentage']
        
        if union_coverage == 0:
            # No tool coverage
            strategy = 'llm_only'
            primary_tool = 'llm'
            tools_used = ['llm']
            reason = "No tool coverage, use LLM"
        elif union_coverage == 100:
            # Complete coverage
            strategy = 'tool_only'
            primary_tool = 'parfum'
            tools_used = ['parfum']
            reason = "Parfum complete coverage"
        else:
            # Partial coverage
            strategy = 'tool_then_llm'
            primary_tool = 'parfum'
            tools_used = ['parfum']
            remaining = 100 - parfum_coverage
            reason = f"Parfum covers {parfum_coverage:.1f}% + LLM fallback {remaining:.1f}%"
        
        return {
            'strategy': strategy,
            'primary_tool': primary_tool,
            'tools_used': tools_used,  # Explicitly list tools used
            'reason': reason,
            'coverage': coverage
        }

class HadolintEvaluator:
    """Hadolint evaluator - focused on obtaining real issue lists"""
    
    def run_hadolint(self, dockerfile_path):
        """Run Hadolint to get real issue list"""
        command = f"docker run --rm -i hadolint/hadolint < {dockerfile_path}"
        try:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True, timeout=60)
            output = result.stdout.strip()
            if not output:
                return []  # If no output, return empty list
            issues = [line.strip() for line in output.splitlines() if line.strip()]
            return issues
        except subprocess.CalledProcessError as e:
            # Hadolint returning non-zero status is normal (indicates smells)
            issues = [line.strip() for line in e.output.splitlines() if line.strip()]
            if issues and all(line.startswith('-:') for line in issues):
                return issues  # Normal formatted output
            else:
                # If output format abnormal, return empty list to avoid affecting subsequent processing
                print(f"Hadolint output format abnormal: {dockerfile_path}")
                return []
        except subprocess.TimeoutExpired:
            print(f"Hadolint execution timeout: {dockerfile_path}")
            return []
        except Exception as e:
            print(f"Hadolint execution exception: {str(e)}")
            return []
    
    def get_remaining_issues(self, dockerfile_content):
        """Get remaining issue list for Dockerfile content"""
        # Create temporary file for Hadolint evaluation
        temp_file = self._create_temp_file(dockerfile_content)
        try:
            issues = self.run_hadolint(temp_file)
            return issues
        finally:
            self._cleanup_temp_file(temp_file)
    
    def _create_temp_file(self, content):
        """Create temporary file"""
        import tempfile
        fd, temp_path = tempfile.mkstemp(suffix='.dockerfile')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return temp_path
    
    def _cleanup_temp_file(self, file_path):
        """Clean up temporary file"""
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

class UnionCoverageDockerfileRepair:
    """Union coverage-based Dockerfile repair system (supports OpenAI API and Ollama API)"""
    
    def __init__(self):
        self.router = UnionCoverageRouter()
        self.evaluator = HadolintEvaluator()
    
    def repair_dockerfile(self, dockerfile_path, original_issues, original_content, 
                        model_name=None, no_think=False, use_openai_api=False):
        """Smart repair process based on union coverage (supports OpenAI API)"""
        
        start_time = time.time()
        repair_steps = []
        current_content = original_content
        
        # Select repair strategy
        strategy_info = self.router.select_repair_strategy(original_issues)
        print(f"🛠️ Repair strategy: {strategy_info['reason']}")
                
        total_issues_resolved = 0
        
        # Initialize current_issues_for_llm
        current_issues_for_llm = original_issues  # Default to original issues
        
        if strategy_info['strategy'] == 'skip':
            # No issues, skip repair
            final_content = original_content
            final_status = 'skipped'
            repair_details = "No issues to repair"
            
        else:
            # Step 1: Use tool repair
            if strategy_info['strategy'] in ['tool_only', 'tool_then_llm']:
                primary_tool = strategy_info['primary_tool']
                tools_used = strategy_info.get('tools_used', [primary_tool])
                
                print(f"🔧 Step 1: Using {primary_tool.upper()} repair...")
                
                tool_content, tool_status, tool_details = self._execute_repair(
                    primary_tool, current_content, original_issues, model_name, no_think, use_openai_api
                )
                
                if tool_status in ['success', 'partial']:
                    current_content = tool_content
                    remaining_issues = self.evaluator.get_remaining_issues(current_content)
                    issues_resolved = len(original_issues) - len(remaining_issues)
                    total_issues_resolved += issues_resolved
                    
                    repair_steps.append({
                        'step': 1, 'tool': primary_tool, 'status': tool_status,
                        'details': tool_details, 'issues_resolved': issues_resolved,
                        'issues_remaining': len(remaining_issues)
                    })
                    
                    print(f"   ✅ {primary_tool.upper()} resolved {issues_resolved} issues, {len(remaining_issues)} issues remaining")
                    current_issues_for_llm = remaining_issues
                else:
                    repair_steps.append({
                        'step': 1, 'tool': primary_tool, 'status': 'failed',
                        'details': tool_details, 'issues_resolved': 0,
                        'issues_remaining': len(original_issues)
                    })
                    current_issues_for_llm = original_issues
                    print(f"   ❌ {primary_tool.upper()} repair failed")
            
            # Step 2: LLM fallback repair
            if len(current_issues_for_llm) > 0:
                
                print(f"🤖 Step 2: LLM fallback repair for {len(current_issues_for_llm)} remaining issues...")
                
                llm_content, llm_status, llm_details = self._execute_repair(
                    'llm', current_content, current_issues_for_llm, model_name, no_think, use_openai_api
                )
                
                if llm_status in ['success', 'partial']:
                    current_content = llm_content
                    remaining_after_llm = self.evaluator.get_remaining_issues(current_content)
                    llm_issues_resolved = len(current_issues_for_llm) - len(remaining_after_llm)
                    total_issues_resolved += llm_issues_resolved
                    
                    repair_steps.append({
                        'step': 2, 'tool': 'llm', 'status': llm_status,
                        'details': llm_details, 'issues_resolved': llm_issues_resolved,
                        'issues_remaining': len(remaining_after_llm)
                    })
                    
                    print(f"   ✅ LLM resolved {llm_issues_resolved} issues, {len(remaining_after_llm)} issues finally remaining")
                else:
                    repair_steps.append({
                        'step': 2, 'tool': 'llm', 'status': 'failed',
                        'details': llm_details, 'issues_resolved': 0,
                        'issues_remaining': len(current_issues_for_llm)
                    })
                    print(f"   ❌ LLM repair failed")
            
            # Direct LLM repair (llm_only strategy)
            elif strategy_info['strategy'] == 'llm_only':
                print(f"🤖 Using LLM to repair all {len(original_issues)} issues...")
                
                current_issues_for_llm = self.evaluator.get_remaining_issues(current_content)
                if not current_issues_for_llm:
                    current_issues_for_llm = original_issues
                
                llm_content, llm_status, llm_details = self._execute_repair(
                    'llm', current_content, current_issues_for_llm, model_name, no_think, use_openai_api
                )
                
                if llm_status in ['success', 'partial']:
                    current_content = llm_content
                    remaining_after_llm = self.evaluator.get_remaining_issues(current_content)
                    total_issues_resolved = len(current_issues_for_llm) - len(remaining_after_llm)
                    
                    repair_steps.append({
                        'step': 1, 'tool': 'llm', 'status': llm_status,
                        'details': llm_details, 'issues_resolved': total_issues_resolved,
                        'issues_remaining': len(remaining_after_llm)
                    })
                    
                    print(f"   ✅ LLM resolved {total_issues_resolved} issues, {len(remaining_after_llm)} issues finally remaining")
                else:
                    repair_steps.append({
                        'step': 1, 'tool': 'llm', 'status': 'failed',
                        'details': llm_details, 'issues_resolved': 0,
                        'issues_remaining': len(original_issues)
                    })
                    print(f"   ❌ LLM repair failed")
            
            final_content = current_content
            final_status = self._determine_final_status(repair_steps)
            repair_details = f"Completed repair through {len(repair_steps)} steps"
        
        repair_time = time.time() - start_time
        
        return final_content, {
            'strategy': strategy_info['strategy'],
            'primary_tool': strategy_info.get('primary_tool'),
            'tools_used': strategy_info.get('tools_used', [strategy_info.get('primary_tool')]),
            'status': final_status,
            'repair_time': repair_time,
            'reason': strategy_info['reason'],
            'steps': repair_steps,
            'total_issues_resolved': total_issues_resolved,
            'total_issues_original': len(original_issues),
            'coverage': strategy_info.get('coverage', {}),
            'details': repair_details,
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'no_think': no_think,
            'api_type': 'openai' if use_openai_api else 'ollama'
        }
    
    def _determine_final_status(self, repair_steps):
        """Determine final status based on repair steps"""
        if not repair_steps:
            return 'skipped'
        
        if all(step['status'] in ['success', 'partial'] for step in repair_steps):
            return 'success'
        elif any(step['status'] == 'success' for step in repair_steps):
            return 'partial'
        else:
            return 'failed'
    
    def _execute_repair(self, tool, content, issues, model_name, no_think, use_openai_api):
        """Execute specific repair tool"""
        try:
            if tool == 'llm':
                return self._llm_repair(content, issues, model_name, no_think, use_openai_api)
            elif tool == 'parfum':
                return self._parfum_repair(content, issues)
            else:
                return content, 'error', f'Unknown tool: {tool}'
        except Exception as e:
            return content, 'error', str(e)
    
    def _llm_repair(self, content, issues, model_name, no_think, use_openai_api):
        """LLM repair - use real issue list provided by Hadolint"""
        if not issues:
            return content, 'success', 'No issues to repair'
        
        issues_str = "\n".join(issues)
        
        # Construct strict prompt
        prompt = (
            f"Original Dockerfile:\n```dockerfile\n{content}\n```\n\n"
            f"Smells need to fix:\n{issues_str}\n\n"
            "Return ONLY the modified Dockerfile that:\n"
            "1. Is directly buildable with `docker build`\n"
            "2. Preserves all original functionality\n"
            "3. NO new features added\n\n"
            "4. CRITICAL: ALL package versions MUST be preserved exactly as in original (apt-get, apk, yum, etc.)\n"
                "- If original has versions, keep them unchanged\n"
                "- If original has NO versions, do NOT add versions\n"
            "5. Format:\n```dockerfile\n...\n```"
        )

        modified_content = self._send_llm_request(prompt, model_name, no_think, use_openai_api)
        
        if modified_content:
            return modified_content, 'success', f'LLM needs to repair {len(issues)} issues'
        else:
            return content, 'error', 'LLM repair failed'
    
    def _parfum_repair(self, content, issues):
        """docker-parfum repair"""
        temp_input = self._create_temp_file(content)
        temp_output = temp_input + ".repaired"
        
        try:
            command = f"docker-parfum repair {temp_input} -o {temp_output}"
            result = subprocess.run(command, shell=True, check=True, 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  timeout=300)
            
            with open(temp_output, 'r') as f:
                repaired_content = f.read()
            
            self._cleanup_temp_files(temp_input, temp_output)
            return repaired_content, 'success', 'Parfum repair completed'
            
        except Exception as e:
            self._cleanup_temp_files(temp_input, temp_output)
            return content, 'error', f'Parfum repair failed: {str(e)}'
    
    def _send_llm_request(self, message, model_name, no_think, use_openai_api):
        """Send LLM request (supports OpenAI API and Ollama API)"""
        if use_openai_api:
            return self._call_openai_api(message, model_name)
        else:
            return self._call_ollama_api(message, model_name, no_think)
    
    def _call_openai_api(self, message, model_name):
        """Call OpenAI-compatible API (Bailian)"""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key="fake_api",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
            messages = [{"role": "user", "content": message}]
            
            # Set timeout
            def timeout_handler(signum, frame):
                raise TimeoutError("Request timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)
            
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=False,
                    temperature=0.3,
                    max_tokens=4096
                )
                signal.alarm(0)  # Reset timeout
                
                message_content = completion.choices[0].message.content
                
                # Extract Dockerfile content
                dockerfile_pattern = re.compile(r'```dockerfile(.*?)```', re.DOTALL | re.IGNORECASE)
                match = dockerfile_pattern.search(message_content)
                if match:
                    dockerfile_content = match.group(1).strip()
                    return dockerfile_content
                else:
                    print("No Dockerfile found in the response")
                    return None
                    
            except TimeoutError:
                print("Request timed out after 180 seconds")
                return None
            except Exception as e:
                print(f"OpenAI API call error: {str(e)}")
                return None
                
        except ImportError:
            print("OpenAI library not installed, please run: pip install openai")
            return None
        except Exception as e:
            print(f"Error initializing OpenAI client: {str(e)}")
            return None
    
    def _call_ollama_api(self, message, model_name, no_think):
        """Call local Ollama API"""
        url = "http://localhost:11434/api/chat"
        
        if no_think and "qwen3" in model_name.lower():
            message = f"/no_think\n\n{message}"
        
        messages = [{"role": "user", "content": message}]
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 4096}
        }

        try:
            def timeout_handler(signum, frame):
                raise requests.exceptions.Timeout("Request timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)

            response = requests.post(url, json=payload)
            signal.alarm(0)

            if response.status_code == 200:
                result = response.json()
                message_content = result['message']['content']
                dockerfile_pattern = re.compile(r'```dockerfile(.*?)```', re.DOTALL | re.IGNORECASE)
                match = dockerfile_pattern.search(message_content)
                if match:
                    return match.group(1).strip()
                
                # Try to extract Dockerfile content directly
                lines = message_content.split('\n')
                dockerfile_lines = []
                in_dockerfile = False
                
                for line in lines:
                    if line.strip().startswith('FROM') and not in_dockerfile:
                        in_dockerfile = True
                    if in_dockerfile:
                        dockerfile_lines.append(line)
                
                if dockerfile_lines:
                    return '\n'.join(dockerfile_lines)
                    
            return None

        except Exception as e:
            print(f"LLM request error: {e}")
            return None
    
    def _create_temp_file(self, content):
        """Create temporary file"""
        import tempfile
        fd, temp_path = tempfile.mkstemp(suffix='.dockerfile')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return temp_path
    
    def _cleanup_temp_files(self, *files):
        """Clean up temporary files"""
        for file_path in files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass

def process_dockerfiles_hd_tool_LLM(json_path, root_folder, mode_name, mode_dir, 
                                    time_log_file=None, no_think=False, use_openai_api=False):
    """Union coverage-based routing processing (supports OpenAI API)"""
    if not os.path.exists(mode_dir):
        os.makedirs(mode_dir)
    
    time_records = []
    repair_system = UnionCoverageDockerfileRepair()
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data_json = json.load(file)
    
    # Initialize tool call statistics
    tool_call_summary = {
        'parfum': 0,
        'llm': 0,
        'total_dockerfiles': len(data_json)
    }
    
    # Initialize time statistics
    total_repair_time = 0
    successful_repairs = 0
    failed_repairs = 0
    skipped_repairs = 0
    
    for dockerfile in tqdm(sorted(data_json, key=lambda x: x['dockerfile_path']), desc="Repairing Dockerfiles"):
        dockerfile_path = dockerfile["dockerfile_path"]
        original_issues = dockerfile["issues"]
        modified_filepath = dockerfile_path.replace(root_folder, mode_dir)
        
        os.makedirs(os.path.dirname(modified_filepath), exist_ok=True)
        
        start_time = time.time()
        
        # Read original content
        with open(dockerfile_path, 'r', encoding='utf-8') as file:
            original_content = file.read()
        
        # Check if file already exists
        if os.path.exists(modified_filepath):
            end_time = time.time()
            repair_time = end_time - start_time
            time_record = {
                'dockerfile': dockerfile_path,
                'repaired_file': modified_filepath,
                'repair_time_seconds': round(repair_time, 2),
                'status': 'skipped',
                'reason': 'already_exists',
                'timestamp': datetime.now().isoformat()
            }
            time_records.append(time_record)
            skipped_repairs += 1
            continue
        
        # Execute union coverage repair
        repaired_content, repair_info = repair_system.repair_dockerfile(
            dockerfile_path, original_issues, original_content, mode_name, no_think, use_openai_api
        )
        
        end_time = time.time()
        repair_time = end_time - start_time
        total_repair_time += repair_time
        
        # Save repair result
        with open(modified_filepath, 'w', encoding='utf-8') as file:
            file.write(repaired_content)
        
        # Count tool call times
        for step in repair_info.get('steps', []):
            tool = step.get('tool', '')
            if tool in tool_call_summary:
                tool_call_summary[tool] += 1
        
        # Count repair status
        if repair_info['status'] == 'success':
            successful_repairs += 1
        elif repair_info['status'] == 'failed':
            failed_repairs += 1
        elif repair_info['status'] == 'skipped':
            skipped_repairs += 1
        
        # Record repair information
        time_record = {
            'dockerfile': dockerfile_path,
            'repaired_file': modified_filepath,
            'repair_time_seconds': round(repair_time, 2),
            'status': repair_info['status'],
            'strategy': repair_info['strategy'],
            'primary_tool': repair_info.get('primary_tool'),
            'reason': repair_info['reason'],
            'steps': repair_info['steps'],
            'total_issues_resolved': repair_info['total_issues_resolved'],
            'total_issues_original': repair_info['total_issues_original'],
            'coverage': repair_info.get('coverage', {}),
            'details': repair_info['details'],
            'timestamp': datetime.now().isoformat(),
            'model': mode_name,
            'no_think': no_think,
            'api_type': 'openai' if use_openai_api else 'ollama'
        }
        time_records.append(time_record)
        
        # Print status
        status_icon = '✅' if repair_info['status'] == 'success' else '⚠️' if repair_info['status'] == 'partial' else '❌'
        coverage_info = repair_info.get('coverage', {})
        
        # Safely get coverage information
        union_cov = coverage_info.get('union', {}).get('percentage', 0) if coverage_info else 0
        parfum_cov = coverage_info.get('parfum', {}).get('percentage', 0) if coverage_info else 0
        
        print(f"{status_icon} {repair_info['strategy']:<15} | "
              f"Resolved {repair_info['total_issues_resolved']:2d}/{len(original_issues):2d} issues | "
              f"Union:{union_cov:3.0f}% Parfum:{parfum_cov:3.0f}% | {repair_info['repair_time']:.1f}s")
    
    # Save time records
    if time_log_file:
        save_time_records(time_records, time_log_file)
    
    # Calculate average repair time (excluding skipped files)
    processed_files = successful_repairs + failed_repairs
    avg_repair_time = total_repair_time / processed_files if processed_files > 0 else 0
    
    # Create detailed statistical summary
    detailed_summary = {
        'total_files': tool_call_summary['total_dockerfiles'],
        'successful_repairs': successful_repairs,
        'failed_repairs': failed_repairs,
        'skipped_repairs': skipped_repairs,
        'avg_repair_time': round(avg_repair_time, 2),
        'total_processing_time': round(total_repair_time, 2),
        'parfum_calls': tool_call_summary['parfum'],
        'llm_calls': tool_call_summary['llm'],
        'success_rate': round(successful_repairs / (successful_repairs + failed_repairs) * 100, 2) if (successful_repairs + failed_repairs) > 0 else 0,
        'timestamp': datetime.now().isoformat(),
        'api_type': 'openai' if use_openai_api else 'ollama'
    }
    
    # Save detailed statistical summary
    if time_log_file:
        summary_file = time_log_file.replace('.json', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_summary, f, indent=2, ensure_ascii=False)
        print(f"📊 Detailed statistical summary saved: {summary_file}")
    
    # Print statistical summary
    print(f"\n📊 Processing statistical summary:")
    print(f"   Total Dockerfile count: {detailed_summary['total_files']}")
    print(f"   Successful repairs: {detailed_summary['successful_repairs']}")
    print(f"   Failed repairs: {detailed_summary['failed_repairs']}")
    print(f"   Skipped repairs: {detailed_summary['skipped_repairs']}")
    print(f"   Success rate: {detailed_summary['success_rate']}%")
    print(f"   Average repair time: {detailed_summary['avg_repair_time']} seconds")
    print(f"   Total processing time: {detailed_summary['total_processing_time']} seconds")
    print(f"   Parfum call count: {detailed_summary['parfum_calls']}")
    print(f"   LLM call count: {detailed_summary['llm_calls']}")
    print(f"   API type: {detailed_summary['api_type']}")
    
    return time_records, detailed_summary

def save_time_records(time_records, filename, mode='w'):
    """Save time records to file"""
    if not time_records:
        return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if filename.endswith('.json'):
        with open(filename, mode, encoding='utf-8') as f:
            if mode == 'a' and os.path.exists(filename) and os.path.getsize(filename) > 0:
                try:
                    f.seek(0)
                    existing_data = json.load(f)
                    existing_data.extend(time_records)
                    f.seek(0)
                    f.truncate()
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error reading existing JSON file: {e}")
                    json.dump(time_records, f, indent=2, ensure_ascii=False)
            else:
                json.dump(time_records, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Time records saved: {filename}")

def remove_comments_in_lines(folder_path):
    """Remove comments in Dockerfiles"""
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        print(f"Processing file: {filename}")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            line = line.rstrip()
            comment_index = line.find('#')
            if comment_index != -1:
                line = line[:comment_index].rstrip()
            new_lines.append(line + '\n')
        
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
    
    print(f"Completed: {folder_path}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python hd_tool_LLM_repair.py json_path root_folder mode_name mode_dir [time_log_dir] [--no-think] [--use-openai-api]")
        print("\nParameter description:")
        print("  json_path: JSON file path")
        print("  root_folder: Original Dockerfile root directory")
        print("  mode_name: Model name")
        print("  mode_dir: Output directory")
        print("  time_log_dir: Time log directory (optional)")
        print("  --no-think: Enable no-think mode (only effective for Qwen)")
        print("  --use-openai-api: Use OpenAI-compatible API (Bailian)")
        sys.exit(1)
    
    json_path = sys.argv[1]
    root_folder = sys.argv[2]
    mode_name = sys.argv[3]
    mode_dir = sys.argv[4]
    
    time_log_dir = 'time/star/hd_tool_LLM'
    no_think = '--no-think' in sys.argv
    use_openai_api = '--use-openai-api' in sys.argv
    
    if len(sys.argv) > 5 and not sys.argv[5].startswith('--'):
        time_log_dir = sys.argv[5]
    
    os.makedirs(time_log_dir, exist_ok=True)
    
    model_safe_name = mode_name.replace(':', '_').replace('/', '_')
    think_suffix = '_nothink' if no_think else ''
    api_suffix = '_openai' if use_openai_api else ''
    time_log_file = os.path.join(time_log_dir, f'hd_tool_LLM_{model_safe_name}{think_suffix}{api_suffix}.json')
    
    print(f"Configuration information:")
    print(f"  JSON path: {json_path}")
    print(f"  Root directory: {root_folder}")
    print(f"  Model: {mode_name}")
    print(f"  Output directory: {mode_dir}")
    print(f"  Time log: {time_log_file}")
    print(f"  No-think mode: {no_think}")
    print(f"  OpenAI API: {use_openai_api}")
    
    # Execute union coverage repair
    repair_times, summary = process_dockerfiles_hd_tool_LLM(
        json_path, root_folder, mode_name, mode_dir, time_log_file, no_think, use_openai_api
    )
    
    # Remove comments
    # remove_comments_in_lines(mode_dir)
    
    print(f"\n🎉 Union coverage repair completed!")
    print(f"📊 Supports OpenAI API and Ollama API")
    
    # Print final statistics
    print(f"\n📈 Final statistical results:")
    print(f"   Total Dockerfiles processed: {summary['total_files']}")
    print(f"   Successful repairs: {summary['successful_repairs']} ({summary['success_rate']}%)")
    print(f"   Failed repairs: {summary['failed_repairs']}")
    print(f"   Skipped repairs: {summary['skipped_repairs']}")
    print(f"   Average repair time: {summary['avg_repair_time']} seconds")
    print(f"   Total processing time: {summary['total_processing_time']} seconds")
    print(f"   Parfum total call count: {summary['parfum_calls']} ({summary['parfum_calls']/summary['total_files']*100:.1f}%)")
    print(f"   LLM total call count: {summary['llm_calls']} ({summary['llm_calls']/summary['total_files']*100:.1f}%)")
    print(f"   API type: {summary['api_type']}")

if __name__ == "__main__":
    main()

#     # Use local Ollama API (default)
# python repair_methods/hd_tool_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3:32b" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_32b_hd_tool_LLM_nothink" --no-think
# python repair_methods/hd_tool_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3:14b" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_14b_hd_tool_LLM_nothink" --no-think

# # Use OpenAI-compatible API (Bailian)
# python repair_methods/hd_tool_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "qwen3-235b-a22b-instruct-2507" "repair_result/dataset_fast/star1000+_dockerfile/qwen3_235b_hd_tool_LLM" --use-openai-api
# python repair_methods/hd_tool_LLM.py "evaluate_result/dataset_fast_star1000+_dockerfile.json" "dataset_fast/star1000+_dockerfile" "deepseek-r1-0528" "repair_result/dataset_fast/star1000+_dockerfile/dsr1_671b_hd_tool_LLM" --use-openai-api