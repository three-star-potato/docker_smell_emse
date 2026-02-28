import re
import json
import os
from tqdm import tqdm
from datetime import datetime
import sys
from typing import Dict, List, Tuple, Set

def parse_log_content(log_content):
    """Parse log content containing <phase>, <path>, and <error> tags"""
    phase_pattern = r'<phase>(.*?)<phase>'
    path_pattern = r'<path>(.*?)<path>'
    error_pattern = r'<error>(.*?)<error>'

    phase = re.search(phase_pattern, log_content)
    path = re.search(path_pattern, log_content)
    error = re.search(error_pattern, log_content)

    phase_content = phase.group(1) if phase else None
    path_content = path.group(1) if path else None
    error_content = error.group(1) if error else None

    return {
        'phase': phase_content,
        'path': path_content,
        'error': error_content
    }

def extract_original_path(full_path):
    """Extract original file path from repair file path"""
    parts = full_path.split('/')
    
    if 'repair_result' in parts:
        repair_idx = parts.index('repair_result')
        original_parts = parts[:repair_idx]
        
        for i in range(repair_idx + 1, len(parts)):
            if parts[i] in ['parfum', 'dockercleaner', 'qwen3_235b_hd_LLM', 'msr25_icl_qwen3_235b']:
                continue
            original_parts.append(parts[i])
        
        original_path = '/'.join(original_parts)
        return original_path
    return None

class SimpleErrorClassifier:
    """Simplified error classifier - based on Dockerfile instruction matching"""
    
    def __init__(self):
        self.classification_stats = {
            'Base image stage errors': 0,
            'Context stage errors': 0, 
            'Command execution stage errors': 0,
            'Environment configuration stage errors': 0,
            'Unknown': 0
        }
    
    def classify_by_phase(self, error_message: str, failed_phase: str) -> Tuple[str, str]:
        """Classify error based on build phase"""
        
        if not failed_phase:
            self.classification_stats['Base image stage errors'] += 1
            return "Base image stage errors", f"Base image stage"
        
        
        # Clean phase information
        clean_phase = self._clean_message(failed_phase)
        clean_error = self._clean_message(error_message)

        # Simple instruction matching - first match uppercase Dockerfile instructions
        if any(keyword in clean_phase for keyword in ['FROM']):
            self.classification_stats['Base image stage errors'] += 1
            return "Base image stage errors", f"Base image stage: {clean_phase[:100]}..."

        elif any(keyword in clean_phase for keyword in ['COPY', 'ADD', 'copy']):
            self.classification_stats['Context stage errors'] += 1
            return "Context stage errors", f"Build context stage: {clean_phase[:100]}..."

        elif any(keyword in clean_phase for keyword in ['RUN']):
            self.classification_stats['Command execution stage errors'] += 1
            return "Command execution stage errors", f"Command execution stage: {clean_phase[:100]}..."

        elif any(keyword in clean_phase for keyword in ['ARG', 'ENV', 'WORKDIR', 'USER', 'EXPOSE', 'VOLUME', 'LABEL']):
            self.classification_stats['Environment configuration stage errors'] += 1
            return "Environment configuration stage errors", f"Environment configuration stage: {clean_phase[:100]}..."

        
        self.classification_stats['Unknown'] += 1
        return "Unknown", f"Cannot classify - phase: {clean_phase[:100]}..."

    def _clean_message(self, message: str) -> str:
        """Clean message (remove color codes, etc.)"""
        if not message:
            return ""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', message).strip()

    def get_classification_stats(self) -> Dict:
        """Get classification statistics"""
        return self.classification_stats

def analyze_build_errors_simple(unbuild_path, output_file=None):
    """Simplified build error analysis"""
    
    if not os.path.exists(unbuild_path):
        print(f"Error: File does not exist - {unbuild_path}")
        return None
    
    with open(unbuild_path, 'r', encoding='utf-8') as file:
        unbuild_content = file.readlines()
    
    classifier = SimpleErrorClassifier()
    analysis_results = []
    
    print("🔧 Starting simplified build error analysis...")
    print("📋 Four build stages:")
    stages = [
        "Base image stage errors - FROM instruction, image pull",
        "Context stage errors - COPY/ADD file operations", 
        "Command execution stage errors - RUN command execution",
        "Environment configuration stage errors - Environment configuration"
    ]
    for stage in stages:
        print(f"  {stage}")
    
    for line in tqdm(unbuild_content, desc="Analyzing errors"):
        line = line.strip()
        if not line:
            print("Empty line")
            continue
            
        parsed_log = parse_log_content(line)
        if not parsed_log['path'] or not parsed_log['error']:
            print(f"Unable to parse log line: {line}")
            continue

            
        repair_path = parsed_log['path']
        original_path = extract_original_path(repair_path)
        
        # Classify error
        error_type, reasoning = classifier.classify_by_phase(
            parsed_log['error'], 
            parsed_log['phase']
        )
        
        analysis_result = {
            'original_path': original_path,
            'repair_path': repair_path,
            'error_message': parsed_log['error'],
            'failed_phase': parsed_log['phase'],
            'error_type': error_type,
            'reasoning': reasoning
        }
        analysis_results.append(analysis_result)
    
    # Generate output file
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"simple_analysis.json"
    
    # Save results
    output_data = {
        'classification_summary': classifier.get_classification_stats(),
        'results': analysis_results,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print statistical results
    print_simple_summary(classifier, output_file)
    
    return output_data

def print_simple_summary(classifier, output_file):
    """Print simplified statistical results"""
    stats = classifier.get_classification_stats()
    total_cases = sum(stats.values())
    
    print("\n" + "="*60)
    print("📊 Simplified Error Analysis Results")
    print("="*60)
    
    print("\nError Type Distribution:")
    for error_type, count in stats.items():
        percentage = (count / total_cases * 100) if total_cases > 0 else 0
        print(f"  {error_type:<35}: {count:>3} ({percentage:>5.1f}%)")
    
    classified = total_cases - stats.get('Unknown', 0)
    classified_pct = (classified / total_cases * 100) if total_cases > 0 else 0
    print(f"\nClassification Success Rate: {classified}/{total_cases} ({classified_pct:.1f}%)")
    print(f"Result File: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_analyzer.py unbuild.log [output_file]")
        print("\nExamples:")
        print("  python simple_analyzer.py /path/to/unbuild.log")
        print("  python simple_analyzer.py /path/to/unbuild.log results.json")
        sys.exit(1)
    
    unbuild_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(unbuild_path):
        print(f"Error: File does not exist - {unbuild_path}")
        sys.exit(1)
    
    print(f"🔧 Starting simplified analysis")
    print(f"Input File: {unbuild_path}")
    if output_file:
        print(f"Output File: {output_file}")
    
    result = analyze_build_errors_simple(unbuild_path, output_file)

if __name__ == "__main__":
    main()