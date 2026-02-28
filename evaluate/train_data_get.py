import json
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional

def read_json(file_path: str) -> List[Dict]:
    """Read JSON file, add type hints and more detailed error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def calculate_smell_score(issues: List[str], severity_mapping: Dict[str, str]) -> int:
    """Calculate issue score for a single Dockerfile, optimize parsing logic"""
    SEVERITY_WEIGHTS = {
        "Error": 5,
        "Warning": 3,
        "Info": 2,
        "Ignore": 1,
        "Unknown": 1
    }
    
    score = 0
    for issue in issues:
        if issue.startswith("-:"):
            parts = issue.split()
            if len(parts) >= 2:
                issue_type = parts[1]
                severity = severity_mapping.get(issue_type, "Unknown")
                score += SEVERITY_WEIGHTS.get(severity, 0)
    return score

def get_dockerfile_name(path: str) -> str:
    """Extract Dockerfile filename from full path, enhance path processing robustness"""
    try:
        return os.path.basename(path)
    except Exception:
        return path

def analyze_best_solutions(dockerfile_scores: Dict[str, List[Tuple[int, int, str]]]) -> Tuple[List[int], List[Dict]]:
    """Analyze optimal solution distribution, now only returns all optimal solution method indices and details"""
    all_best = []   # All optimal solution method indices
    best_details = []   # Details of optimal solutions
    
    for dockerfile_name, scores in dockerfile_scores.items():
        if not scores:
            continue
            
        # Find the lowest score
        min_score = min(score for (_, score, _) in scores)
        
        # Find all optimal solution methods
        best_methods = [file_idx for (file_idx, score, _) in scores if score == min_score]
        all_best.extend(best_methods)
        
        # Save optimal solution details
        for file_idx, score, path in scores:
            if score == min_score:
                best_details.append({
                    "dockerfile_name": dockerfile_name,
                    "dockerfile_path": path,
                    "best_score": score,
                    "method_index": file_idx
                })
    
    return all_best, best_details

def get_stats(best_sources: List[int], file_paths: List[str]) -> Tuple[List[Dict], int]:
    """Calculate statistical results, extracted as independent function"""
    total_best = len(best_sources) if best_sources else 1
    file_counts = Counter(best_sources)
    
    results = []
    for file_idx, count in file_counts.items():
        percentage = (count / total_best) * 100
        results.append({
            "file": file_paths[file_idx],
            "file_name": os.path.basename(file_paths[file_idx]),
            "best_count": count,
            "percentage": round(percentage, 2),
            "rank": file_idx + 1
        })
    
    # Sort in descending order by percentage (sort by original order when percentages are equal)
    results.sort(key=lambda x: (-x["percentage"], x["rank"]))
    return results, total_best

def print_analysis_results(results: Dict, analysis_name: str) -> None:
    """Print analysis results, optimize output format"""
    title = f"{analysis_name} Analysis Results"

    print(f"\n{title}")
    print("=" * 120)
    print(f"{'Rank':<5} {'Repair Method':<60} {'Optimal Count':<12} {'Percentage(%)':<10} {'File Order':<10} {'Coverage':<10}")
    print("-" * 120)
    
    for i, result in enumerate(results["results"], 1):
        print(f"{i:<5} {result['file_name']:<60} {result['best_count']:<12} "
              f"{result['percentage']:<10.2f} #{result['rank']:<10} "
              f"{(result['best_count']/results['dockerfile_count'])*100:.1f}%")
    
    print("=" * 120)
    print(f"Summary: Total Dockerfiles={results['dockerfile_count']} | "
          f"Total Optimal Solutions={results['total_best']} | "
          f"Average Coverage={(results['total_best']/results['dockerfile_count'])*100:.1f}%")

def generate_all_solutions_report(
    dockerfile_scores: Dict[str, List[Tuple[int, int, str]]], 
    file_paths: List[str],
    output_file: Optional[str] = None
) -> None:
    """
    Generate detailed report for all solutions, including optimal and non-optimal solutions
    :param dockerfile_scores: Dockerfile score data
    :param file_paths: List of all method file paths
    :param output_file: Output file path, if None then print to console
    """
    report = []
    
    for dockerfile_name, scores in dockerfile_scores.items():
        if not scores:
            continue
            
        # Find the lowest score (optimal solution)
        min_score = min(score for (_, score, _) in scores)
        
        # Collect score information for all methods
        methods_info = []
        for file_idx, score, original_path in scores:
            method_name = os.path.basename(file_paths[file_idx])
            is_best = score == min_score
            gap = score - min_score if not is_best else 0

            methods_info.append({
                "method": method_name,
                "method_path": file_paths[file_idx],
                "dockerfile_repair_path": original_path,
                "score": score,
                "is_best": is_best,
                "gap": gap,
                "method_index": file_idx
            })
        
        # Sort by score
        methods_info.sort(key=lambda x: x["score"])
        
        report.append({
            "dockerfile_name": dockerfile_name,
            "min_score": min_score,
            "methods": methods_info,
            "method_count": len(methods_info)
        })
    
    # Sort by Dockerfile name
    report.sort(key=lambda x: x["dockerfile_name"])
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Detailed report of all solutions saved to {output_file}")
    else:
        print("\nDetailed report of solutions for all Dockerfiles:")
        print("=" * 150)
        for item in report:
            print(f"\nDockerfile: {item['dockerfile_name']} (Lowest Score: {item['min_score']})")
            print("-" * 150)
            print(f"{'Method Name':<30} {'Score':<8} {'Optimal?':<10} {'Gap':<8} {'Method Index':<10} {'Original Dockerfile Path':<60}")
            print("-" * 150)
            for method in item["methods"]:
                print(f"{method['method']:<30} {method['score']:<8} "
                      f"{'✓' if method['is_best'] else '✗':<10} "
                      f"{method['gap']:<8}"
                      f"{method['method_index']:<10} {method['dockerfile_repair_path']:<60}")
        print("=" * 150)

def process_dataset(file_paths: List[str], severity_file: str, dataset_name: str) -> Dict:
    """Process a single dataset, add functionality to generate all solutions report"""
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Read severity level mapping
    severity_data = read_json(severity_file)
    severity_mapping = {item['id']: item['defaultSeverity'] for item in severity_data}
    
    # Collect scores for all Dockerfiles (grouped by filename)
    dockerfile_scores = defaultdict(list)
    
    # Collect scores for all Dockerfiles for each file
    for file_idx, file_path in enumerate(file_paths):
        data = read_json(file_path)
        for item in data:
            dockerfile_path = item.get("dockerfile_path", "")
            dockerfile_name = get_dockerfile_name(dockerfile_path)
            score = calculate_smell_score(item.get("issues", []), severity_mapping)
            dockerfile_scores[dockerfile_name].append((file_idx, score, dockerfile_path))
    
    # Analyze optimal solutions - now only returns all optimal solutions
    all_best, best_details = analyze_best_solutions(dockerfile_scores)
    
    # Calculate statistical results
    results, total_best = get_stats(all_best, file_paths)

    # Update method name in best_details
    for detail in best_details:
        detail["best_method"] = os.path.basename(file_paths[detail["method_index"]])
    
    # Generate report for all solutions (including non-optimal ones)
    generate_all_solutions_report(
        dockerfile_scores, 
        file_paths,
        output_file=f"evaluate_result/all_solutions_report_{dataset_name.replace(' ', '_')}.json"
    )
    
    return {
        "results": results,
        "dockerfile_count": len(dockerfile_scores),
        "total_best": total_best,
        "best_details": best_details,
        "all_scores": dockerfile_scores  # Retain all score data
    }

def main():
    # Configure file paths
    severity_file = "evaluate/level.json"
    
    # Only process Star1000+ dataset
    star_dataset = {
        "name": "Star1000+ Dockerfiles",
        "files": [
            "evaluate_result/dataset_fast_star1000+_dockerfile.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_parfum.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_LLM.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_LLM_1.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_LLM_2.json",

        ]
    }
    
    # Process Star1000+ dataset
    results = process_dataset(star_dataset["files"], severity_file, star_dataset["name"])
    print_analysis_results(results, star_dataset["name"])

if __name__ == "__main__":
    main()