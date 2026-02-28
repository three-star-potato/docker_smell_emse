import json
from collections import defaultdict, Counter
import os

def read_json_to_dict(file_path):
    """Read JSON file and return dictionary"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {file_path}")
        return {}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def count_smells(dockerfiles_data, severity_mapping, impact_mapping):
    """Count number of issues and severity levels"""
    smell_count = Counter()       # Issue type count
    severity_count = Counter()    # Severity level count
    impact_count = Counter()      # Functional impact category count
    missing_severity = set()      # Rules missing severity definition
    missing_impact = set()        # Rules missing functional impact definition
    smell_details = defaultdict(list)  # Store detailed information for each issue type
    no_smell_count = 0            # Count of Dockerfiles with no issues
    
    for dockerfile in dockerfiles_data:
        issues = dockerfile.get('issues', [])
        has_smell = False
        
        for issue in issues:
            if issue.startswith("-:"):
                has_smell = True
                parts = issue.split()
                if len(parts) >= 2:
                    issue_type = parts[1]  # Get issue type (e.g., DL3008)
                    smell_count[issue_type] += 1
                    smell_details[issue_type].append(issue)  # Store complete issue description
                    
                    # Get severity level
                    if issue_type in severity_mapping:
                        severity = severity_mapping[issue_type]
                    else:
                        severity = "Unknown"
                        missing_severity.add(issue_type)
                    severity_count[severity] += 1
                    
                    # Get functional impact category
                    if issue_type in impact_mapping:
                        impact = impact_mapping[issue_type]
                    else:
                        impact = "Unknown"
                        missing_impact.add(issue_type)
                    impact_count[impact] += 1
        
        if not has_smell:
            no_smell_count += 1
    
    return smell_count, severity_count, impact_count, missing_severity, missing_impact, smell_details, no_smell_count

def calculate_quality_score(severity_distribution):
    """Calculate quantitative quality score (higher score indicates more severe problems)"""
    # Weight configuration (including Unknown type)
    SEVERITY_WEIGHTS = {
        "Error": 5,
        "Warning": 3,
        "Info": 2,
        "Ignore": 1,
        "Unknown": 1  # 
    }
    
    total_score = 0
    for severity, count in severity_distribution.items():
        weight = SEVERITY_WEIGHTS.get(severity, 0)  # Unconfigured severity level weight is 0
        total_score += count * weight
    return total_score

def process_file(file_path, severity_mapping, impact_mapping):
    """Process single file and return statistical results"""
    data = read_json_to_dict(file_path)
    if not data:
        return None
    
    smell_count, severity_count, impact_count, missing_severity, missing_impact, smell_details, no_smell_count = count_smells(
        data, severity_mapping, impact_mapping
    )
    quality_score = calculate_quality_score(severity_count)
    
    return {
        "file": file_path,
        "total_smells": sum(smell_count.values()),
        "unique_smell_types": len(smell_count),
        "quality_score": quality_score,
        "smell_distribution": dict(smell_count),
        "severity_distribution": dict(severity_count),
        "impact_distribution": dict(impact_count),
        "missing_severity_rules": list(missing_severity),
        "missing_impact_rules": list(missing_impact),
        "smell_details": dict(smell_details),  # Add detailed issue information
        "no_smell_count": no_smell_count,      # Add count of completely problem-free files
        "total_files": len(data)               # Add total file count
    }

def calculate_ratio(base_result, processed_result):
    """Calculate ratio of processed result relative to base result (processed/before)"""
    def calculate_rate(base_value, processed_value):
        if base_value == 0:
            # Avoid division by zero error, if base is 0
            if processed_value == 0:
                return 1.0  # 0/0 = 1
            else:
                return float('inf')  # Non-zero divided by infinity
        return processed_value / base_value * 100
    
    # Calculate net change in issue types
    base_smell_types = set(base_result["smell_distribution"].keys())
    processed_smell_types = set(processed_result["smell_distribution"].keys())
    
    # Completely eliminated issue types
    completely_removed_types = base_smell_types - processed_smell_types
    
    # Newly introduced issue types
    newly_introduced_types = processed_smell_types - base_smell_types
    
    # Calculate quantity changes in common types
    common_types = base_smell_types & processed_smell_types
    count_increased_types = []
    count_decreased_types = []
    
    for smell_type in common_types:
        base_count = base_result["smell_distribution"][smell_type]
        processed_count = processed_result["smell_distribution"][smell_type]
        count_diff = processed_count - base_count
        
        if count_diff > 0:
            count_increased_types.append({
                "type": smell_type,
                "increase": count_diff,
                "from": base_count,
                "to": processed_count
            })
        elif count_diff < 0:
            count_decreased_types.append({
                "type": smell_type,
                "decrease": -count_diff,
                "from": base_count,
                "to": processed_count
            })
    
    # Improved net change calculation: consider type changes and quantity changes
    net_type_change = (len(completely_removed_types) + len(count_decreased_types)) - (len(newly_introduced_types) + len(count_increased_types))
    
    ratio_results = {
        "file": os.path.basename(processed_result["file"]),
        "total_smells": {
            "base": base_result["total_smells"],
            "processed": processed_result["total_smells"],
            "ratio": calculate_rate(base_result["total_smells"], processed_result["total_smells"]),
            "display": f"{processed_result['total_smells']}/{base_result['total_smells']}"
        },
        "unique_smell_types": {
            "base": base_result["unique_smell_types"],
            "processed": processed_result["unique_smell_types"],
            "ratio": calculate_rate(base_result["unique_smell_types"], processed_result["unique_smell_types"]),
            "display": f"{processed_result['unique_smell_types']}/{base_result['unique_smell_types']}"
        },
        "net_smell_type_change": {  # Improved: include quantity changes in net change
            "completely_removed": len(completely_removed_types),
            "newly_introduced": len(newly_introduced_types),
            "count_increased": len(count_increased_types),
            "count_decreased": len(count_decreased_types),
            "net_change": net_type_change,
            "completely_removed_types": list(completely_removed_types),
            "newly_introduced_types": list(newly_introduced_types),
            "count_increased_details": count_increased_types,
            "count_decreased_details": count_decreased_types
        },
        "quality_score": {
            "base": base_result["quality_score"],
            "processed": processed_result["quality_score"],
            "ratio": calculate_rate(base_result["quality_score"], processed_result["quality_score"])
        },
        "no_smell_files": {
            "base": base_result["no_smell_count"],
            "processed": processed_result["no_smell_count"],
            "ratio": calculate_rate(base_result["no_smell_count"], processed_result["no_smell_count"]),
            "display": f"{processed_result['no_smell_count']}/{base_result['no_smell_count']}"
        },
        "severity_distribution": {},
        "impact_distribution": {},
        "added_smells": [],
        "removed_smells": []
    }
    
    # Calculate ratio of severity level distribution
    for severity, base_count in base_result["severity_distribution"].items():
        processed_count = processed_result["severity_distribution"].get(severity, 0)
        ratio_results["severity_distribution"][severity] = {
            "base": base_count,
            "processed": processed_count,
            "ratio": calculate_rate(base_count, processed_count)
        }
    
    # Calculate ratio of functional impact distribution
    for impact, base_count in base_result["impact_distribution"].items():
        processed_count = processed_result["impact_distribution"].get(impact, 0)
        ratio_results["impact_distribution"][impact] = {
            "base": base_count,
            "processed": processed_count,
            "ratio": calculate_rate(base_count, processed_count)
        }
    
    # Compare issue type changes between base file and processed file
    base_smells = set(base_result["smell_distribution"].keys())
    processed_smells = set(processed_result["smell_distribution"].keys())
    
    # Newly added issue types
    added_smells = processed_smells - base_smells
    for smell in added_smells:
        ratio_results["added_smells"].append({
            "type": smell,
            "count": processed_result["smell_distribution"][smell],
            "examples": processed_result["smell_details"].get(smell, [])[:3]  # Show at most 3 examples
        })
    
    # Reduced issue types
    removed_smells = base_smells - processed_smells
    for smell in removed_smells:
        ratio_results["removed_smells"].append({
            "type": smell,
            "count": base_result["smell_distribution"][smell],
            "examples": base_result["smell_details"].get(smell, [])[:3]  # Show at most 3 examples
        })
    
    # Compare quantity changes of same issue types
    common_smells = base_smells & processed_smells
    for smell in common_smells:
        base_count = base_result["smell_distribution"][smell]
        processed_count = processed_result["smell_distribution"][smell]
        
        if processed_count > base_count:
            ratio_results["added_smells"].append({
                "type": smell,
                "count_change": f"+{processed_count - base_count}",
                "new_count": processed_count,
                "old_count": base_count,
                "examples": processed_result["smell_details"].get(smell, [])[:3]
            })
        elif processed_count < base_count:
            ratio_results["removed_smells"].append({
                "type": smell,
                "count_change": f"-{base_count - processed_count}",
                "new_count": processed_count,
                "old_count": base_count,
                "examples": base_result["smell_details"].get(smell, [])[:3]
            })
    
    return ratio_results

def process_group(base_file, processed_files, severity_mapping, impact_mapping):
    """Process all files in a group (star1000+)"""
    # Process base file
    base_result = process_file(base_file, severity_mapping, impact_mapping)
    if not base_result:
        print(f"Failed to process base file: {base_file}")
        return []
    
    # Process all processed files
    processed_results = [process_file(file, severity_mapping, impact_mapping) for file in processed_files]
    processed_results = [res for res in processed_results if res]  # Filter failed ones
    
    # Calculate ratio of all processed files relative to base file (processed/before)
    ratio_results = []
    for res in processed_results:
        ratio_results.append(calculate_ratio(base_result, res))
    
    return {
        "base_file": base_file,
        "base_result": base_result,
        "processed_results": processed_results,
        "ratio_results": ratio_results
    }

def print_group_summary(group_name, group_results):
    """Print group summary results"""
    print(f"\n{'=' * 80}")
    print(f"Result Summary: {group_name.upper()} Dataset")
    print(f"{'=' * 80}")
    
    # Print base file information
    base_result = group_results["base_result"]
    print(f"\nBase File: {group_results['base_file']}")
    print(f"  Total files: {base_result['total_files']}")
    print(f"  Problem-free files: {base_result['no_smell_count']} ({base_result['no_smell_count']/base_result['total_files']*100:.2f}%)")
    print(f"  Total issues: {base_result['total_smells']}")
    print(f"  Issue types: {base_result['unique_smell_types']}")
    print(f"  Weighted score: {base_result['quality_score']}")
    
    # Print ratio and raw values of each processed file relative to base file
    print(f"\n{'File':<60}{'Total Issues':>15}{'Ratio(%)':>15}{'Type Count':>15}{'Ratio(%)':>15}{'Weighted Score':>15}{'Ratio(%)':>15}{'Problem-free':>15}{'Ratio(%)':>15}")
    
    # First sort by total issue ratio (lower ratio is better)
    sorted_pairs = sorted(
        zip(group_results["processed_results"], group_results["ratio_results"]),
        key=lambda x: x[1]['total_smells']['ratio'] 
    )
    
    # Print table
    for idx, (processed_result, ratio_result) in enumerate(sorted_pairs):
        total_base = ratio_result['total_smells']['base']
        total_processed = ratio_result['total_smells']['processed']
        total_ratio = ratio_result['total_smells']['ratio']
        
        quality_base = ratio_result['quality_score']['base']
        quality_processed = ratio_result['quality_score']['processed']
        quality_ratio = ratio_result['quality_score']['ratio']
        
        no_smell_base = ratio_result['no_smell_files']['base']
        no_smell_processed = ratio_result['no_smell_files']['processed']
        no_smell_ratio = ratio_result['no_smell_files']['ratio']
        
        # Get improved net change information
        net_change_info = ratio_result['net_smell_type_change']
        net_change = net_change_info['net_change']
        removed_count = net_change_info['completely_removed']
        introduced_count = net_change_info['newly_introduced']
        count_decreased = net_change_info['count_decreased']
        count_increased = net_change_info['count_increased']
        
        # Improved net change display: include type changes and quantity changes
        net_change_display = f"{net_change:+d} ({removed_count+count_decreased}↓/{introduced_count+count_increased}↑)"
        
        file_name = os.path.basename(ratio_result['file'])
        
        # Mark optimal result
        rank_marker = "★" if idx == 0 else ""
        print(f"{file_name:<60}{total_processed:>15}{total_ratio:>15.2f}%{net_change_display:>15}{quality_processed:>15}{quality_ratio:>15.2f}%{no_smell_processed:>15}{no_smell_ratio:>15.2f}%{rank_marker:>5}")    # Print detailed distribution information and change types
# Print detailed distribution information and change types
    for processed_result, ratio_result in sorted_pairs:
        net_change_info = ratio_result['net_smell_type_change']
        net_change = net_change_info['net_change']
        removed_count = net_change_info['completely_removed']
        introduced_count = net_change_info['newly_introduced']
        count_decreased = net_change_info['count_decreased']
        count_increased = net_change_info['count_increased']
        
        print(f"\nFile: {os.path.basename(ratio_result['file'])}")
        print(f"  Net issue type change: {net_change:+d} (completely eliminated {removed_count} types + quantity decreased {count_decreased} types - newly introduced {introduced_count} types - quantity increased {count_increased} types)")
        print(f"  Problem-free files: {ratio_result['no_smell_files']['processed']}/{ratio_result['no_smell_files']['base']} ({ratio_result['no_smell_files']['ratio']:.2f}%)")
        
        # # Print issue types with quantity increase
        # if net_change_info['count_increased_details']:
        #     print(f"  Issue types with quantity increase ({len(net_change_info['count_increased_details'])} types):")
        #     for item in net_change_info['count_increased_details']:
        #         print(f"    {item['type']}: +{item['increase']} (from {item['from']} to {item['to']})")
        
        # # Print issue types with quantity decrease  
        # if net_change_info['count_decreased_details']:
        #     print(f"  Issue types with quantity decrease ({len(net_change_info['count_decreased_details'])} types):")
        #     for item in net_change_info['count_decreased_details']:
        #         print(f"    {item['type']}: -{item['decrease']} (from {item['from']} to {item['to']})")
        
        # # Original printing of newly added and reduced issue types remains unchanged...
        # # Print newly added issue types
        # if ratio_result["added_smells"]:
        #     print(f"  Newly added issue types ({len(ratio_result['added_smells'])} types):")
        #     for added in ratio_result["added_smells"]:
        #         if "count_change" in added:
        #             print(f"    {added['type']}: Quantity change {added['count_change']} (from {added['old_count']} to {added['new_count']})")
        #         else:
        #             print(f"    {added['type']}: Newly added {added['count']}")
        #         # Print examples
        #         for example in added.get("examples", [])[:1]:
        #             pass
        #             # print(f"      - Example: {example}")
        
        # # Print reduced issue types
        # if ratio_result["removed_smells"]:
        #     print(f"  Reduced issue types ({len(ratio_result['removed_smells'])} types):")
        #     for removed in ratio_result["removed_smells"]:
        #         if "count_change" in removed:
        #             print(f"    {removed['type']}: Quantity change {removed['count_change']} (from {removed['old_count']} to {removed['new_count']})")
        #         else:
        #             print(f"    {removed['type']}: Completely eliminated (originally {removed['count']})")
        #         # Print examples
        #         for example in removed.get("examples", [])[:1]:
        #             pass
                    # print(f"      - Example: {example}")
    return sorted_pairs

def main():
    # Read severity mapping and functional impact mapping
    severity_file = "evaluate/level.json"
    severity_data = read_json_to_dict(severity_file)
    if not severity_data:
        print(f"Failed to load severity mapping from {severity_file}")
        return
    
    # Create two mapping dictionaries
    severity_mapping = {}
    impact_mapping = {}
    
    for item in severity_data:
        if "id" in item:
            rule_id = item["id"]
            severity_mapping[rule_id] = item.get("defaultSeverity", "Unknown")
            impact_mapping[rule_id] = item.get("function_impact", "Unknown")
    
    # Define file groups for Star1000+
    groups = {
        "star1000+": {
            "base": "evaluate_result/dataset_fast_star1000+_dockerfile.json",
        "processed": [
            "evaluate_result/dataset_fast_star1000+_dockerfile_parfum.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_dockercleaner.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_8b_hd_LLM_nothink.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_8b_hd_LLM_nothink_1.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_8b_hd_LLM_nothink_2.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_06b_hd_LLM_nothink.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_06b_hd_LLM_nothink_1.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_06b_hd_LLM_nothink_2.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_06b_finetune.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_06b_finetunenoparfum.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_LLM_1.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_LLM_2.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_LLM.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_32b_hd_LLM_nothink.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_32b_hd_LLM_nothink_1.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_32b_hd_LLM_nothink_2.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_06b.json"]
        }
    }
    
    # Process all groups
    all_results = {}
    for group_name, group_data in groups.items():
        group_results = process_group(
            group_data["base"],
            group_data["processed"],
            severity_mapping,
            impact_mapping
        )
        
        if group_results:
            sorted_results = print_group_summary(group_name, group_results)
            all_results[group_name] = {
                "processed_results": [r[0] for r in sorted_results],
                "ratio_results": [r[1] for r in sorted_results]
            }
    
    # Analyze performance of each model across different datasets
    print("\n\n" + "="*80)
    print("Comprehensive Model Performance Analysis")
    print("="*80)
    
    # Collect ratio data for each model
    model_performance = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Define all metrics we care about
    severity_metrics = ["Error", "Warning", "Info"]
    impact_metrics = ["Security", "Correctness", "Maintainability", "Efficiency"]
    
    for group_name, results in all_results.items():
        for processed_result, ratio_result in zip(results["processed_results"], results["ratio_results"]):
            file_name = os.path.basename(ratio_result["file"])
            
            # Extract model name and method
            model_info = extract_model_and_method(file_name)
            
            if model_info:
                model_name = model_info["model"]
                method_name = model_info["method"]
                                  
                if "star1000+" in group_name:
                    # Collect basic metrics
                    model_performance[model_name][method_name]["star_total"] = ratio_result["total_smells"]["processed"]
                    model_performance[model_name][method_name]["star_total_ratio"] = ratio_result["total_smells"]["ratio"]
                    model_performance[model_name][method_name]["star_types"] = ratio_result["unique_smell_types"]["processed"]
                    model_performance[model_name][method_name]["star_types_ratio"] = ratio_result["unique_smell_types"]["ratio"]
                    model_performance[model_name][method_name]["star_quality"] = ratio_result["quality_score"]["processed"]
                    model_performance[model_name][method_name]["star_quality_ratio"] = ratio_result["quality_score"]["ratio"]
                    model_performance[model_name][method_name]["star_no_smell"] = ratio_result["no_smell_files"]["processed"]
                    model_performance[model_name][method_name]["star_no_smell_ratio"] = ratio_result["no_smell_files"]["ratio"]
                    
                    # Collect severity level ratios
                    for severity in severity_metrics:
                        if severity in ratio_result["severity_distribution"]:
                            model_performance[model_name][method_name][f"star_{severity.lower()}"] = ratio_result["severity_distribution"][severity]["processed"]
                            model_performance[model_name][method_name][f"star_{severity.lower()}_ratio"] = ratio_result["severity_distribution"][severity]["ratio"]
                    
                    # Collect functional impact ratios
                    for impact in impact_metrics:
                        if impact in ratio_result["impact_distribution"]:
                            model_performance[model_name][method_name][f"star_{impact.lower()}"] = ratio_result["impact_distribution"][impact]["processed"]
                            model_performance[model_name][method_name][f"star_{impact.lower()}_ratio"] = ratio_result["impact_distribution"][impact]["ratio"]
    
    # Print model performance comparison table
    print("\nModel Comprehensive Performance Comparison:")
    print(f"{'Model':<15}{'Method':<25}{'Dataset':<10}{'Total Issues':>15}{'Ratio(%)':>15}{'Type Count':>15}{'Ratio(%)':>15}{'Score':>15}{'Ratio(%)':>15}{'Problem-free':>15}{'Ratio(%)':>15}")
    
    # Collect all model-method combinations
    model_methods = []
    for model, methods in model_performance.items():
        for method in methods:
            model_methods.append((model, method))
    
    # Sort by model name and method
    model_methods.sort(key=lambda x: (x[0], x[1]))
    
    for model, method in model_methods:
        data = model_performance[model][method]
        
        # Star1000+ data
        print(f"{'':<15}{'':<25}{'Star':<10}", end="")
        print(f"{data.get('star_total', 'N/A'):>15}", end="")
        print(f"{data.get('star_total_ratio', 'N/A'):>15.2f}%", end="")
        print(f"{data.get('star_types', 'N/A'):>15}", end="")
        print(f"{data.get('star_types_ratio', 'N/A'):>15.2f}%", end="")
        print(f"{data.get('star_quality', 'N/A'):>15}", end="")
        print(f"{data.get('star_quality_ratio', 'N/A'):>15.2f}%", end="")
        print(f"{data.get('star_no_smell', 'N/A'):>15}", end="")
        print(f"{data.get('star_no_smell_ratio', 'N/A'):>15.2f}%")
        
        # Print separator line
        print("-" * 180)

    # Add severity metrics table
    print("\nModel Comprehensive Performance Comparison (including severity metrics):")
    print(f"{'Model':<15}{'Method':<20}{'Dataset':<10}{'Total':>8}{'Ratio':>8}{'Error':>10}{'Ratio':>8}{'Warning':>10}{'Ratio':>8}{'Info':>10}{'Ratio':>8}")

    for model, method in model_methods:
        data = model_performance[model][method]
        
        # Star1000+ data
        print(f"{model:<15}{method:<20}{'Star':<10}", end="")
        print(f"{data.get('star_total', 'N/A'):>8}", end="")
        print(f"{data.get('star_total_ratio', 'N/A'):>8.1f}%", end="")
        
        # Add severity metrics
        for severity in ["error", "warning", "info"]:
            severity_key = f"star_{severity}"
            ratio_key = f"star_{severity}_ratio"
            print(f"{data.get(severity_key, 'N/A'):>10}", end="")
            print(f"{data.get(ratio_key, 'N/A'):>8.1f}%", end="")
        
        print()  # New line

    # Modify header to include impact metrics
    print("\nModel Comprehensive Performance Comparison (including Impact metrics):")
    print(f"{'Model':<15}{'Method':<20}{'Dataset':<10}{'Total':>8}{'Ratio':>8}{'Security':>10}{'Ratio':>8}{'Correctness':>12}{'Ratio':>8}{'Maintain':>10}{'Ratio':>8}{'Efficiency':>10}{'Ratio':>8}")

    for model, method in model_methods:
        data = model_performance[model][method]
        
        # Star1000+ data
        print(f"{model:<15}{method:<20}{'Star':<10}", end="")
        print(f"{data.get('star_total', 'N/A'):>8}", end="")
        print(f"{data.get('star_total_ratio', 'N/A'):>8.1f}%", end="")
        
        # Add impact metrics
        for impact in ["security", "correctness", "maintainability", "efficiency"]:
            impact_key = f"star_{impact}"
            ratio_key = f"star_{impact}_ratio"
            print(f"{data.get(impact_key, 'N/A'):>10}", end="")
            print(f"{data.get(ratio_key, 'N/A'):>8.1f}%", end="")
        
        print()  # New line

def extract_model_and_method(file_name):
    """Extract model name and method from file name"""
    if "qwen3_" in file_name:
        # Extract model size and method
        parts = file_name.split("qwen3_")[1].split("_")
        model_size = parts[0]
        method = "_".join(parts[1:]).replace(".json", "")
        if "msr" in file_name:
            return {"model": f"qwen3_235b", "method": "msricl"}

        return {"model": f"qwen3_{model_size}", "method": method}
    elif "dockercleaner" in file_name:
        return {"model": "Dockercleaner", "method": "default"}
    elif "parfum" in file_name:
        return {"model": "Parfum", "method": "default"}
    
    return None

if __name__ == "__main__":
    main()