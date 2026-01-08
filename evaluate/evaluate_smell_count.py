import json
from collections import defaultdict, Counter
import os

def read_json_to_dict(file_path):
    """读取JSON文件并返回字典"""
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
    """统计问题数量和严重级别"""
    smell_count = Counter()       # 问题类型计数
    severity_count = Counter()    # 严重级别计数
    impact_count = Counter()      # 功能影响类别计数
    missing_severity = set()      # 缺少严重性定义的规则
    missing_impact = set()         # 缺少功能影响定义的规则
    smell_details = defaultdict(list)  # 存储每个问题类型的详细信息
    no_smell_count = 0            # 完全没有问题的Dockerfile计数
    
    for dockerfile in dockerfiles_data:
        issues = dockerfile.get('issues', [])
        has_smell = False
        
        for issue in issues:
            if issue.startswith("-:"):
                has_smell = True
                parts = issue.split()
                if len(parts) >= 2:
                    issue_type = parts[1]  # 获取问题类型 (如 DL3008)
                    smell_count[issue_type] += 1
                    smell_details[issue_type].append(issue)  # 存储完整的问题描述
                    
                    # 获取严重级别
                    if issue_type in severity_mapping:
                        severity = severity_mapping[issue_type]
                    else:
                        severity = "Unknown"
                        missing_severity.add(issue_type)
                    severity_count[severity] += 1
                    
                    # 获取功能影响类别
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
    """计算量化质量分数（分数越高问题越严重）"""
    # 权重配置（包含Unknown类型）
    SEVERITY_WEIGHTS = {
        "Error": 5,
        "Warning": 3,
        "Info": 2,
        "Ignore": 1,
        "Unknown": 1  # 
    }
    
    total_score = 0
    for severity, count in severity_distribution.items():
        weight = SEVERITY_WEIGHTS.get(severity, 0)  # 未配置的严重级别权重为0
        total_score += count * weight
    return total_score

def process_file(file_path, severity_mapping, impact_mapping):
    """处理单个文件并返回统计结果"""
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
        "smell_details": dict(smell_details),  # 添加详细问题信息
        "no_smell_count": no_smell_count,      # 添加完全没有问题的计数
        "total_files": len(data)               # 添加总文件数
    }

def calculate_ratio(base_result, processed_result):
    """计算处理结果相对于基础结果的比率（处理后/处理前）"""
    def calculate_rate(base_value, processed_value):
        if base_value == 0:
            # 避免除以零错误，如果基数为0
            if processed_value == 0:
                return 1.0  # 0/0 = 1
            else:
                return float('inf')  # 非零除以无穷大
        return processed_value / base_value * 100
    
    # 计算问题类型的净变化
    base_smell_types = set(base_result["smell_distribution"].keys())
    processed_smell_types = set(processed_result["smell_distribution"].keys())
    
    # 完全消除的问题类型
    completely_removed_types = base_smell_types - processed_smell_types
    
    # 新增的问题类型
    newly_introduced_types = processed_smell_types - base_smell_types
    
    # 计算共同类型中的数量变化
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
    
    # 改进的净变化计算：考虑类型变化和数量变化
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
        "net_smell_type_change": {  # 改进：包含数量变化的净变化
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
    
    # 计算严重级别分布的比率
    for severity, base_count in base_result["severity_distribution"].items():
        processed_count = processed_result["severity_distribution"].get(severity, 0)
        ratio_results["severity_distribution"][severity] = {
            "base": base_count,
            "processed": processed_count,
            "ratio": calculate_rate(base_count, processed_count)
        }
    
    # 计算功能影响分布的比率
    for impact, base_count in base_result["impact_distribution"].items():
        processed_count = processed_result["impact_distribution"].get(impact, 0)
        ratio_results["impact_distribution"][impact] = {
            "base": base_count,
            "processed": processed_count,
            "ratio": calculate_rate(base_count, processed_count)
        }
    
    # 比较基础文件和处理后文件的问题类型变化
    base_smells = set(base_result["smell_distribution"].keys())
    processed_smells = set(processed_result["smell_distribution"].keys())
    
    # 新增的问题类型
    added_smells = processed_smells - base_smells
    for smell in added_smells:
        ratio_results["added_smells"].append({
            "type": smell,
            "count": processed_result["smell_distribution"][smell],
            "examples": processed_result["smell_details"].get(smell, [])[:3]  # 最多显示3个例子
        })
    
    # 减少的问题类型
    removed_smells = base_smells - processed_smells
    for smell in removed_smells:
        ratio_results["removed_smells"].append({
            "type": smell,
            "count": base_result["smell_distribution"][smell],
            "examples": base_result["smell_details"].get(smell, [])[:3]  # 最多显示3个例子
        })
    
    # 比较相同问题类型的数量变化
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
    """处理一个组（star1000+）中的所有文件"""
    # 处理基础文件
    base_result = process_file(base_file, severity_mapping, impact_mapping)
    if not base_result:
        print(f"Failed to process base file: {base_file}")
        return []
    
    # 处理所有的处理文件
    processed_results = [process_file(file, severity_mapping, impact_mapping) for file in processed_files]
    processed_results = [res for res in processed_results if res]  # 过滤失败的
    
    # 计算所有处理文件相对于基础文件的比率（处理后/处理前）
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
    """打印组的汇总结果"""
    print(f"\n{'=' * 80}")
    print(f"结果汇总: {group_name.upper()} 数据集")
    print(f"{'=' * 80}")
    
    # 打印基础文件信息
    base_result = group_results["base_result"]
    print(f"\n基础文件: {group_results['base_file']}")
    print(f"  总文件数: {base_result['total_files']}")
    print(f"  无问题文件数: {base_result['no_smell_count']} ({base_result['no_smell_count']/base_result['total_files']*100:.2f}%)")
    print(f"  总问题数: {base_result['total_smells']}")
    print(f"  问题类型数: {base_result['unique_smell_types']}")
    print(f"  加权评分: {base_result['quality_score']}")
    
    # 打印各处理文件相对于基础文件的比率和原始值
    print(f"\n{'文件':<60}{'总问题数':>15}{'比率(%)':>15}{'类型数':>15}{'比率(%)':>15}{'加权评分':>15}{'比率(%)':>15}{'无问题文件':>15}{'比率(%)':>15}")
    
    # 先按总问题比率排序（比率越低越好）
    sorted_pairs = sorted(
        zip(group_results["processed_results"], group_results["ratio_results"]),
        key=lambda x: x[1]['total_smells']['ratio'] 
    )
    
    # 打印表格
   # 打印表格
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
        
        # 获取改进的净变化信息
        net_change_info = ratio_result['net_smell_type_change']
        net_change = net_change_info['net_change']
        removed_count = net_change_info['completely_removed']
        introduced_count = net_change_info['newly_introduced']
        count_decreased = net_change_info['count_decreased']
        count_increased = net_change_info['count_increased']
        
        # 改进的净变化显示：包含类型变化和数量变化
        net_change_display = f"{net_change:+d} ({removed_count+count_decreased}↓/{introduced_count+count_increased}↑)"
        
        file_name = os.path.basename(ratio_result['file'])
        
        # 标记最优结果
        rank_marker = "★" if idx == 0 else ""
        print(f"{file_name:<60}{total_processed:>15}{total_ratio:>15.2f}%{net_change_display:>15}{quality_processed:>15}{quality_ratio:>15.2f}%{no_smell_processed:>15}{no_smell_ratio:>15.2f}%{rank_marker:>5}")    # 打印详细分布信息和变化类型
# 打印详细分布信息和变化类型
    for processed_result, ratio_result in sorted_pairs:
        net_change_info = ratio_result['net_smell_type_change']
        net_change = net_change_info['net_change']
        removed_count = net_change_info['completely_removed']
        introduced_count = net_change_info['newly_introduced']
        count_decreased = net_change_info['count_decreased']
        count_increased = net_change_info['count_increased']
        
        print(f"\n文件: {os.path.basename(ratio_result['file'])}")
        print(f"  问题类型净变化: {net_change:+d} (完全消除{removed_count}种 + 数量减少{count_decreased}种 - 新增{introduced_count}种 - 数量增加{count_increased}种)")
        print(f"  无问题文件数: {ratio_result['no_smell_files']['processed']}/{ratio_result['no_smell_files']['base']} ({ratio_result['no_smell_files']['ratio']:.2f}%)")
        
        # # 打印数量增加的问题类型
        # if net_change_info['count_increased_details']:
        #     print(f"  数量增加的问题类型 ({len(net_change_info['count_increased_details'])}种):")
        #     for item in net_change_info['count_increased_details']:
        #         print(f"    {item['type']}: +{item['increase']} (从{item['from']}到{item['to']})")
        
        # # 打印数量减少的问题类型  
        # if net_change_info['count_decreased_details']:
        #     print(f"  数量减少的问题类型 ({len(net_change_info['count_decreased_details'])}种):")
        #     for item in net_change_info['count_decreased_details']:
        #         print(f"    {item['type']}: -{item['decrease']} (从{item['from']}到{item['to']})")
        
        # # 原有的新增和减少问题类型打印保持不变 ...
        # # 打印新增的问题类型
        # if ratio_result["added_smells"]:
        #     print(f"  新增的问题类型 ({len(ratio_result['added_smells'])}种):")
        #     for added in ratio_result["added_smells"]:
        #         if "count_change" in added:
        #             print(f"    {added['type']}: 数量变化 {added['count_change']} (从{added['old_count']}到{added['new_count']})")
        #         else:
        #             print(f"    {added['type']}: 新增 {added['count']}个")
        #         # 打印示例
        #         for example in added.get("examples", [])[:1]:
        #             pass
        #             # print(f"      - 示例: {example}")
        
        # # 打印减少的问题类型
        # if ratio_result["removed_smells"]:
        #     print(f"  减少的问题类型 ({len(ratio_result['removed_smells'])}种):")
        #     for removed in ratio_result["removed_smells"]:
        #         if "count_change" in removed:
        #             print(f"    {removed['type']}: 数量变化 {removed['count_change']} (从{removed['old_count']}到{removed['new_count']})")
        #         else:
        #             print(f"    {removed['type']}: 完全消除 (原{removed['count']}个)")
        #         # 打印示例
        #         for example in removed.get("examples", [])[:1]:
        #             pass
                    # print(f"      - 示例: {example}")
    return sorted_pairs

def main():
    # 读取严重级别映射和功能影响映射
    severity_file = "evaluate/level.json"
    severity_data = read_json_to_dict(severity_file)
    if not severity_data:
        print(f"Failed to load severity mapping from {severity_file}")
        return
    
    # 创建两个映射字典
    severity_mapping = {}
    impact_mapping = {}
    
    for item in severity_data:
        if "id" in item:
            rule_id = item["id"]
            severity_mapping[rule_id] = item.get("defaultSeverity", "Unknown")
            impact_mapping[rule_id] = item.get("function_impact", "Unknown")
    
    # 定义Star1000+的文件分组
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
    
    # 处理所有组
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
    
    # 分析每个模型在不同数据集上的表现
    print("\n\n" + "="*80)
    print("模型表现综合分析")
    print("="*80)
    
    # 收集每个模型的比率数据
    model_performance = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # 定义我们关心的所有指标
    severity_metrics = ["Error", "Warning", "Info"]
    impact_metrics = ["Security", "Correctness", "Maintainability", "Efficiency"]
    
    for group_name, results in all_results.items():
        for processed_result, ratio_result in zip(results["processed_results"], results["ratio_results"]):
            file_name = os.path.basename(ratio_result["file"])
            
            # 提取模型名称和方法
            model_info = extract_model_and_method(file_name)
            
            if model_info:
                model_name = model_info["model"]
                method_name = model_info["method"]
                                  
                if "star1000+" in group_name:
                    # 收集基本指标
                    model_performance[model_name][method_name]["star_total"] = ratio_result["total_smells"]["processed"]
                    model_performance[model_name][method_name]["star_total_ratio"] = ratio_result["total_smells"]["ratio"]
                    model_performance[model_name][method_name]["star_types"] = ratio_result["unique_smell_types"]["processed"]
                    model_performance[model_name][method_name]["star_types_ratio"] = ratio_result["unique_smell_types"]["ratio"]
                    model_performance[model_name][method_name]["star_quality"] = ratio_result["quality_score"]["processed"]
                    model_performance[model_name][method_name]["star_quality_ratio"] = ratio_result["quality_score"]["ratio"]
                    model_performance[model_name][method_name]["star_no_smell"] = ratio_result["no_smell_files"]["processed"]
                    model_performance[model_name][method_name]["star_no_smell_ratio"] = ratio_result["no_smell_files"]["ratio"]
                    
                    # 收集严重级别比率
                    for severity in severity_metrics:
                        if severity in ratio_result["severity_distribution"]:
                            model_performance[model_name][method_name][f"star_{severity.lower()}"] = ratio_result["severity_distribution"][severity]["processed"]
                            model_performance[model_name][method_name][f"star_{severity.lower()}_ratio"] = ratio_result["severity_distribution"][severity]["ratio"]
                    
                    # 收集功能影响比率
                    for impact in impact_metrics:
                        if impact in ratio_result["impact_distribution"]:
                            model_performance[model_name][method_name][f"star_{impact.lower()}"] = ratio_result["impact_distribution"][impact]["processed"]
                            model_performance[model_name][method_name][f"star_{impact.lower()}_ratio"] = ratio_result["impact_distribution"][impact]["ratio"]
    
    # 打印模型性能比较表
    print("\n模型综合性能比较:")
    print(f"{'模型':<15}{'方法':<25}{'数据集':<10}{'总问题数':>15}{'比率(%)':>15}{'类型数':>15}{'比率(%)':>15}{'评分':>15}{'比率(%)':>15}{'无问题文件':>15}{'比率(%)':>15}")
    
    # 收集所有模型方法组合
    model_methods = []
    for model, methods in model_performance.items():
        for method in methods:
            model_methods.append((model, method))
    
    # 按模型名称和方法排序
    model_methods.sort(key=lambda x: (x[0], x[1]))
    
    for model, method in model_methods:
        data = model_performance[model][method]
        
        # Star1000+数据
        print(f"{'':<15}{'':<25}{'Star':<10}", end="")
        print(f"{data.get('star_total', 'N/A'):>15}", end="")
        print(f"{data.get('star_total_ratio', 'N/A'):>15.2f}%", end="")
        print(f"{data.get('star_types', 'N/A'):>15}", end="")
        print(f"{data.get('star_types_ratio', 'N/A'):>15.2f}%", end="")
        print(f"{data.get('star_quality', 'N/A'):>15}", end="")
        print(f"{data.get('star_quality_ratio', 'N/A'):>15.2f}%", end="")
        print(f"{data.get('star_no_smell', 'N/A'):>15}", end="")
        print(f"{data.get('star_no_smell_ratio', 'N/A'):>15.2f}%")
        
        # 打印分隔线
        print("-" * 180)



    # 添加严重性指标表格
    print("\n模型综合性能比较 (包含严重性指标):")
    print(f"{'模型':<15}{'方法':<20}{'数据集':<10}{'总问题':>8}{'比率':>8}{'Error':>10}{'比率':>8}{'Warning':>10}{'比率':>8}{'Info':>10}{'比率':>8}")

    for model, method in model_methods:
        data = model_performance[model][method]
        
        # Star1000+数据
        print(f"{model:<15}{method:<20}{'Star':<10}", end="")
        print(f"{data.get('star_total', 'N/A'):>8}", end="")
        print(f"{data.get('star_total_ratio', 'N/A'):>8.1f}%", end="")
        
        # 添加严重性指标
        for severity in ["error", "warning", "info"]:
            severity_key = f"star_{severity}"
            ratio_key = f"star_{severity}_ratio"
            print(f"{data.get(severity_key, 'N/A'):>10}", end="")
            print(f"{data.get(ratio_key, 'N/A'):>8.1f}%", end="")
        
        print()  # 换行

                # 修改表头，添加impact指标
    print("\n模型综合性能比较 (包含Impact指标):")
    print(f"{'模型':<15}{'方法':<20}{'数据集':<10}{'总问题':>8}{'比率':>8}{'Security':>10}{'比率':>8}{'Correctness':>12}{'比率':>8}{'Maintain':>10}{'比率':>8}{'Efficiency':>10}{'比率':>8}")

    for model, method in model_methods:
        data = model_performance[model][method]
        
        # Star1000+数据
        print(f"{model:<15}{method:<20}{'Star':<10}", end="")
        print(f"{data.get('star_total', 'N/A'):>8}", end="")
        print(f"{data.get('star_total_ratio', 'N/A'):>8.1f}%", end="")
        
        # 添加impact指标
        for impact in ["security", "correctness", "maintainability", "efficiency"]:
            impact_key = f"star_{impact}"
            ratio_key = f"star_{impact}_ratio"
            print(f"{data.get(impact_key, 'N/A'):>10}", end="")
            print(f"{data.get(ratio_key, 'N/A'):>8.1f}%", end="")
        
        print()  # 换行
        

def extract_model_and_method(file_name):
    """从文件名中提取模型名称和方法"""
    if "qwen3_" in file_name:
        # 提取模型大小和方法
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