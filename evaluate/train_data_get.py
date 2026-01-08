import json
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional

def read_json(file_path: str) -> List[Dict]:
    """读取JSON文件，增加类型提示和更详细的错误处理"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误 {file_path}: {e}")
        return []
    except Exception as e:
        print(f"读取文件错误 {file_path}: {e}")
        return []

def calculate_smell_score(issues: List[str], severity_mapping: Dict[str, str]) -> int:
    """计算单个Dockerfile的问题分数，优化解析逻辑"""
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
    """从完整路径中提取Dockerfile文件名，增加路径处理健壮性"""
    try:
        return os.path.basename(path)
    except Exception:
        return path

def analyze_best_solutions(dockerfile_scores: Dict[str, List[Tuple[int, int, str]]]) -> Tuple[List[int], List[Dict]]:
    """分析最优解分布，现在只返回所有最优解的方法索引和详细信息"""
    all_best = []   # 所有最优解的方法索引
    best_details = []   # 最优解的详细信息
    
    for dockerfile_name, scores in dockerfile_scores.items():
        if not scores:
            continue
            
        # 找出最低分
        min_score = min(score for (_, score, _) in scores)
        
        # 找出所有最优解的方法
        best_methods = [file_idx for (file_idx, score, _) in scores if score == min_score]
        all_best.extend(best_methods)
        
        # 保存最优解详细信息
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
    """计算统计结果，提取为独立函数"""
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
    
    # 按比例降序排序（比例相同时按原始顺序排序）
    results.sort(key=lambda x: (-x["percentage"], x["rank"]))
    return results, total_best

def print_analysis_results(results: Dict, analysis_name: str) -> None:
    """打印分析结果，优化输出格式"""
    title = f"{analysis_name}分析结果"

    print(f"\n{title}")
    print("=" * 120)
    print(f"{'排名':<5} {'修复方法':<60} {'最优解次数':<12} {'占比(%)':<10} {'文件顺序':<10} {'覆盖率':<10}")
    print("-" * 120)
    
    for i, result in enumerate(results["results"], 1):
        print(f"{i:<5} {result['file_name']:<60} {result['best_count']:<12} "
              f"{result['percentage']:<10.2f} #{result['rank']:<10} "
              f"{(result['best_count']/results['dockerfile_count'])*100:.1f}%")
    
    print("=" * 120)
    print(f"统计摘要: 总Dockerfile数量={results['dockerfile_count']} | "
          f"最优解总数={results['total_best']} | "
          f"平均覆盖率={(results['total_best']/results['dockerfile_count'])*100:.1f}%")

def generate_all_solutions_report(
    dockerfile_scores: Dict[str, List[Tuple[int, int, str]]], 
    file_paths: List[str],
    output_file: Optional[str] = None
) -> None:
    """
    生成所有解决方案的详细报告，包括最优解和非最优解
    :param dockerfile_scores: Dockerfile评分数据
    :param file_paths: 所有方法文件路径列表
    :param output_file: 输出文件路径，如果为None则打印到控制台
    """
    report = []
    
    for dockerfile_name, scores in dockerfile_scores.items():
        if not scores:
            continue
            
        # 找出最低分(最优解)
        min_score = min(score for (_, score, _) in scores)
        
        # 收集所有方法的得分信息
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
        
        # 按分数排序
        methods_info.sort(key=lambda x: x["score"])
        
        report.append({
            "dockerfile_name": dockerfile_name,
            "min_score": min_score,
            "methods": methods_info,
            "method_count": len(methods_info)
        })
    
    # 按Dockerfile名称排序
    report.sort(key=lambda x: x["dockerfile_name"])
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"所有解决方案的详细报告已保存到 {output_file}")
    else:
        print("\n所有Dockerfile的解决方案详细报告:")
        print("=" * 150)
        for item in report:
            print(f"\nDockerfile: {item['dockerfile_name']} (最低分: {item['min_score']})")
            print("-" * 150)
            print(f"{'方法名称':<30} {'得分':<8} {'是否最优':<10} {'差距':<8} {'方法索引':<10} {'原始Dockerfile路径':<60}")
            print("-" * 150)
            for method in item["methods"]:
                print(f"{method['method']:<30} {method['score']:<8} "
                      f"{'✓' if method['is_best'] else '✗':<10} "
                      f"{method['gap']:<8}"
                      f"{method['method_index']:<10} {method['dockerfile_repair_path']:<60}")
        print("=" * 150)

def process_dataset(file_paths: List[str], severity_file: str, dataset_name: str) -> Dict:
    """处理单个数据集，添加生成所有解决方案报告的功能"""
    print(f"\n正在处理数据集: {dataset_name}")
    
    # 读取严重级别映射
    severity_data = read_json(severity_file)
    severity_mapping = {item['id']: item['defaultSeverity'] for item in severity_data}
    
    # 收集所有Dockerfile的分数（按文件名分组）
    dockerfile_scores = defaultdict(list)
    
    # 为每个文件收集所有Dockerfile的分数
    for file_idx, file_path in enumerate(file_paths):
        data = read_json(file_path)
        for item in data:
            dockerfile_path = item.get("dockerfile_path", "")
            dockerfile_name = get_dockerfile_name(dockerfile_path)
            score = calculate_smell_score(item.get("issues", []), severity_mapping)
            dockerfile_scores[dockerfile_name].append((file_idx, score, dockerfile_path))
    
    # 分析最优解 - 现在只返回所有最优解
    all_best, best_details = analyze_best_solutions(dockerfile_scores)
    
    # 计算统计结果
    results, total_best = get_stats(all_best, file_paths)

    # 更新best_details中的方法名
    for detail in best_details:
        detail["best_method"] = os.path.basename(file_paths[detail["method_index"]])
    
    # 生成所有解决方案的报告（包括非最优解）
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
        "all_scores": dockerfile_scores  # 保留所有评分数据
    }

def main():
    # 配置文件路径
    severity_file = "evaluate/level.json"
    
    # 只处理Star1000+数据集
    star_dataset = {
        "name": "Star1000+ Dockerfiles",
        "files": [
            "evaluate_result/dataset_fast_star1000+_dockerfile.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_parfum.json",
            # "evaluate_result/dataset_fast_star1000+_dockerfile_dockercleaner.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_LLM.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_LLM_1.json",
            "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_235b_hd_LLM_2.json",
            # "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_32b_hd_LLM_nothink.json",
            # "evaluate_result/dataset_fast_star1000+_dockerfile_qwen3_14b_hd_LLM_nothink.json",
        ]
    }
    
    # 处理Star1000+数据集
    results = process_dataset(star_dataset["files"], severity_file, star_dataset["name"])
    print_analysis_results(results, star_dataset["name"])

if __name__ == "__main__":
    main()
