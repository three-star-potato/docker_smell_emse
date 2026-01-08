import json
import random
import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
import pandas as pd
from datasets import Dataset
import re
import argparse

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def build_prompt(dockerfile_name):
    """构建训练样本的prompt"""
    prompt_data = []
    
    # 只使用star1000+目录
    original_dockerfile_path = os.path.join("dataset_fast", "star1000+_dockerfile", dockerfile_name)
    smells_file = os.path.join("evaluate_result", "dataset_fast_star1000+_dockerfile.json")
    
    # 检查文件是否存在
    if not os.path.exists(original_dockerfile_path):
        print(f"Warning: Dockerfile not found: {dockerfile_name}")
        return prompt_data
    
    # 读取原始Dockerfile内容
    with open(original_dockerfile_path, 'r', encoding='utf-8') as file:
        original_content = file.read()
    
    # 读取并查找对应的smells
    smells = ""
    with open(smells_file, 'r', encoding='utf-8') as file:
        ask_smell = json.load(file)
        for item in ask_smell:
            if os.path.basename(item["dockerfile_path"]) == dockerfile_name:
                smells = "\n".join(item["issues"])
                break
    
    # 构建输入提示
    input_prompt = (
        f"Original Dockerfile:\n```dockerfile\n{original_content}\n```\n\n"
        f"Smells need to fix:\n{smells}\n\n"
        "Return ONLY the modified Dockerfile that:\n"
        "1. Is directly buildable with `docker build`\n"
        "2. Preserves all original functionality\n"
        "3. NO new features added\n\n"
        "4. Format:\n```dockerfile\n...\n```"
    )
    
    # 读取所有解决方案并查找匹配的修复
    with open('evaluate_result/all_solutions_report_Star1000+_Dockerfilesnoparum.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for item in data:
        if item['dockerfile_name'] == dockerfile_name:
            for method in item["methods"]:
                output_path = method['dockerfile_repair_path']
                try:
                    with open(output_path, 'r', encoding='utf-8') as file:
                        repair_dockerfile_content = file.read()
                        prompt_data.append({
                            "input": input_prompt,
                            "output": repair_dockerfile_content,
                            "score": method['score'],
                            "gap": method['gap']
                        })
                except FileNotFoundError:
                    print(f"Warning: Repair file not found: {output_path}")
                    continue
    
    return prompt_data

def create_fold_datasets(dockerfile_names, output_base_dir, n_folds=5):
    """创建五折交叉验证的数据集"""
    
    # 使用KFold进行分割
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1010)
    
    fold_datasets = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dockerfile_names)):
        print(f"\nProcessing Fold {fold + 1}/{n_folds}")
        
        # 获取训练集和验证集的Dockerfile名称
        train_files = [dockerfile_names[i] for i in train_idx]
        val_files = [dockerfile_names[i] for i in val_idx]
        
        # 为当前fold创建输出目录
        fold_dir = os.path.join(output_base_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # 保存训练集和验证集文件列表
        with open(os.path.join(fold_dir, 'train_files.txt'), 'w') as f:
            f.write('\n'.join(train_files))
        with open(os.path.join(fold_dir, 'val_files.txt'), 'w') as f:
            f.write('\n'.join(val_files))
        
        # 构建训练集数据
        train_data = []
        for dockerfile_name in tqdm(train_files, desc="Building training data"):
            prompt_data = build_prompt(dockerfile_name)
            for item in prompt_data:
                if item["gap"] == 0:  # 只使用gap_zero数据
                    alpaca_item = {
                        "instruction": "As a Docker expert, please fix the following Dockerfile issues",
                        "input": item["input"],
                        "output": f"Optimized Dockerfile:\n```dockerfile\n{item['output']}\n```",
                    }
                    train_data.append(alpaca_item)
        
        # 保存数据集
        with open(os.path.join(fold_dir, 'train.json'), 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        fold_info = {
            'fold': fold + 1,
            'fold_dir': fold_dir,
            'train_files': train_files,
            'val_files': val_files,
            'train_size': len(train_data),
            'val_size': len(val_files)  # 验证集大小等于验证集文件数
        }
        
        fold_datasets.append(fold_info)
        
        print(f"Fold {fold + 1}: Train samples={len(train_data)}, Val files={len(val_files)}")
    
    return fold_datasets

def load_dataset_from_json(json_path):
    """从JSON文件加载数据集"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为pandas DataFrame
    instructions = []
    inputs = []
    outputs = []
    
    for item in data:
        instructions.append(item['instruction'])
        inputs.append(item['input'])
        outputs.append(item['output'])
    
    df = pd.DataFrame({
        'instruction': instructions,
        'input': inputs,
        'output': outputs
    })
    
    return df

def process_func(example, tokenizer, max_length=4096):
    """数据处理函数"""
    # 构建Qwen3格式的对话消息
    messages = [
        {"role": "system", "content": "As a Docker expert, please fix the following Dockerfile issues"},
        {"role": "user", "content": example['input']},
        {"role": "assistant", "content": example['output']}
    ]
    
    # 使用apply_chat_template应用模板
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # Tokenize整个文本
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    # 创建标签（只计算assistant回复部分的loss）
    assistant_start = text.find("<|im_start|>assistant")
    if assistant_start != -1:
        # Tokenize assistant之前的内容
        prefix = text[:assistant_start]
        prefix_tokens = tokenizer(
            prefix,
            truncation=False,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        prefix_len = len(prefix_tokens['input_ids'])
        
        # 创建labels，assistant之前的部分设为-100
        labels = [-100] * len(tokenized['input_ids'])
        # 只保留assistant部分的标签
        assistant_tokens = tokenized['input_ids'][prefix_len:]
        labels[prefix_len:prefix_len + len(assistant_tokens)] = assistant_tokens
    else:
        # 如果没有找到assistant标记，全部计算loss
        labels = tokenized['input_ids'].copy()
    
    return {
        "input_ids": tokenized['input_ids'],
        "attention_mask": tokenized['attention_mask'],
        "labels": labels
    }

def train_fold_model(fold_info, base_model_path, output_base_dir, max_length=4096):
    """训练单个fold的模型"""
    fold = fold_info['fold']
    fold_dir = fold_info['fold_dir']
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")
    
    # 创建模型输出目录
    model_output_dir = os.path.join(output_base_dir, f"fold_{fold}_model")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 加载tokenizer和模型
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    
    # 加载数据集
    print("Loading training dataset...")
    train_json_path = os.path.join(fold_dir, 'train.json')
    train_df = load_dataset_from_json(train_json_path)
    train_ds = Dataset.from_pandas(train_df)
    
    # 处理数据
    print("Processing training data...")
    train_dataset = train_ds.map(
        lambda x: process_func(x, tokenizer, max_length),
        remove_columns=train_ds.column_names
    )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="no",  # 不进行验证
        save_steps=200,
        logging_steps=50,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_dir=os.path.join(model_output_dir, "logs"),
        report_to=None,
        save_total_limit=1,
        bf16=True,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    
    # 创建data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(model_output_dir)
    
    print(f"Fold {fold} training completed! Model saved to {model_output_dir}")
    
    return model_output_dir

def extract_dockerfile_content(model_output):
    """从模型输出中提取Dockerfile内容"""
    # 尝试匹配代码块格式
    code_block_pattern = r'```dockerfile\n(.*?)\n```'
    match = re.search(code_block_pattern, model_output, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # 如果没有找到dockerfile代码块，尝试普通代码块
    code_block_pattern2 = r'```\n(.*?)\n```'
    match2 = re.search(code_block_pattern2, model_output, re.DOTALL)
    
    if match2:
        return match2.group(1).strip()
    
    # 如果还是没有代码块，尝试寻找以FROM开头的Dockerfile内容
    from_pattern = r'(FROM\s+.*?)(?=\n```|\n$|\Z)'
    match3 = re.search(from_pattern, model_output, re.DOTALL)
    
    if match3:
        # 提取从FROM开始到文件结束的内容
        from_index = model_output.find('FROM')
        if from_index != -1:
            potential_dockerfile = model_output[from_index:].strip()
            potential_dockerfile = re.sub(r'\n```.*', '', potential_dockerfile)
            return potential_dockerfile
    
    # 如果所有方法都失败，返回原始输出
    return model_output.strip()

def generate_dockerfiles_for_fold_sequential(model_path, val_files, dockerfile_base_path, output_dir):
    """为验证集生成修复后的Dockerfile - 逐个生成版本"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和tokenizer
    print("Loading model for Dockerfile generation...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # 生成配置
    generation_config = {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.05,
    }
    
    # 加载验证集Dockerfiles
    def load_val_dockerfiles(val_files, dockerfile_base_path):
        val_dockerfiles = []
        smells_file = os.path.join("evaluate_result", "dataset_fast_star1000+_dockerfile.json")
        
        with open(smells_file, 'r', encoding='utf-8') as f:
            smells_data = json.load(f)
        
        for dockerfile_name in val_files:
            dockerfile_path = os.path.join(dockerfile_base_path, dockerfile_name)
            
            if not os.path.exists(dockerfile_path):
                print(f"Warning: Dockerfile not found: {dockerfile_name}")
                continue
            
            with open(dockerfile_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            smells = ""
            for item in smells_data:
                if os.path.basename(item["dockerfile_path"]) == dockerfile_name:
                    smells = "\n".join(item["issues"])
                    break
            
            val_dockerfiles.append({
                "name": dockerfile_name,
                "original_content": original_content,
                "smells": smells,
            })
        
        return val_dockerfiles
    
    val_data = load_val_dockerfiles(val_files, dockerfile_base_path)
    print(f"Loaded {len(val_data)} validation Dockerfiles")
    
    # 逐个生成修复后的Dockerfile
    for item in tqdm(val_data, desc="Generating repaired Dockerfiles (sequential)"):
        try:
            # 构建输入提示
            input_prompt = (
                f"Original Dockerfile:\n```dockerfile\n{item['original_content']}\n```\n\n"
                f"Smells need to fix:\n{item['smells']}\n\n"
                "Return ONLY the modified Dockerfile that:\n"
                "1. Is directly buildable with `docker build`\n"
                "2. Preserves all original functionality\n"
                "3. NO new features added\n\n"
                "4. Format:\n```dockerfile\n...\n```"
            )
            
            # 构建对话消息
            messages = [
                {"role": "system", "content": "As a Docker expert, please fix the following Dockerfile issues"},
                {"role": "user", "content": input_prompt}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 解码回复
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # 提取assistant回复
            if '<|im_start|>assistant' in response:
                response_parts = response.split('<|im_start|>assistant')
                if len(response_parts) > 1:
                    assistant_response = response_parts[-1]
                    if '<|im_end|>' in assistant_response:
                        assistant_response = assistant_response.split('<|im_end|>')[0]
                    model_output = assistant_response.strip()
                else:
                    model_output = response
            else:
                model_output = response
            
            # 提取Dockerfile内容
            dockerfile_content = extract_dockerfile_content(model_output)
            
            # 保存修复后的Dockerfile
            output_filename = f"{item['name']}"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dockerfile_content)
            
        except Exception as e:
            print(f"Error processing {item['name']}: {str(e)}")
            # 保存错误信息
            error_filename = f"{item['name']}.error"
            error_path = os.path.join(output_dir, error_filename)
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Error generating Dockerfile: {str(e)}")
    
    print(f"Generated {len(val_data)} repaired Dockerfiles in {output_dir}")
    
    return len(val_data)

def generate_dockerfiles_for_fold_batch(model_path, val_files, dockerfile_base_path, output_dir, batch_size=4):
    """为验证集生成修复后的Dockerfile - 批量生成版本"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和tokenizer
    print("Loading model for Dockerfile generation...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # 生成配置
    generation_config = {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.05,
    }
    
    # 加载验证集Dockerfiles
    def load_val_dockerfiles(val_files, dockerfile_base_path):
        val_dockerfiles = []
        smells_file = os.path.join("evaluate_result", "dataset_fast_star1000+_dockerfile.json")
        
        with open(smells_file, 'r', encoding='utf-8') as f:
            smells_data = json.load(f)
        
        for dockerfile_name in val_files:
            dockerfile_path = os.path.join(dockerfile_base_path, dockerfile_name)
            
            if not os.path.exists(dockerfile_path):
                print(f"Warning: Dockerfile not found: {dockerfile_name}")
                continue
            
            with open(dockerfile_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            smells = ""
            for item in smells_data:
                if os.path.basename(item["dockerfile_path"]) == dockerfile_name:
                    smells = "\n".join(item["issues"])
                    break
            
            val_dockerfiles.append({
                "name": dockerfile_name,
                "original_content": original_content,
                "smells": smells,
            })
        
        return val_dockerfiles
    
    val_data = load_val_dockerfiles(val_files, dockerfile_base_path)
    print(f"Loaded {len(val_data)} validation Dockerfiles")
    
    # 批量生成修复后的Dockerfile
    for i in tqdm(range(0, len(val_data), batch_size), desc="Generating repaired Dockerfiles (batch)"):
        batch_items = val_data[i:i+batch_size]
        batch_inputs = []
        
        # 准备批量输入
        for item in batch_items:
            input_prompt = (
                f"Original Dockerfile:\n```dockerfile\n{item['original_content']}\n```\n\n"
                f"Smells need to fix:\n{item['smells']}\n\n"
                "Return ONLY the modified Dockerfile that:\n"
                "1. Is directly buildable with `docker build`\n"
                "2. Preserves all original functionality\n"
                "3. NO new features added\n\n"
                "4. Format:\n```dockerfile\n...\n```"
            )
            
            messages = [
                {"role": "system", "content": "As a Docker expert, please fix the following Dockerfile issues"},
                {"role": "user", "content": input_prompt}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_inputs.append(text)
        
        # 批量编码
        inputs = tokenizer(
            batch_inputs, 
            padding=True, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        ).to(model.device)
        
        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )
        
        # 批量解码和处理
        for j, output in enumerate(outputs):
            item_index = i + j
            if item_index >= len(val_data):
                break
                
            response = tokenizer.decode(output, skip_special_tokens=False)
            
            # 提取assistant回复
            if '<|im_start|>assistant' in response:
                response_parts = response.split('<|im_start|>assistant')
                if len(response_parts) > 1:
                    assistant_response = response_parts[-1]
                    if '<|im_end|>' in assistant_response:
                        assistant_response = assistant_response.split('<|im_end|>')[0]
                    model_output = assistant_response.strip()
                else:
                    model_output = response
            else:
                model_output = response
            
            # 提取Dockerfile内容
            dockerfile_content = extract_dockerfile_content(model_output)
            
            # 保存修复后的Dockerfile
            output_filename = batch_items[j]['name']
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dockerfile_content)
    
    print(f"Generated {len(val_data)} repaired Dockerfiles in {output_dir}")
    return len(val_data)

def generate_dockerfiles_for_fold(model_path, val_files, dockerfile_base_path, output_dir, batch_size=1):
    """为验证集生成修复后的Dockerfile - 主函数，根据batch_size选择生成方式"""
    if batch_size > 1:
        return generate_dockerfiles_for_fold_batch(model_path, val_files, dockerfile_base_path, output_dir, batch_size)
    else:
        return generate_dockerfiles_for_fold_sequential(model_path, val_files, dockerfile_base_path, output_dir)

def main():
    """主函数 - 简化的五折交叉验证，专注于生成Dockerfile"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="5-Fold Cross Validation for Dockerfile Repair")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Batch size for generation (1 for sequential, >1 for batch generation)")
    parser.add_argument("--base_model", type=str, default="repair_methods/Distillation/Qwen3-0.6B",
                       help="Base model path")
    parser.add_argument("--output_dir", type=str, default="repair_methods/Distillation/cross_validation_simplenoparum",
                       help="Output directory")
    parser.add_argument("--dockerfile_path", type=str, default="dataset_fast/star1000+_dockerfile",
                       help="Dockerfile base path")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    
    args = parser.parse_args()
    
    # 配置参数
    BASE_MODEL_PATH = args.base_model
    OUTPUT_BASE_DIR = args.output_dir
    DOCKERFILE_BASE_PATH = args.dockerfile_path
    N_FOLDS = args.folds
    BATCH_SIZE = args.batch_size
    
    print("=" * 70)
    print("Simplified 5-Fold Cross Validation - Dockerfile Generation")
    print(f"Batch size: {BATCH_SIZE} ({'Batch generation' if BATCH_SIZE > 1 else 'Sequential generation'})")
    print("=" * 70)
    
    # 步骤1: 读取所有Dockerfile名称
    print("\nStep 1: Loading Dockerfile names...")
    dockerfile_names = set()
    with open('evaluate_result/all_solutions_report_Star1000+_Dockerfiles.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            dockerfile_names.add(item['dockerfile_name'])
    
    dockerfile_list = list(dockerfile_names)
    print(f"Total Dockerfiles: {len(dockerfile_list)}")
    
    # 步骤2: 创建五折数据集
    print("\nStep 2: Creating 5-fold datasets...")
    fold_datasets = create_fold_datasets(dockerfile_list, OUTPUT_BASE_DIR, N_FOLDS)
    
    # 步骤3: 进行五折交叉验证
    print("\nStep 3: Starting 5-fold cross-validation...")
    
    for fold_info in fold_datasets:
        fold = fold_info['fold']
        
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold}/5")
        print(f"{'='*60}")
        
        # 训练当前fold的模型
        model_dir = train_fold_model(
            fold_info, 
            BASE_MODEL_PATH, 
            OUTPUT_BASE_DIR
        )
        
        # 为验证集生成修复后的Dockerfile
        dockerfile_output_dir = os.path.join(OUTPUT_BASE_DIR, f"repaired")
        generated_count = generate_dockerfiles_for_fold(
            model_dir,
            fold_info['val_files'],
            DOCKERFILE_BASE_PATH,
            dockerfile_output_dir,
            batch_size=BATCH_SIZE
        )
        
        print(f"Fold {fold}: Generated {generated_count} repaired Dockerfiles")
        
        # 保存fold信息
        fold_summary = {
            'fold': fold,
            'model_path': model_dir,
            'repaired_dockerfiles_dir': dockerfile_output_dir,
            'train_size': fold_info['train_size'],
            'val_files_count': len(fold_info['val_files']),
            'generated_count': generated_count,
            'batch_size': BATCH_SIZE
        }
        
        with open(os.path.join(OUTPUT_BASE_DIR, f'fold_{fold}_summary.json'), 'w') as f:
            json.dump(fold_summary, f, indent=2)
    
    # 步骤4: 生成总体摘要
    print("\nStep 4: Generating overall summary...")
    
    overall_summary = {
        'total_folds': N_FOLDS,
        'batch_size': BATCH_SIZE,
        'folds': []
    }
    
    for fold in range(1, N_FOLDS + 1):
        summary_path = os.path.join(OUTPUT_BASE_DIR, f'fold_{fold}_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                fold_summary = json.load(f)
                overall_summary['folds'].append(fold_summary)
    
    with open(os.path.join(OUTPUT_BASE_DIR, 'overall_summary.json'), 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    # 打印最终结果
    print("\n" + "=" * 70)
    print("5-Fold Cross Validation Completed")
    print("=" * 70)
    
    for fold_summary in overall_summary['folds']:
        print(f"Fold {fold_summary['fold']}:")
        print(f"  Train samples: {fold_summary['train_size']}")
        print(f"  Val files: {fold_summary['val_files_count']}")
        print(f"  Generated Dockerfiles: {fold_summary['generated_count']}")
        print(f"  Batch size: {fold_summary['batch_size']}")
        print(f"  Output directory: {fold_summary['repaired_dockerfiles_dir']}")
        print()
    
    print(f"All repaired Dockerfiles have been saved to respective fold directories.")
    print(f"You can now evaluate the generated Dockerfiles manually.")
    print(f"\nResults saved to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()


# python repair_methods/Distillation/5_floder.py --batch_size 16