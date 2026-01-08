# Dockerfile Generation and Repair (DGR) System

This repository contains a comprehensive system for analyzing, repairing, and evaluating Dockerfiles using various methods including large language models and rule-based approaches.

## 📁 Project Structure

### 1. Dataset Construction (`dataset_build/`)
Collects high-popularity repositories with Dockerfiles for analysis.

| File | Description |
|------|-------------|
| `star1000+_repo_with_dockerfile.py` | Fetches repositories with 1000+ stars that contain Dockerfiles |
| `star1000+_context_get.py` | Clones qualifying repositories for analysis |

### 2. Fast-build Dataset (`dataset_fast_build/`)
Identifies Dockerfiles that can be built quickly for efficient experimentation.

| File | Description |
|------|-------------|
| `star1000+_fast_build_dockerfile_get.py` | Tests and identifies quickly buildable Dockerfiles |
| `star1000+_fast_build_repo_copy.py` | Copies code from repositories with fast-building Dockerfiles |

### 3. Evaluation Module (`evaluate/`)
Tools for evaluating the effectiveness of Dockerfile repairs.

| File | Description |
|------|-------------|
| `build_analyzer_pure_rules.py` | Analyzes which stage of Dockerfile builds fail |
| `build.py` | Executes build tests for Dockerfiles |
| `evaluate_smell_count.py` | Evaluates repair effectiveness by counting Dockerfile smells |
| `hadolint.py` | Runs Hadolint to generate evaluation reports on Dockerfile smells |
| `train_data_get.py` | Generate dataset for Fine-tune |
| `level.json` | Functionality and severity evaluation metrics |

### 4. Storage Directories
| Directory | Purpose |
|-----------|---------|
| `evaluate_result/` | Stores repair results and build outcomes |

### 5. Repair Methods (`repair_methods/`)
Implementation of various Dockerfile repair techniques.

| Method | File | Description |
|--------|------|-------------|
| **Model Evaluation** | `Distillation/5_floder.py` | 5-fold cross-validation for evaluating fine-tuned models |
| **Error Correction** | `build_repair.py` | Error correction methods for build failures |
| **DockerCleaner** | `dockercleaner.py` | Implementation of DockerCleaner repair method |
| **Parfum** | `parfum.py` | Implementation of Parfum rule-based repair method |
| **MSR'25 ICL** | `msr25_icl.py` | Implementation of MSR'25 in-context learning method |
| **DGR (LLM)** | `hd_LLM.py` | Dockerfile Generation and Repair using LLMs |
| **Hybrid DGR** | `hd_tool_LLM.py` | Hybrid DGR combining LLMs with rule-based tools |

## 🔄 Workflow

1. **Data Collection** → **Fast-build Filtering** → **Repair Application** → **Evaluation**
