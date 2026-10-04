# PathCraft

面向 ComplexWebQuestions（CWQ）和 WebQSP 的语义引导知识库问答系统。包含问题分解、语义路径、Compose、Operator、图检索、答案选择及可选 GLM 分解复核。

本仓库发布运行代码、WebQSP/CWQ 数据及运行所需的本体和训练数据检索索引。历史 trace、逐题 artifact、实验结果、日志、模型权重和 Freebase 数据库不随仓库发布。运行时新生成的输出由 `.gitignore` 排除。

## 目录

- `src/semantic_guided_kbqa/`：问答运行代码及提示样例。
- `configs/`：CWQ 和 WebQSP 配置、模型服务启动配置。
- `data/cwq/`：CWQ 原始训练数据、Semantic/Compose/Operator 训练数据、测试分解和 gold entities。
- `data/webqsp/`：WebQSP 预处理训练/测试数据、Semantic/Compose/Operator 训练数据、测试问题及分解输入。
- `data/indexes/`：当前问答算法依赖的训练数据检索索引，不是逐题执行 trace。
- `ontology/`：Freebase 本体和关系资源。
- `prompts/`：GLM 分解复核提示配置。
- `scripts/`：数据检查、模型服务管理和运行入口。

数据为本项目实际使用的处理版本，分解预测文件是问答输入，不应视为原始人工标注。数据保留其上游权利和适用条款；本仓库未对第三方数据重新授权。配套数据生成工程：https://github.com/zhaohuiyang94-tech/data_generation 。

## 环境和离线检查

问答客户端需要 Python 3.11+，使用 Python 标准库。模型服务器的依赖需另行安装。

```bash
git clone https://github.com/zhaohuiyang94-tech/data_generation.git
cd data_generation/pathcraft
python3 scripts/verify_bundle.py
PYTHONPATH=src python3 -m semantic_guided_kbqa.cli --help
```

检查覆盖全部三个 pipeline 配置、训练契约、数据 JSON、检索索引和本体文件，不需要历史 trace，也不会调用模型。

## 外部服务

运行前修改 `configs/pipeline*.json` 中的服务地址。默认本地端口如下：

| 服务 | 地址 |
|---|---|
| Semantic | `http://127.0.0.1:18002/v1` |
| Compose | `http://127.0.0.1:18003/v1` |
| Operator | `http://127.0.0.1:18004/v1` |
| Selector | `http://127.0.0.1:18005/v1` |
| BGE embedding | `http://127.0.0.1:8008/embed` |
| Freebase SPARQL | `http://127.0.0.1:3005/sparql` |

使用 GLM 分解复核时，需要在 shell 中设置 `GLM_API_KEY`。`.env.example` 是变量示例，不会自动加载。

若使用附带的 LlamaFactory 启动脚本，先修改 `configs/model_services/*.yaml` 中的模型和适配器路径，并设置 `FACTORY_ROOT`、`FACTORY_CLI` 及各模型对应的 `*_CUDA_DEVICES`。YAML 中保留了原部署参数，模型权重需自行准备。

```bash
bash scripts/start_model_services.sh
bash scripts/check_model_services.sh
```

## 运行 CWQ

启动所有外部服务后：

```bash
export GLM_API_KEY='替换为自己的密钥'
PYTHON_BIN=python3 QUESTION_WORKERS=4 bash scripts/run_full_v42.sh
```

禁用 GLM 分解复核：

```bash
PYTHON_BIN=python3 NO_DECOMPOSITION_REVIEW=1 bash scripts/run_full_v42.sh
```

输出位于 `outputs/`。将 `CWQ_RUN_DIR` 指向已有运行目录可继续未完成任务。

## 运行 WebQSP

下面命令使用仓库附带的 WebQSP 分解输入，以及 CWQ 训练的问答模型；先运行 3 条验证服务配置：

```bash
mkdir -p outputs/webqsp
PYTHONPATH=src python3 -m semantic_guided_kbqa.cli \
  --config configs/pipeline.webqsp_cwq_transfer.json \
  --decompositions data/webqsp/decompose_test_pred.json \
  --no-decomposition-review \
  --limit 3 \
  --output outputs/webqsp/results.json \
  --errors-output outputs/webqsp/errors.json \
  --artifacts-dir outputs/webqsp/artifacts
```

改用 `--limit 0` 运行全量。如果需要重新生成分解并进行 GLM 复核，可配置上述模型服务路径后执行 `scripts/run_webqsp_cwq_end_to_end_serial.sh`。该脚本还需要 Decompose 服务模型配置，并会按阶段启动和停止它管理的服务。

这个发行版本不提供历史严格回放材料，因此不能仅凭仓库文件重放原实验；在线问答需要上述外部服务。
