# GPT-5.4 数据飞轮

本仓库提供运行代码和 WebQSP 输入训练数据，不包含历史生成结果和 trace。
CWQ 数据和问答系统见 https://github.com/zhaohuiyang94-tech/data_generation/tree/main/pathcraft 。

## 独立运行

需要 Python 3.10+。在仓库根目录执行：

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m gpt54_data_flywheel preflight
# 设置 OPENAI_API_KEY，并将 OPENAI_RESPONSES_URL 指向你实际使用的 Responses 服务。
PYTHONPATH=src python3 -m gpt54_data_flywheel generate --api-mode responses --limit 3
```

`--api-mode responses` 使用仓库内置客户端，不依赖外部 `webqsp_mas` 项目。
下文保留原始方法和兼容模式说明；默认 `kaede` 模式需要另行安装该外部项目。
`.env.example` 仅用于参考，程序不会自动加载它，请在 shell 中导出对应环境变量。

这是一个独立于 `/home/yangzhaohui/kaede_vqg` 的训练数据生成工程。它读取现有
`semantic_path_*.json` 和完整 JSON program 版本的 `compose_*.json`，调用
GPT-5.4 改写样本，经过静态校验、可选外部验证和独立 GPT 复核后，才把样本写入
新的训练集。

默认上游输入是：

```text
data/webqsp/semantic_path_train.json
data/webqsp/compose_train.json
```

上游文件始终只读。所有检查点、拒绝记录和新训练集默认写到本项目的
`data/generated/`。

## 数据流

```text
原 semantic + 原 compose gold
          |
          v
GPT-5.4 重写 semantic
  anchors + semantic_paths + operators
          |
          v
静态校验 operator/路径/变量/关系标签
          |
          v
GPT-5.4 重构 compose JSON program
          |
          v
静态校验 + 可选外部验证器 + GPT-5.4 独立复核
          |
       pass/corrected
          |
          v
accepted.jsonl -> semantic_path_train.json + compose_train.json
```

operator 在 semantic 阶段生成，包括 `AND`、`COUNT`、`ARGMAX`、`ARGMIN`、`TC`
和四种比较约束。`AND` 会显式写入 semantic 标签，输入是参与交集的路径 ID；compose
再把它降解为共享全局变量，因此完整 program 的可执行 operator 列表中不重复保存
`AND`。其他 operator 只能把 `P0.V0` 之类的路径变量映射为 `V0` 之类的全局变量，
类型、值、value type 和属性关系必须逐项保持一致。

## Operator 契约

| 形式 | JSON `type` | 作用域 |
| --- | --- | --- |
| `AND(E1,E2)` | `AND` | 求实体集合交集；semantic 中使用路径 ID，compose 中降解为共享变量 |
| `COUNT(E)` | `COUNT` | 返回集合 `E` 的大小 |
| `ARGMAX(E,r)` | `ARGMAX` | 按投影 `{JOIN(r,e) | e in E}` 的最大 literal 选择 `E` |
| `ARGMIN(E,r)` | `ARGMIN` | 按投影 `{JOIN(r,e) | e in E}` 的最小 literal 选择 `E` |
| `GT(E,i)` | `GREATER_THAN` | 保留大于 literal `i` 的元素 |
| `GE(E,i)` | `GREATER_OR_EQUAL` | 保留大于等于 literal `i` 的元素 |
| `LT(E,i)` | `LESS_THAN` | 保留小于 literal `i` 的元素 |
| `LE(E,i)` | `LESS_OR_EQUAL` | 保留小于等于 literal `i` 的元素 |
| `TC(E,i)` | `TC` | 按时间 literal `i` 约束集合 `E` |

所有 operator 都固定包含 `type`、`inputs`、`input_var`、
`attribute_relation_label`、`attribute_relation_labels`、`value`、`value_type` 七个字段；
未使用字段必须是空字符串或空数组。无 operator 时必须输出 `"operators": []`。

GPT semantic/compose 生成 prompt 已改为 few-shot，包含上述九种 operator 和一个无
operator 示例。比较操作在形式定义中可写 `GT/GE/LT/LE`，JSON 标签始终使用表中的
长名称，以兼容当前 KaeDe runtime。

## 先做离线检查

无需 API key：

```bash
cd data_generation
PYTHONPATH=src python3 -m gpt54_data_flywheel preflight
```

它会检查 semantic/compose 是否能按 question 一一配对，并分别统计 semantic
operator 和 compose 可执行 operator。

## 小批量试跑

设置 key 后先跑 3 条，不建议第一次就处理完整训练集：

```bash
cd data_generation
export OPENAI_API_KEY='your-key'

PYTHONPATH=src python3 -m gpt54_data_flywheel generate \
  --limit 3 \
  --output data/pilot
```

默认沿用 `kaede_vqg` 的 GPT 调用配置：模型为 `gpt-5.4`，Responses endpoint 为
`https://ai.gs88.shop/v1/responses`。也可以通过 `WEBQSP_MODEL`、
`WEBQSP_RESPONSES_URL`（兼容 `OPENAI_MODEL`、`OPENAI_RESPONSES_URL`）或命令行
`--model`、`--responses-url` 显式覆盖。请求体同样使用 system/user input 和严格
JSON Schema；默认不发送 `max_output_tokens`，网关不接受 reasoning 字段时会自动
重试无 reasoning 的请求，与 `kaede_vqg` 的 GPT 客户端保持一致。

CLI 默认使用 `--api-mode kaede`，直接导入
`/home/yangzhaohui/webqsp_mas/src/webqsp_mas/llm_client.py`，而不是在本项目中仿写
KAEDE 的 HTTP 调用。可用 `WEBQSP_MAS_SRC` 覆盖该源码目录。`auto` pipeline profile
在 KAEDE 模式下使用 `lean`：仍覆盖无 operator 和全部九种 operator，但压缩重复描述，
降低代理在完整 JSON Schema 生成期间触发 Cloudflare `524` 的概率。需要原始完整版
few-shot 时可显式传 `--pipeline-profile standard`。

若代理的普通文本请求可用、但严格 `text.format.json_schema` 持续返回 `5xx`，可使用
`--json-mode prompt`。此模式把完整 schema 放入文本 prompt，输出仍会经过相同的
静态校验、operator 一致性检查和独立 GPT 复核。为兼容当前代理，它使用已验证可用的
普通文本输出请求：顶层 `instructions` 放 few-shot 和输出结构，字符串 `input` 只放
样本 JSON，避免与代理默认注入的 instructions 叠加。网关请求使用覆盖全部 operator
但去除重复叙述的紧凑 few-shot；训练数据中仍保留完整版 few-shot。成功记录的 trace
会标记实际模式。

若 Responses 路由只能处理极简请求而复杂请求持续返回 `502/503`，可增加
`--api-mode chat --json-mode prompt`。它会从 Responses URL 自动得到同一代理的
`/v1/chat/completions`，以单条 user message 请求紧凑 prompt，后续校验流程不变。
Chat 模式默认发送与 KAEDE 客户端一致的 `max_tokens=2048`。

`--api-mode chat` 默认启用 `--pipeline-profile micro`，用于该 endpoint 固定注入约 4K
tokens 上下文的情况：semantic/compose 只发送必要源图与规则，输出上限为 384 tokens；
verifier 只返回 `pass/reject + issues` 且上限为 96 tokens。完整候选不再由 verifier
重复输出，但仍须通过所有本地严格校验和可选外部验证器。`lean` 仍可显式选择。

只处理 semantic 阶段含 operator 的样本（包括 `AND`）：

```bash
PYTHONPATH=src python3 -m gpt54_data_flywheel generate \
  --operator-only \
  --limit 20 \
  --output data/operator-pilot
```

确认 pilot 后，去掉 `--limit` 即可处理选中的全部样本。每个样本默认最多尝试 5 次；
始终未通过的样本只写 `rejected.jsonl`，不会进入训练 JSON。再次运行相同命令会根据
`accepted.jsonl` 自动跳过已经成功的样本，并重试之前失败的样本。

生成命令默认显示进度条，包括整体完成比例、当前样本、semantic/compose/verifier
阶段、内容重试轮次、accepted/rejected/skipped 计数、耗时和 ETA。某个模型请求较慢时，
进度条会停留在对应的 `waiting for model` 阶段，这表示正在等待 endpoint 返回，并非本地
程序卡死。后台运行或需要纯 JSON 日志时可传 `--no-progress`。

## 输出文件

```text
data/generated/
├── accepted.jsonl             # 可恢复、去重用的成功检查点
├── rejected.jsonl             # 每轮最终失败的候选和原因
├── semantic_path_train.json   # 新 semantic SFT 数据
├── compose_train.json         # 新 compose SFT 数据
├── dataset_info.json          # LLaMA-Factory 数据定义
└── manifest.json              # 模型、输入、计数和输出清单
```

如遇中断，可从检查点重新物化 JSON 数组：

```bash
PYTHONPATH=src python3 -m gpt54_data_flywheel materialize --output data/generated
```

## 验证策略

内置静态校验会拒绝以下情况：路径或变量断裂、非法/重复 ID、点分 Freebase ID
泄漏、非规范 relation label、断开的 compose 图、错误 answer variable、operator
遗漏/新增、operator 值变化，以及 semantic 路径变量没有正确映射到 compose 全局变量。

如果已有 Freebase grounding、SPARQL 编译或执行验证脚本，可通过
`--external-validator-command` 接入。命令从 stdin 读取一个 JSON 对象，并在 stdout
返回：

```json
{"valid": true, "errors": []}
```

拒绝时返回 `{"valid": false, "errors": ["具体原因"]}`。命令采用参数数组执行，
不会经过 shell。例如：

```bash
PYTHONPATH=src python3 -m gpt54_data_flywheel generate \
  --limit 10 \
  --external-validator-command "python3 /absolute/path/to/validator.py"
```

没有外部验证器时，“验证成功”表示静态结构校验和独立 GPT 复核均通过，不等同于已在
真实 Freebase 上执行成功。要把执行正确性作为硬门槛，应接入上述外部验证器。

## 数据检查

```bash
PYTHONPATH=src python3 -m gpt54_data_flywheel preflight
```

本次精简发行只包含运行代码与数据。发布前已使用原项目的 20 项测试验证运行代码。

工程默认直接复用 `webqsp_mas.llm_client`，需要其已有的 `requests` 依赖，但不需要
OpenAI SDK，也不会记录 API key。`accepted.jsonl` 记录 response id、token usage 和
每轮校验结果，便于审计数据来源。

## PathCraft 问答系统

`pathcraft/` 包含 CWQ 和 WebQSP 问答运行代码、数据、配置及检索索引。环境配置与运行方法见 [PathCraft README](pathcraft/README.md)。
