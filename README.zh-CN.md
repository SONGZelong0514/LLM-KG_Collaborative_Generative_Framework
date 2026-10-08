# 基于大语言模型与知识图谱协同的飞机制造系统生成式设计框架

**演示视频：** [https://youtu.be/icpT_mcnjMk](https://youtu.be/icpT_mcnjMk)  
**对应论文：** [https://doi.org/10.1016/j.aei.2026.105349](https://doi.org/10.1016/j.aei.2026.105349)

> **代码公开说明：** MBSE 与仿真转换组件 `csv2GOPPRRE.py`、`GOPPRRE2sim.py` 和 `GOPPRRE.owl` 包含私有实现，因此不随公开仓库提供。缺少这些文件只会影响 MBSE OWL 模型和 MATLAB SimEvents 仿真模型的生成；知识图谱问答、装配方案生成、方案验证和方案重生成功能均可正常使用。程序仅在请求相应功能时加载这些组件，组件不存在时会返回明确的错误信息。

[English README](README.md)

## 项目简介

本项目是一个融合大语言模型、Neo4j 知识图谱、基于模型的系统工程（MBSE）和仿真模型生成的飞机机身装配系统智能设计应用。系统通过统一的 Gradio 界面提供领域知识检索、装配方案自动生成、工程约束验证、基于反馈的方案重生成，以及在私有转换组件可用时的 MBSE 与仿真模型生成。

## 主要功能

- 知识图谱自然语言问答，包括 Cypher 生成、执行、失败重试和交互式图谱可视化。
- 基于工艺、操作、资源、资源需求和前驱关系知识的机身装配方案生成。
- 生成方案和重生成方案的版本化 CSV 输出。
- 对操作持续时间、资源分配、操作成本和资源峰值占用进行确定性检查。
- 基于大语言模型验证用户定义的工程约束。
- 融合人工反馈进行方案重生成。
- 按需加载私有 MBSE 与仿真转换组件。
- 运行时间监控和可选的 NVIDIA GPU 显存监控。

## 项目结构

```text
.
├── src/aircraft_assembly_design/
│   ├── assets/                 # 静态资源
│   ├── ui/                     # Gradio 界面、样式和回调
│   ├── verification/           # 确定性检查和 LLM 验证
│   ├── app.py                  # 应用入口
│   ├── clients.py              # OpenAI 兼容接口与 Neo4j 客户端
│   ├── config.py               # 环境配置与文件路径
│   ├── graph_visualization.py  # PyVis 图谱可视化
│   ├── monitoring.py           # 运行时间与 GPU 监控
│   ├── plans.py                # 方案解析、保存与版本管理
│   ├── private_plugins.py      # 私有组件按需加载适配层
│   ├── prompts.py              # 大语言模型提示词
│   ├── qa.py                   # 请求路由、知识问答和方案生成
│   ├── regeneration.py         # 基于反馈的方案重生成
│   └── streaming.py            # 流式响应工具
├── scripts/
│   └── ontology_import.cypher  # 本体导入与知识图谱整理脚本
├── pyproject.toml
└── requirements.txt
```

程序会按需创建 `plans/`、`constraints/`、`MBSE/`、`Simulation/`、`Verification/` 和 `static/`。这些运行目录已排除在版本控制之外。

## 环境要求

- Python 3.10–3.12
- Neo4j Community 5.26.1
- APOC 5.26.1
- Neosemantics 5.20.0
- 能够访问 `gpt-5.6-sol` 或其他已配置模型的 OpenAI 兼容接口
- 仅在显示 GPU 显存时需要 NVIDIA 驱动和 `nvidia-smi`
- 仅在执行生成的仿真模型时需要 MATLAB 与 SimEvents

## 安装

创建并激活 Python 环境：

```powershell
conda create -n aircraft-design python=3.10 -y
conda activate aircraft-design
```

安装全部依赖和应用：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

创建本地配置文件：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中配置必要的服务连接：

```dotenv
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=OpenAI兼容接口地址
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=你的Neo4j密码
AIRCRAFT_DESIGN_MODEL=gpt-5.6-sol
```

系统默认使用 `gpt-5.6-sol`，以获得更高的性能。如果 OpenAI 兼容接口需要使用其他模型，可以修改 `AIRCRAFT_DESIGN_MODEL`。应用监听地址和端口、GPU 数量、监控间隔、运行目录以及私有组件目录，可以通过 `.env.example` 中的其他 `AIRCRAFT_DESIGN_*` 变量配置。`.env` 已排除在版本控制之外，不应提交真实密钥。

## Neo4j 配置与本体导入

完整导入流程位于 [`scripts/ontology_import.cypher`](scripts/ontology_import.cypher)，内容依据 `docs/复现流程_AEI.docx` 中的复现步骤整理。

### 1. 安装 APOC 与 Neosemantics

1. 下载 APOC 5.26.1 和 Neosemantics 5.20.0。
2. 将两个 JAR 文件复制到 Neo4j 的 `plugins` 目录。
3. 在 `neo4j.conf` 中加入或更新以下内容：

```properties
dbms.unmanaged_extension_classes=n10s.endpoint=/rdf
dbms.security.procedures.unrestricted=apoc.*,n10s.*
dbms.security.procedures.allowlist=apoc.*,n10s.*
dbms.security.allow_csv_import_from_file_urls=true
```

4. 重启 Neo4j。Windows 环境可以使用 `neo4j.bat console` 启动。
5. 检查 APOC 是否可用：

```cypher
RETURN apoc.version();
```

### 2. 导出并导入本体

1. 在 Protégé 中将飞机装配本体导出为 Turtle（`.ttl`）格式。
2. 打开 [`scripts/ontology_import.cypher`](scripts/ontology_import.cypher)。
3. 将以下占位路径替换为导出本体的本地文件 URL：

```text
file:///ABSOLUTE/PATH/aircraft_assembly_process_ontology.ttl
```

Windows 路径示例：`file:///E:/ontology/aircraft_assembly_process_ontology.ttl`。

4. 在 Neo4j Browser 中按章节执行脚本。该脚本将：

   - 初始化 Neosemantics 并创建 URI 唯一约束；
   - 导入 Turtle 本体；
   - 规范导入后的节点名称和本体关系；
   - 为 20 个操作设置持续时间和 Manual/Auto 类型；
   - 为 10 个资源设置小时成本、日历和容量；
   - 重建操作与资源之间的需求关系；
   - 创建应用使用的 `Process`、`Operation` 和 `Resource` 标签；
   - 将本体约束转换为程序使用的关系类型；
   - 规范最终属性和资源名称。

如需全新导入，请仅在确认删除现有数据后执行：

```cypher
MATCH (n) DETACH DELETE n;
```

如需在导入完成后删除 Neosemantics URI 约束，请执行：

```cypher
DROP CONSTRAINT n10s_unique_uri;
```

### 3. 检查导入结果

规范处理前，Neosemantics 可能生成 `Resource`、`_GraphConfig`、`Class`、`Property`、`Relationship` 等标签，以及 `DOMAIN`、`RANGE`、`SCO`、`SCO_RESTRICTION`、`SPO` 等关系类型。导入脚本会将这些本体层元素转换为应用需要的知识图谱结构。

## 启动应用

在项目根目录运行：

```powershell
python -m aircraft_assembly_design
```

也可以使用安装后的命令：

```powershell
aircraft-assembly-design
```

如未修改监听地址或端口，请访问 [http://localhost:7860](http://localhost:7860)。

## 使用流程

1. 查询工艺、操作、资源、资源需求或前驱关系知识。
2. 选择 **Plan** 或输入自定义设计要求，生成装配方案。
3. 选择 **Verification**，执行确定性检查并生成 LLM 验证报告。
4. 输入人工反馈并选择 **Regeneration**，生成新的方案版本。
5. 如果私有组件存在于所配置的私有组件目录中，可以选择 **MBSE** 和 **Simulation** 生成相应的 OWL 与 MATLAB 模型。

## 引用

如果本项目用于学术研究，请通过 DOI 引用对应论文：[10.1016/j.aei.2026.105349](https://doi.org/10.1016/j.aei.2026.105349)。
