<div align="center">

# 🐕 Doc Thinker

**文档即思考** — 像小狗一样拿着放大镜读懂每一份文档

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-Yang--Jiashu%2Fdoc--thinker-black?logo=github)](https://github.com/Yang-Jiashu/doc-thinker)

<img src="logo.png" alt="Doc Thinker Logo" width="220" />

**基于 AutoThink 架构** — 多模态文档解析 · 图 RAG 检索

[快速开始](#-快速开始) · [特性](#-特性) · [项目结构](#-项目结构)

</div>

---

## ✨ 特性

- **📄 多模态解析** — MinerU / Docling 解析 PDF，支持文本、图片、表格、公式
- **🕸️ 图 RAG** — AutoThink 图引擎 + 图遍历（BFS 多跳），实体 / 关系检索更准
- **🔀 超图检索** — 超图存储与检索，适合复杂关系与多跳推理

---

## 🚀 快速开始

```bash
git clone https://github.com/Yang-Jiashu/doc-thinker.git
cd doc-thinker

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -U pip && pip install -r requirements.txt && pip install -e .
```

复制 `env.example` 为 `.env`，填入 LLM / Embedding 的 API 地址与 Key。

## 📁 项目结构

| 目录 | 说明 |
|------|------|
| 核心库目录 | **AutoThink 核心**：解析、入库、查询、知识图谱、自动思考、超图 |
| `scripts/` | 入库脚本 |
| `neuro_memory/` | 类脑记忆引擎（扩散激活、巩固、类比检索） |
| `docs/` | 项目说明与文档 |

---

## ⚙️ 配置

在 `.env` 中配置：`WORKING_DIR`、`LLM_BINDING_HOST` / `LLM_BINDING_API_KEY`、`EMBEDDING_*`、`RERANK_*`。完整项见 `env.example`。

---

## 📄 License & Contributing

- **License**：[MIT](LICENSE)
- **Contributing**：[CONTRIBUTING.md](CONTRIBUTING.md)
