# LexiScope

Team-16

# LexiScope · 文脉 · 智能文本洞察者

**一键上传，三秒读懂文章核心。**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)](https://flask.palletsprojects.com/)
[![transformers](https://img.shields.io/badge/transformers-4.0%2B-yellow)](https://huggingface.co/docs/transformers/index)

---

## 🧭 项目是什么

**LexiScope** 是一个轻量级的自动化文本分析工具，专为**学生、内容创作者和阅读爱好者**设计。

你只需上传一份 `.txt` 或 `.docx` 文件，或直接粘贴文本内容，它就能在几秒内自动完成三项核心分析：

- 📊 **高频词统计**（Top 20，过滤停用词）
- 🏷️ **TF‑IDF 关键词提取**（Top 10，权重排序）
- ❤️ **情感倾向判断**（基于深度学习的多标签模型，输出综合得分、倾向标签及情感点评）

所有结果以**三栏卡片**形式清晰展示，无需手动阅读全文，即可快速把握文章要点。

---

## 💡 解决什么问题

在日常学习、阅读或内容创作中，我们常常需要：

- 快速了解一篇文章或新闻稿**主要谈论什么**；
- 分析自己写的文章**风格是否偏正面**，关键词是否突出；
- 在长篇材料中**快速定位核心概念**，节省阅读时间。

LexiScope 通过自动化文本分析技术，**降低分析门槛**，让信息获取更加高效、客观。

---

## ✅ 当前版本（V3.0）能完成什么

| 功能 | 说明 |
|------|------|
| 📂 上传 `.txt` / `.docx` 文件 | 支持 UTF‑8 编码的文本文件及 Word 文档，大小 ≤5MB |
| 📝 粘贴纯文本 | 直接在文本框输入内容 |
| 🇨🇳 中文文本分析 | 基于 `jieba` 分词，过滤常见停用词 |
| 📊 词频统计 | 返回出现次数最高的 **20 个词** |
| 🏷️ TF‑IDF 关键词 | 手动实现算法，返回权重最高的 **10 个词**（保留 3 位小数） |
| ❤️ 情感分析 | 基于深度学习多标签模型（GoEmotions 中文版），输出综合得分（-100~100）、倾向标签（积极/中性/消极）及情感点评 |
| 🖥️ 卡片式结果展示 | 三栏横向布局，响应式设计，适配桌面与移动端 |
| ⚠️ 友好错误提示 | 空文件、格式错误、编码问题等均有明确反馈 |

### ❌ 本次不包含（后续迭代）

- 支持 `.pdf` 格式
- 批量文件分析
- 词云可视化
- 文章自动摘要
- 接入大模型 API
- 结果导出为 PDF/Word

---

## 🛠️ 技术架构

- **后端**：Flask（Web 服务）
- **分词**：jieba（用于词频与关键词提取）
- **情感分析**：HuggingFace Transformers（`liudev/roberta-multilabel-28-3-classes` 模型）
- **算法实现**：词频统计、TF‑IDF 使用 Python 内置库（re, math, collections）
- **前端**：原生 HTML + CSS + JavaScript（单页应用，无外部依赖）
- **部署**：单文件 `app.py`，开箱即用

### 模块划分

| 模块 | 核心函数 | 说明 |
|------|----------|------|
| Web 服务 | `create_app()` / `analyze()` | 路由、文件接收、调用各模块、返回 JSON |
| 词频统计 | `get_word_freq(text)` | 中文分词、停用词过滤、频次排序，返回 Top 20 |
| 关键词提取 | `extract_keywords(text)` | 手写 TF‑IDF 算法（句子级），返回 Top 10 权重词 |
| 情感分析 | `get_sentiment(text)` | 加载预训练模型，计算多标签概率，输出综合得分与标签 |

---

## 🚀 如何运行与使用

### 1. 环境要求

- Python 3.8 或更高版本
- 建议使用虚拟环境（如 `venv`）
- 磁盘空间：约 1.5 GB（用于下载模型文件，仅首次）
- 网络：首次启动需下载模型（约 1.5 GB），请确保网络畅通

### 2. 安装依赖

```bash
pip install flask jieba transformers torch
