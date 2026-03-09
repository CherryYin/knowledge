# RAG 知识库文档摄入工具 — 技术实现方案（v2）

## 一、系统总览

### 1.1 目标

构建一个通用的文档解析与 Embedding 工具，作为 RAG 知识库的文档摄入层。支持多种文档格式的深度解析（文本、表格、公式、图片），并使用 Qwen 多模态 Embedding 模型生成统一的向量表示，最终存入向量数据库供检索使用。

### 1.2 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          文档摄入 Pipeline                               │
│                                                                          │
│  ┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌────────────────┐   │
│  │ 文档上传  │──▶│  格式路由器   │──▶│  统一解析  │──▶│  LLM TOC 抽取  │   │
│  │ & 预处理  │   │ (Format      │   │ (Multi-   │   │  + 结构化树    │   │
│  │          │   │  Router)     │   │  Parser)  │   │               │   │
│  └──────────┘   └──────────────┘   └───────────┘   └──────┬───────┘   │
│                                                           │            │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐     │            │
│  │ 向量数据库│◀──│  Embedding   │◀──│ TOC-Driven    │◀────┘            │
│  │ (Milvus/ │   │  Service     │   │ Chunking      │                  │
│  │  Qdrant) │   │              │   │ + 上下文注入   │                  │
│  └──────────┘   └──────────────┘   └───────────────┘                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.3 技术栈选型

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| 开发语言 | Python 3.10+ | 生态丰富，AI/ML 支持好 |
| Web 框架 | FastAPI | 异步高性能，自带 OpenAPI 文档 |
| 任务队列 | Celery + Redis | 异步处理大文件，支持重试 |
| 文档解析 | MinerU / Marker / Unstructured | 深度文档解析（见 2.2） |
| OCR 引擎 | PaddleOCR / Surya | 中文支持好，精度高 |
| 公式识别 | LaTeX-OCR / Pix2Tex | 公式转 LaTeX |
| 表格识别 | TableTransformer / MinerU 内置 | 表格结构化提取 |
| TOC 抽取 LLM | Qwen2.5-72B-Instruct | 中文理解强，结构化输出好 |
| Embedding 模型 | Qwen2.5-VL-Embedding (多模态) | 文本+图片统一向量空间 |
| 向量数据库 | Milvus / Qdrant / Weaviate | 高性能向量检索 |
| 对象存储 | MinIO / S3 | 存储原始文档和解析产物 |
| 元数据存储 | PostgreSQL | 文档元数据、分块索引 |

---

## 二、文档解析模块（Document Parser）

### 2.1 支持的文档格式

| 格式 | 扩展名 | 解析方案 |
|------|--------|---------|
| PDF | .pdf | MinerU（首选）/ Marker / PyMuPDF |
| Word | .docx / .doc | python-docx + pandoc |
| PPT | .pptx / .ppt | python-pptx |
| Excel | .xlsx / .xls / .csv | openpyxl / pandas |
| Markdown | .md | markdown-it-py |
| HTML | .html | BeautifulSoup4 |
| 纯文本 | .txt | 直接读取 |
| 图片 | .png / .jpg / .tiff | OCR + 多模态 Embedding |
| LaTeX | .tex | 直接解析 + 公式提取 |
| EPUB | .epub | ebooklib |

### 2.2 核心解析引擎选型：MinerU

**推荐使用 MinerU (magic-pdf)** 作为核心 PDF/文档解析引擎，原因如下：

- 开源免费（Apache 2.0），中文社区活跃
- 内置 Layout 分析、表格识别、公式识别、OCR
- 输出结构化 JSON，包含位置信息
- 支持 GPU 加速

```python
# MinerU 安装
# pip install magic-pdf[full]

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe

class MinerUParser:
    """基于 MinerU 的 PDF 深度解析器"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

    def parse_pdf(self, pdf_path: str, output_dir: str) -> dict:
        """
        解析 PDF，输出结构化内容
        返回: {
            "text_blocks": [...],
            "tables": [...],
            "formulas": [...],
            "images": [...],
            "layout": [...]
        }
        """
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)

        writer = FileBasedDataWriter(output_dir)

        pipe = UNIPipe(pdf_bytes, jso_useful_key={}, image_writer=writer)
        pipe.pipe_classify()   # 文档分类
        pipe.pipe_analyze()    # 版面分析
        pipe.pipe_parse()      # 内容解析

        content_list = pipe.pipe_mk_uni_format(output_dir)
        md_content = pipe.pipe_mk_markdown(output_dir)

        return self._structure_output(content_list, md_content, output_dir)

    def _structure_output(self, content_list, md_content, output_dir):
        """将 MinerU 输出转化为统一结构"""
        result = {
            "text_blocks": [],
            "tables": [],
            "formulas": [],
            "images": [],
            "metadata": {}
        }

        for block in content_list:
            block_type = block.get("type")

            if block_type == "text":
                result["text_blocks"].append({
                    "content": block["text"],
                    "page": block.get("page_idx"),
                    "bbox": block.get("bbox"),
                    "is_title": block.get("is_title", False),
                    "level": block.get("level", 0)
                })

            elif block_type == "table":
                result["tables"].append({
                    "html": block.get("html", ""),
                    "markdown": block.get("markdown", ""),
                    "cells": block.get("cells", []),
                    "page": block.get("page_idx"),
                    "bbox": block.get("bbox"),
                    "image_path": block.get("img_path")
                })

            elif block_type == "equation":
                result["formulas"].append({
                    "latex": block.get("latex", ""),
                    "page": block.get("page_idx"),
                    "bbox": block.get("bbox"),
                    "inline": block.get("inline", False),
                    "image_path": block.get("img_path")
                })

            elif block_type == "image":
                result["images"].append({
                    "path": block.get("img_path"),
                    "caption": block.get("caption", ""),
                    "page": block.get("page_idx"),
                    "bbox": block.get("bbox")
                })

        return result
```

### 2.3 多格式统一解析器

```python
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ContentType(Enum):
    TEXT = "text"
    TABLE = "table"
    FORMULA = "formula"
    IMAGE = "image"
    CODE = "code"
    HEADING = "heading"


@dataclass
class ContentBlock:
    """统一内容块"""
    type: ContentType
    content: str                          # 文本内容或文件路径
    page: Optional[int] = None            # 所在页码
    bbox: Optional[list] = None           # 位置坐标 [x0, y0, x1, y1]
    metadata: dict = field(default_factory=dict)  # 额外元信息
    image_path: Optional[str] = None      # 关联的图片路径（表格截图等）
    children: list = field(default_factory=list)   # 子元素


@dataclass
class ParsedDocument:
    """解析后的文档"""
    doc_id: str
    filename: str
    format: str
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: Optional[int] = None
    blocks: list[ContentBlock] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict = field(default_factory=dict)


class BaseParser(ABC):
    """解析器基类"""

    @abstractmethod
    def parse(self, file_path: str, output_dir: str) -> ParsedDocument:
        pass

    @abstractmethod
    def supported_formats(self) -> list[str]:
        pass


class PDFParser(BaseParser):
    """PDF 解析器 - 基于 MinerU"""

    def __init__(self):
        self.mineru = MinerUParser(use_gpu=True)

    def supported_formats(self) -> list[str]:
        return [".pdf"]

    def parse(self, file_path: str, output_dir: str) -> ParsedDocument:
        result = self.mineru.parse_pdf(file_path, output_dir)
        blocks = []

        for tb in result["text_blocks"]:
            block_type = ContentType.HEADING if tb["is_title"] else ContentType.TEXT
            blocks.append(ContentBlock(
                type=block_type,
                content=tb["content"],
                page=tb["page"],
                bbox=tb["bbox"],
                metadata={"level": tb.get("level", 0)}
            ))

        for table in result["tables"]:
            blocks.append(ContentBlock(
                type=ContentType.TABLE,
                content=table["markdown"],
                page=table["page"],
                bbox=table["bbox"],
                image_path=table.get("image_path"),
                metadata={"html": table["html"]}
            ))

        for formula in result["formulas"]:
            blocks.append(ContentBlock(
                type=ContentType.FORMULA,
                content=formula["latex"],
                page=formula["page"],
                bbox=formula["bbox"],
                image_path=formula.get("image_path"),
                metadata={"inline": formula["inline"]}
            ))

        for img in result["images"]:
            blocks.append(ContentBlock(
                type=ContentType.IMAGE,
                content=img["caption"],
                page=img["page"],
                bbox=img["bbox"],
                image_path=img["path"]
            ))

        return ParsedDocument(
            doc_id=generate_doc_id(file_path),
            filename=Path(file_path).name,
            format="pdf",
            blocks=sorted(blocks, key=lambda b: (b.page or 0, (b.bbox or [0])[1])),
            raw_text="\n".join(b.content for b in blocks if b.type in [ContentType.TEXT, ContentType.HEADING])
        )


class DocxParser(BaseParser):
    """Word 文档解析器"""

    def supported_formats(self) -> list[str]:
        return [".docx", ".doc"]

    def parse(self, file_path: str, output_dir: str) -> ParsedDocument:
        from docx import Document as DocxDocument
        from docx.table import Table

        doc = DocxDocument(file_path)
        blocks = []

        for element in doc.element.body:
            tag = element.tag.split("}")[-1]

            if tag == "p":
                para = self._find_paragraph(doc, element)
                if para:
                    style = para.style.name if para.style else ""
                    is_heading = style.startswith("Heading")
                    blocks.append(ContentBlock(
                        type=ContentType.HEADING if is_heading else ContentType.TEXT,
                        content=para.text,
                        metadata={"style": style}
                    ))

            elif tag == "tbl":
                table = self._find_table(doc, element)
                if table:
                    md = self._table_to_markdown(table)
                    blocks.append(ContentBlock(
                        type=ContentType.TABLE,
                        content=md,
                        metadata={"rows": len(table.rows), "cols": len(table.columns)}
                    ))

        # 提取图片
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                img_data = rel.target_part.blob
                img_filename = Path(rel.target_ref).name
                img_path = Path(output_dir) / img_filename
                img_path.write_bytes(img_data)
                blocks.append(ContentBlock(
                    type=ContentType.IMAGE,
                    content="",
                    image_path=str(img_path)
                ))

        return ParsedDocument(
            doc_id=generate_doc_id(file_path),
            filename=Path(file_path).name,
            format="docx",
            title=doc.core_properties.title or "",
            author=doc.core_properties.author or "",
            blocks=blocks,
            raw_text="\n".join(b.content for b in blocks if b.type == ContentType.TEXT)
        )

    def _find_paragraph(self, doc, element):
        for para in doc.paragraphs:
            if para._element is element:
                return para
        return None

    def _find_table(self, doc, element):
        for table in doc.tables:
            if table._element is element:
                return table
        return None

    def _table_to_markdown(self, table) -> str:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
        if len(rows) > 1:
            header_sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
            rows.insert(1, header_sep)
        return "\n".join(rows)


class FormatRouter:
    """格式路由器 - 根据文件扩展名选择解析器"""

    def __init__(self):
        self.parsers: dict[str, BaseParser] = {}

    def register(self, parser: BaseParser):
        for fmt in parser.supported_formats():
            self.parsers[fmt.lower()] = parser

    def route(self, file_path: str) -> BaseParser:
        ext = Path(file_path).suffix.lower()
        if ext == ".doc":
            ext = ".docx"  # 需先转换
        if ext not in self.parsers:
            raise ValueError(f"Unsupported format: {ext}")
        return self.parsers[ext]

    def parse(self, file_path: str, output_dir: str) -> ParsedDocument:
        parser = self.route(file_path)
        return parser.parse(file_path, output_dir)
```

---

## 三、LLM TOC 抽取模块

### 3.1 设计思路

用大模型从文档全文中提取结构化目录（TOC），而非依赖文档样式（Heading1/2/3），核心优势：

| 方面 | 规则提取 | LLM 抽取 |
|------|---------|---------|
| 有明确标题样式的文档 | ✅ 可以 | ✅ 可以 |
| 标题无特殊格式（纯文本） | ❌ 无法识别 | ✅ 语义理解 |
| 扫描版/OCR 文档 | ❌ 样式丢失 | ✅ 从内容推断 |
| 非标准结构（如会议纪要、邮件） | ❌ 无规则可循 | ✅ 理解逻辑段落 |
| 隐式结构（无编号但有逻辑层次） | ❌ 不可能 | ✅ 推断层次 |

### 3.2 TOC 数据模型

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class TOCNodeType(Enum):
    """TOC 节点类型"""
    PART = "part"                  # 部/篇
    CHAPTER = "chapter"            # 章
    SECTION = "section"            # 节
    SUBSECTION = "subsection"      # 小节
    PARAGRAPH = "paragraph"        # 段落主题
    APPENDIX = "appendix"          # 附录
    FRONT_MATTER = "front_matter"  # 前言/摘要
    BACK_MATTER = "back_matter"    # 参考文献/索引


@dataclass
class TOCNode:
    """TOC 树节点"""
    id: str                                   # 节点唯一 ID (如 "1", "1.1", "1.1.2")
    title: str                                # 标题文本
    level: int                                # 层级深度 (0=根, 1=一级标题, ...)
    node_type: TOCNodeType = TOCNodeType.SECTION
    start_marker: str = ""                    # 在原文中的起始定位文本
    page: Optional[int] = None                # 起始页码
    children: list[TOCNode] = field(default_factory=list)
    parent: Optional[TOCNode] = field(default=None, repr=False)

    # ── 内部定位字段（解析流程使用，不对外暴露）──
    _marker_pos: int = field(default=-1, repr=False)

    @property
    def path(self) -> str:
        """返回从根到当前节点的完整路径，如 '2. 方法 > 2.1 数据采集 > 2.1.1 传感器'"""
        parts = []
        node = self
        while node and node.level > 0:
            parts.append(node.title)
            node = node.parent
        return " > ".join(reversed(parts))

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def flatten(self) -> list[TOCNode]:
        """前序遍历扁平化"""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result


@dataclass
class TOCTree:
    """完整的 TOC 树"""
    doc_id: str
    root: TOCNode                             # 虚拟根节点 (level=0)
    total_nodes: int = 0
    max_depth: int = 0

    @property
    def all_nodes(self) -> list[TOCNode]:
        return self.root.flatten()[1:]        # 排除虚拟根节点

    @property
    def leaf_nodes(self) -> list[TOCNode]:
        return [n for n in self.all_nodes if n.is_leaf]

    def get_nodes_at_level(self, level: int) -> list[TOCNode]:
        return [n for n in self.all_nodes if n.level == level]

    def to_outline_str(self) -> str:
        """输出缩进格式的大纲"""
        lines = []
        for node in self.all_nodes:
            indent = "  " * (node.level - 1)
            lines.append(f"{indent}{node.title}")
        return "\n".join(lines)
```

### 3.3 LLM TOC 抽取器

```python
import json
import re
from typing import Optional
from openai import OpenAI  # 兼容 OpenAI API 格式的客户端


class LLMTOCExtractor:
    """
    使用大模型从文档全文中抽取结构化 TOC

    支持的 LLM 后端:
    - Qwen2.5-72B / 32B (推荐, 中文效果最好)
    - GPT-4o / Claude (英文文档)
    - 本地 vLLM / Ollama

    长文档策略: 分段送入 LLM → 各段独立抽取局部 TOC → 合并校验
    """

    # ── 核心 Prompt ───────────────────────────────────────────

    SYSTEM_PROMPT = """你是一个专业的文档结构分析助手。你的任务是从给定的文档文本中提取目录结构（Table of Contents）。

## 任务要求

1. **识别所有层级的标题和章节**，包括：
   - 明确的编号标题（如 "1. 引言"、"2.1 方法"）
   - 无编号但有逻辑层次的标题（如 "背景"、"实验设置"）
   - 隐式的段落主题（当文档没有明确标题时，提取关键段落的主题作为标题）

2. **准确判断层级关系**：
   - level 1: 最高级标题（章/部分）
   - level 2: 次级标题（节）
   - level 3: 三级标题（小节）
   - level 4+: 更深层级

3. **提供定位锚点**：
   - `start_marker` 必须是文档原文中**完全存在**的一段文字（通常是标题本身）
   - 这个锚点用于在原文中精确定位该章节的起始位置
   - 如果标题文字在原文中有细微差异，以原文为准

4. **判断节点类型**：
   - front_matter: 摘要、前言、目录本身
   - chapter: 主要章节
   - section: 章节下的节
   - subsection: 更细的小节
   - appendix: 附录
   - back_matter: 参考文献、致谢

## 输出格式

严格返回 JSON 数组，每个元素结构如下：
```json
[
  {
    "id": "1",
    "title": "标题文本",
    "level": 1,
    "node_type": "chapter",
    "start_marker": "原文中用于定位的文本片段",
    "page": null
  }
]
```

## 注意事项
- 只返回 JSON，不要有任何其他内容
- 如果文档本身含有目录页，优先参考但仍需根据正文验证
- id 使用层级编号（如 1, 1.1, 1.1.1）
- start_marker 必须能在原文中**精确匹配**到"""

    MERGE_PROMPT = """你是文档结构分析助手。以下是从一份长文档的不同片段中分别提取的局部 TOC。
请合并为一个完整的、无重复的全局 TOC。

## 合并规则
1. 去除重复节点（跨片段可能重复提取了相同章节）
2. 修正层级关系，确保逻辑一致
3. 重新编排 id 使其连续（1, 1.1, 1.2, 2, 2.1, ...）
4. 保留所有有效的 start_marker

## 输入
{partial_tocs}

## 输出
严格返回合并后的 JSON 数组，格式与输入相同。只返回 JSON。"""

    # ── 初始化 ────────────────────────────────────────────────

    def __init__(
        self,
        model: str = "qwen2.5-72b-instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        max_segment_tokens: int = 6000,
        temperature: float = 0.1,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_segment_tokens = max_segment_tokens
        self.temperature = temperature

    # ── 主入口 ────────────────────────────────────────────────

    def extract_toc(self, full_text: str, doc_id: str = "") -> TOCTree:
        """
        从完整文档文本中抽取 TOC

        流程:
        1. 检测文档是否自带 TOC 页 → 有则优先解析
        2. 文档分段 → 每段调用 LLM 抽取局部 TOC
        3. 合并局部 TOC → LLM 全局校验
        4. 校验 start_marker → 构建 TOCTree
        """

        # Step 1: 检测已有 TOC 页
        existing_toc = self._detect_existing_toc(full_text)

        if existing_toc:
            toc_nodes = self._parse_existing_toc(existing_toc, full_text)
        else:
            segments = self._split_into_segments(full_text)

            if len(segments) == 1:
                toc_nodes = self._extract_from_segment(segments[0])
            else:
                partial_tocs = []
                for i, segment in enumerate(segments):
                    partial = self._extract_from_segment(
                        segment,
                        context_hint=f"这是文档的第 {i+1}/{len(segments)} 部分"
                    )
                    partial_tocs.append(partial)
                toc_nodes = self._merge_partial_tocs(partial_tocs)

        # Step 2: 校验 start_marker
        toc_nodes = self._validate_markers(toc_nodes, full_text)

        # Step 3: 构建树
        tree = self._build_tree(toc_nodes, doc_id)
        return tree

    # ── 检测已有目录 ──────────────────────────────────────────

    def _detect_existing_toc(self, text: str) -> Optional[str]:
        """检测文档中是否包含目录页"""
        toc_patterns = [
            r"(?i)(table\s+of\s+contents|目\s*录|目\s*次)\s*\n",
            r"(?i)(contents)\s*\n.*?(?=\n\s*(?:chapter|第|1[\.\s]))",
        ]
        for pattern in toc_patterns:
            match = re.search(pattern, text)
            if match:
                start = match.start()
                end = min(start + 3000, len(text))
                candidate = text[start:end]
                lines = [l.strip() for l in candidate.split("\n") if l.strip()]
                numbered_lines = sum(1 for l in lines if re.match(r"^[\d\.\s]", l))
                if numbered_lines >= 3:
                    return candidate
        return None

    def _parse_existing_toc(self, toc_text: str, full_text: str) -> list[dict]:
        """用 LLM 将已有目录文本结构化"""
        prompt = f"""以下是从文档中提取的目录页内容：

---
{toc_text}
---

以下是文档正文的前 2000 字（用于验证）：

---
{full_text[:2000]}
---

请根据目录页内容，结合正文验证，输出结构化的 TOC JSON。
start_marker 必须能在正文中精确定位。"""

        return self._call_llm(prompt)

    # ── 分段 & 提取 ──────────────────────────────────────────

    def _split_into_segments(self, text: str) -> list[str]:
        """按 token 限制分段，在段落边界切分"""
        max_chars = self.max_segment_tokens * 2  # 粗略换算

        if len(text) <= max_chars:
            return [text]

        segments = []
        paragraphs = text.split("\n")
        current_segment = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > max_chars and current_segment:
                segments.append("\n".join(current_segment))
                # 保留最后 2 段作为上下文重叠
                overlap = current_segment[-2:] if len(current_segment) >= 2 else current_segment[-1:]
                current_segment = list(overlap)
                current_length = sum(len(p) for p in current_segment)

            current_segment.append(para)
            current_length += para_len

        if current_segment:
            segments.append("\n".join(current_segment))

        return segments

    def _extract_from_segment(self, segment_text: str, context_hint: str = "") -> list[dict]:
        """从单个文本段中提取 TOC"""
        prompt = f"""请从以下文档文本中提取目录结构。

{f"提示: {context_hint}" if context_hint else ""}

---文档内容开始---
{segment_text}
---文档内容结束---

请提取其中所有标题和章节结构，输出 JSON。"""

        return self._call_llm(prompt)

    # ── 合并多段 TOC ─────────────────────────────────────────

    def _merge_partial_tocs(self, partial_tocs: list[list[dict]]) -> list[dict]:
        """合并多段提取的局部 TOC"""
        formatted = []
        for i, toc in enumerate(partial_tocs):
            formatted.append(f"## 片段 {i+1}:\n{json.dumps(toc, ensure_ascii=False, indent=2)}")

        merged_input = "\n\n".join(formatted)
        prompt = self.MERGE_PROMPT.format(partial_tocs=merged_input)

        return self._call_llm(prompt, raw_prompt=prompt)

    # ── LLM 调用 ─────────────────────────────────────────────

    def _call_llm(self, user_prompt: str, system_prompt: str = None, raw_prompt: str = None) -> list[dict]:
        """调用 LLM 并解析 JSON 输出"""
        if raw_prompt:
            messages = [{"role": "user", "content": raw_prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=4096,
        )

        content = response.choices[0].message.content.strip()
        json_str = self._extract_json(content)

        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                for key in ["toc", "items", "nodes", "data", "result"]:
                    if key in result and isinstance(result[key], list):
                        return result[key]
                for v in result.values():
                    if isinstance(v, list):
                        return v
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        return self._retry_extraction(content, user_prompt)

    def _extract_json(self, text: str) -> str:
        """从 LLM 输出中提取 JSON"""
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        for pattern in [r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"]:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return text

    def _retry_extraction(self, failed_output: str, original_prompt: str) -> list[dict]:
        """JSON 解析失败时重试"""
        retry_prompt = f"""上一次输出无法解析为有效 JSON，请重新输出。
只输出 JSON 数组，不要有其他文字。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": original_prompt},
                {"role": "assistant", "content": failed_output[:500]},
                {"role": "user", "content": retry_prompt}
            ],
            temperature=0.0,
            max_tokens=4096
        )

        content = response.choices[0].message.content.strip()
        json_str = self._extract_json(content)
        try:
            result = json.loads(json_str)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []

    # ── 校验 & 修复 ──────────────────────────────────────────

    def _validate_markers(self, toc_nodes: list[dict], full_text: str) -> list[dict]:
        """
        校验每个节点的 start_marker 能否在原文中定位

        三级匹配策略:
        1. 精确匹配
        2. 标准化匹配（去空格/标点差异）
        3. 标题关键词模糊搜索兜底
        """
        validated = []

        for node in toc_nodes:
            marker = node.get("start_marker", "")
            title = node.get("title", "")

            # 1. 精确匹配
            if marker and marker in full_text:
                node["_marker_pos"] = full_text.index(marker)
                validated.append(node)
                continue

            # 2. 标准化后匹配
            normalized_marker = re.sub(r"\s+", " ", marker).strip()
            normalized_text = re.sub(r"\s+", " ", full_text)
            if normalized_marker and normalized_marker in normalized_text:
                pos = normalized_text.index(normalized_marker)
                node["start_marker"] = normalized_marker
                node["_marker_pos"] = pos
                validated.append(node)
                continue

            # 3. 用标题模糊搜索
            if title:
                escaped_title = re.escape(title)
                pattern = rf"(?:[\d\.]+\s*)?{escaped_title}"
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    node["start_marker"] = match.group(0)
                    node["_marker_pos"] = match.start()
                    validated.append(node)
                    continue

            # 兜底
            node["_marker_pos"] = -1
            node["_unlocated"] = True
            validated.append(node)

        validated.sort(key=lambda n: n.get("_marker_pos", float("inf")))
        return validated

    # ── 构建树 ───────────────────────────────────────────────

    def _build_tree(self, toc_nodes: list[dict], doc_id: str) -> TOCTree:
        """将扁平 TOC 列表构建为树"""
        root = TOCNode(id="root", title="Document Root", level=0)
        stack = [root]
        max_depth = 0

        for node_dict in toc_nodes:
            level = node_dict.get("level", 1)
            max_depth = max(max_depth, level)

            node = TOCNode(
                id=node_dict.get("id", ""),
                title=node_dict.get("title", ""),
                level=level,
                node_type=TOCNodeType(node_dict.get("node_type", "section")),
                start_marker=node_dict.get("start_marker", ""),
                page=node_dict.get("page"),
            )
            node._marker_pos = node_dict.get("_marker_pos", -1)

            # 找到合适的父节点
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()

            parent = stack[-1]
            node.parent = parent
            parent.children.append(node)
            stack.append(node)

        return TOCTree(
            doc_id=doc_id,
            root=root,
            total_nodes=len(toc_nodes),
            max_depth=max_depth
        )
```

---

## 四、TOC 驱动的分块模块（Chunking Engine）

### 4.1 核心设计思路

```
┌────────────────────────────────────────────────────────────────┐
│  TOC-Driven + min_tokens 自适应合并分块策略                      │
│                                                                │
│  Step 1: 按 TOC 节点 start_marker 将全文切成各节内容             │
│                                                                │
│  Step 2: 自底向上检查每个 TOC 节点的 token 数                    │
│          ├─ >= min_tokens → 独立成 chunk                        │
│          ├─ < min_tokens → 向上合并到父节点                      │
│          └─ 合并后 > max_tokens → 内部滑动窗口二次切分            │
│                                                                │
│  Step 3: 表格/图片/公式 → 独立 chunk，携带 TOC 路径上下文         │
│                                                                │
│  Step 4: 每个 chunk 注入 section_path 上下文前缀用于 Embedding   │
└────────────────────────────────────────────────────────────────┘
```

**与纯 TOC 方案的区别**：不是机械地按 TOC 叶子节点切分，而是以 `min_tokens` 为阈值做自底向上的自适应合并——太短的子节点向上合并到父节点，直到满足最小长度要求。这保证了每个 chunk 既有足够的语义密度，又不破坏 TOC 的逻辑边界。

### 4.2 数据模型

```python
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ChunkType(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    MIXED = "mixed"


@dataclass
class Chunk:
    """文档分块"""
    chunk_id: str
    doc_id: str
    chunk_type: ChunkType
    text_content: str
    image_path: Optional[str] = None
    page: Optional[int] = None
    section_path: str = ""            # 完整层级路径: "2. 方法 > 2.1 数据采集"
    section_title: str = ""           # 当前节标题
    toc_node_id: str = ""             # 关联的 TOC 节点 ID
    depth: int = 0                    # TOC 树深度
    metadata: dict = field(default_factory=dict)
    token_count: int = 0

    @property
    def context_enriched_text(self) -> str:
        """带层级上下文的增强文本（用于 Embedding）"""
        parts = []
        if self.section_path:
            parts.append(f"[章节路径] {self.section_path}")
        if self.section_title:
            parts.append(f"[标题] {self.section_title}")
        parts.append(self.text_content)
        return "\n".join(parts)

    @property
    def embedding_inputs(self) -> dict:
        """返回适用于 Qwen 多模态 Embedding 的输入"""
        inputs = {"text": self.context_enriched_text}
        if self.image_path:
            inputs["image"] = self.image_path
        return inputs
```

### 4.3 TOC 驱动分块器

```python
import re


class TOCDrivenChunker:
    """
    基于 TOC + min_tokens 自适应合并的分块器

    核心算法:
    ┌──────────────────────────────────────────────────────────┐
    │ 1. 按 TOC 节点的 start_marker 将全文切成各节原始内容      │
    │ 2. 自底向上遍历 TOC 树:                                  │
    │    - 叶子节点内容 >= min_tokens → 独立 chunk              │
    │    - 叶子节点内容 < min_tokens → 标记为"待合并"           │
    │ 3. 父节点收集所有"待合并"子节点 + 自身直属文本:            │
    │    - 合并后 >= min_tokens 且 <= max_tokens → 一个 chunk   │
    │    - 合并后 > max_tokens → 按段落边界二次切分              │
    │    - 合并后仍 < min_tokens → 继续向上冒泡给祖父节点       │
    │ 4. 表格/图片 → 始终独立 chunk，不参与合并                 │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        min_chunk_tokens: int = 200,       # 低于此值触发向上合并
        max_chunk_tokens: int = 512,       # 超过此值触发内部二次切分
        overlap_tokens: int = 64,          # 二次切分时的重叠 token 数
    ):
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-VL-Embedding",
                trust_remote_code=True
            )
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ═══════════════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════════════

    def chunk_document(
        self,
        full_text: str,
        toc_tree: TOCTree,
        parsed_doc: ParsedDocument = None,
    ) -> list[Chunk]:
        """
        基于 TOC 树对文档进行分块

        Args:
            full_text:   文档全文
            toc_tree:    LLM 抽取的 TOC 树
            parsed_doc:  文档解析结果（可选，提供表格/图片等结构化信息）

        Returns:
            list[Chunk]: 分块结果
        """
        doc_id = toc_tree.doc_id

        # Step 1: 按 TOC 节点定位点切分全文
        node_contents = self._split_text_by_toc(full_text, toc_tree)

        # Step 2: 计算每个节点的"自身直属文本"（排除子节点内容后的部分）
        node_own_text = self._compute_own_text(node_contents, toc_tree)

        # Step 3: 自底向上合并 + 切分 → 生成 text chunks
        text_chunks = self._merge_bottom_up(toc_tree, node_own_text, doc_id)

        # Step 4: 提取表格/图片 → 生成 media chunks
        media_chunks = []
        if parsed_doc:
            media_chunks = self._extract_media_chunks(
                parsed_doc, toc_tree, node_contents, doc_id
            )

        all_chunks = text_chunks + media_chunks

        # 按文档顺序排序
        all_chunks.sort(key=lambda c: (c.page or 0, c.toc_node_id))

        return all_chunks

    # ═══════════════════════════════════════════════════════════
    #  Step 1: 按 TOC 切分全文
    # ═══════════════════════════════════════════════════════════

    def _split_text_by_toc(
        self, full_text: str, toc_tree: TOCTree
    ) -> dict[str, str]:
        """
        根据 TOC 节点的 start_marker 定位，切分出每个节点对应的全部文本。

        返回: {node_id: "该节点及其所有子节点覆盖的文本"}

        注意: 这里返回的是节点"管辖区域"的完整文本（包含子节点），
        后续 _compute_own_text 会进一步分离出节点自身的直属文本。
        """
        all_nodes = toc_tree.all_nodes
        node_contents: dict[str, str] = {}

        # 收集有有效定位的节点
        positioned = []  # [(pos_in_text, node), ...]
        for node in all_nodes:
            if node._marker_pos >= 0:
                positioned.append((node._marker_pos, node))

        positioned.sort(key=lambda x: x[0])

        # 切分
        for i, (pos, node) in enumerate(positioned):
            content_start = pos + len(node.start_marker)
            content_end = positioned[i + 1][0] if i + 1 < len(positioned) else len(full_text)
            node_contents[node.id] = full_text[content_start:content_end].strip()

        # 处理文档开头（TOC 第一个节点之前的内容）
        if positioned and positioned[0][0] > 100:
            preamble_text = full_text[:positioned[0][0]].strip()
            node_contents["__preamble__"] = preamble_text

        return node_contents

    # ═══════════════════════════════════════════════════════════
    #  Step 2: 计算每个节点的"自身直属文本"
    # ═══════════════════════════════════════════════════════════

    def _compute_own_text(
        self, node_contents: dict[str, str], toc_tree: TOCTree
    ) -> dict[str, str]:
        """
        从每个节点的"管辖区域文本"中去掉子节点覆盖的部分，
        得到该节点自身直属的文本。

        例如:
          "2. 方法" 的管辖文本 = "方法概述...(2.1 的内容)(2.2 的内容)"
          "2. 方法" 的自身文本 = "方法概述..."（只有直属段落）
        """
        own_text: dict[str, str] = {}

        for node in toc_tree.all_nodes:
            full = node_contents.get(node.id, "")
            if not full:
                own_text[node.id] = ""
                continue

            if not node.children:
                # 叶子节点：全部文本都是自身的
                own_text[node.id] = full
            else:
                # 非叶子节点：去掉第一个子节点 start_marker 之后的部分
                # 即取从节点开头到第一个子节点之间的文本作为自身直属文本
                first_child = node.children[0]
                if first_child.start_marker and first_child.start_marker in full:
                    cut_pos = full.index(first_child.start_marker)
                    own_text[node.id] = full[:cut_pos].strip()
                else:
                    # 找不到子节点定位点，保守地认为自身没有直属文本
                    own_text[node.id] = ""

        # 前言
        if "__preamble__" in node_contents:
            own_text["__preamble__"] = node_contents["__preamble__"]

        return own_text

    # ═══════════════════════════════════════════════════════════
    #  Step 3: 自底向上合并 + 切分
    # ═══════════════════════════════════════════════════════════

    def _merge_bottom_up(
        self,
        toc_tree: TOCTree,
        node_own_text: dict[str, str],
        doc_id: str
    ) -> list[Chunk]:
        """
        自底向上遍历 TOC 树，根据 min_tokens 阈值决定合并策略:

        对每个节点，收集以下文本:
          - 该节点的自身直属文本
          - 所有标记为"待合并"（too_short）的子节点文本

        然后判断:
          - 总 token 数 >= min_tokens 且 <= max_tokens → 独立 chunk
          - 总 token 数 > max_tokens → 二次切分
          - 总 token 数 < min_tokens → 标记自己为"待合并"，冒泡给父节点
        """
        chunks: list[Chunk] = []

        # 记录每个节点的合并状态
        # "independent" = 已独立成 chunk
        # "bubble_up"   = 太短，文本需要冒泡给父节点
        node_status: dict[str, str] = {}

        # 每个节点冒泡给父节点的文本
        bubble_text: dict[str, str] = {}

        # 后序遍历（自底向上）
        all_nodes = toc_tree.all_nodes
        for node in reversed(all_nodes):
            # ── 收集该节点的所有文本 ──
            text_parts = []
            title_prefix = node.title

            # 自身直属文本
            own = node_own_text.get(node.id, "").strip()
            if own:
                text_parts.append(own)

            # 收集所有冒泡上来的子节点文本
            for child in node.children:
                if node_status.get(child.id) == "bubble_up":
                    child_bubble = bubble_text.get(child.id, "")
                    if child_bubble:
                        text_parts.append(child_bubble)

            combined_text = "\n\n".join(text_parts).strip()
            token_count = self.count_tokens(combined_text) if combined_text else 0

            # ── 决策 ──
            if token_count == 0:
                node_status[node.id] = "bubble_up"
                bubble_text[node.id] = ""
                continue

            if token_count >= self.min_chunk_tokens and token_count <= self.max_chunk_tokens:
                # 刚好合适 → 独立 chunk
                chunks.append(self._make_chunk(
                    text=combined_text,
                    doc_id=doc_id,
                    node=node,
                    token_count=token_count,
                ))
                node_status[node.id] = "independent"

            elif token_count > self.max_chunk_tokens:
                # 太长 → 二次切分
                sub_chunks = self._split_long_text(combined_text, doc_id, node)
                chunks.extend(sub_chunks)
                node_status[node.id] = "independent"

            else:
                # 太短 (< min_tokens) → 冒泡给父节点
                # 冒泡内容带上标题前缀，保留语义
                bubble_content = f"{title_prefix}\n{combined_text}" if title_prefix else combined_text
                node_status[node.id] = "bubble_up"
                bubble_text[node.id] = bubble_content

        # ── 处理最终仍在冒泡的顶层节点 ──
        # 这些是根的直接子节点中仍然太短的，强制输出为 chunk
        remaining_parts = []
        for child in toc_tree.root.children:
            if node_status.get(child.id) == "bubble_up":
                bt = bubble_text.get(child.id, "")
                if bt:
                    remaining_parts.append(bt)

        if remaining_parts:
            remaining_text = "\n\n".join(remaining_parts).strip()
            if remaining_text:
                token_count = self.count_tokens(remaining_text)
                if token_count > self.max_chunk_tokens:
                    chunks.extend(self._split_long_text(
                        remaining_text, doc_id, toc_tree.root
                    ))
                else:
                    chunks.append(self._make_chunk(
                        text=remaining_text,
                        doc_id=doc_id,
                        node=toc_tree.root,
                        token_count=token_count,
                    ))

        # ── 处理前言 ──
        preamble = node_own_text.get("__preamble__", "").strip()
        if preamble:
            preamble_tokens = self.count_tokens(preamble)
            preamble_node = TOCNode(
                id="0", title="前言/概述", level=1,
                node_type=TOCNodeType.FRONT_MATTER
            )
            preamble_node.parent = toc_tree.root

            if preamble_tokens > self.max_chunk_tokens:
                chunks.extend(self._split_long_text(preamble, doc_id, preamble_node))
            else:
                chunks.append(self._make_chunk(
                    text=preamble, doc_id=doc_id,
                    node=preamble_node, token_count=preamble_tokens
                ))

        return chunks

    # ═══════════════════════════════════════════════════════════
    #  二次切分（超长文本）
    # ═══════════════════════════════════════════════════════════

    def _split_long_text(
        self, text: str, doc_id: str, node: TOCNode
    ) -> list[Chunk]:
        """
        对超过 max_chunk_tokens 的文本做二次切分

        优先在段落边界切分 → 其次在句子边界切分 → 最后按 token 硬切
        """
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_parts: list[str] = []
        current_tokens = 0
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)

            # 单段超长 → 按句子切
            if para_tokens > self.max_chunk_tokens:
                # flush 当前缓冲
                if current_parts:
                    chunks.append(self._make_chunk(
                        text="\n\n".join(current_parts),
                        doc_id=doc_id, node=node,
                        token_count=current_tokens,
                        sub_index=chunk_idx
                    ))
                    chunk_idx += 1
                    current_parts = []
                    current_tokens = 0

                sent_chunks = self._split_by_sentences(para, doc_id, node, chunk_idx)
                chunks.extend(sent_chunks)
                chunk_idx += len(sent_chunks)
                continue

            # 加入后是否超限
            if current_tokens + para_tokens > self.max_chunk_tokens and current_parts:
                chunks.append(self._make_chunk(
                    text="\n\n".join(current_parts),
                    doc_id=doc_id, node=node,
                    token_count=current_tokens,
                    sub_index=chunk_idx
                ))
                chunk_idx += 1

                # 保留最后一段作为上下文重叠
                if self.overlap_tokens > 0 and current_parts:
                    last = current_parts[-1]
                    last_tokens = self.count_tokens(last)
                    current_parts = [last]
                    current_tokens = last_tokens
                else:
                    current_parts = []
                    current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        # flush 剩余
        if current_parts:
            combined = "\n\n".join(current_parts)
            combined_tokens = self.count_tokens(combined)
            # 如果剩余太短且前面已有 chunk，尝试合并到上一个
            if combined_tokens < self.min_chunk_tokens and chunks:
                last_chunk = chunks[-1]
                merged = last_chunk.text_content + "\n\n" + combined
                merged_tokens = self.count_tokens(merged)
                if merged_tokens <= self.max_chunk_tokens:
                    last_chunk.text_content = merged
                    last_chunk.token_count = merged_tokens
                else:
                    chunks.append(self._make_chunk(
                        text=combined, doc_id=doc_id, node=node,
                        token_count=combined_tokens, sub_index=chunk_idx
                    ))
            else:
                chunks.append(self._make_chunk(
                    text=combined, doc_id=doc_id, node=node,
                    token_count=combined_tokens, sub_index=chunk_idx
                ))

        return chunks

    def _split_by_sentences(
        self, text: str, doc_id: str, node: TOCNode, start_idx: int
    ) -> list[Chunk]:
        """按句子边界切分超长段落"""
        sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
        chunks = []
        current = []
        current_tokens = 0
        idx = start_idx

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_tokens = self.count_tokens(sent)

            if current_tokens + sent_tokens > self.max_chunk_tokens and current:
                chunks.append(self._make_chunk(
                    text="".join(current), doc_id=doc_id,
                    node=node, token_count=current_tokens, sub_index=idx
                ))
                idx += 1
                current = []
                current_tokens = 0

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(self._make_chunk(
                text="".join(current), doc_id=doc_id,
                node=node, token_count=current_tokens, sub_index=idx
            ))

        return chunks

    # ═══════════════════════════════════════════════════════════
    #  表格/图片 独立 Chunk
    # ═══════════════════════════════════════════════════════════

    def _extract_media_chunks(
        self,
        parsed_doc: ParsedDocument,
        toc_tree: TOCTree,
        node_contents: dict[str, str],
        doc_id: str
    ) -> list[Chunk]:
        """
        从解析结果中提取表格和图片，生成独立 chunks。
        每个表格/图片 chunk 携带所属 TOC 节点的路径上下文。
        """
        chunks = []

        for block in parsed_doc.blocks:
            if block.type not in (ContentType.TABLE, ContentType.IMAGE):
                continue

            owner_node = self._find_owning_node(block, toc_tree)
            section_path = owner_node.path if owner_node else ""
            section_title = owner_node.title if owner_node else ""

            if block.type == ContentType.TABLE:
                text_desc = f"[表格] {section_title}\n{block.content}" if block.content else f"[表格] {section_title}"
                chunks.append(Chunk(
                    chunk_id=self._gen_id(doc_id, f"table_{block.page}", 0),
                    doc_id=doc_id,
                    chunk_type=ChunkType.TABLE,
                    text_content=text_desc,
                    image_path=block.image_path,
                    page=block.page,
                    section_path=section_path,
                    section_title=section_title,
                    toc_node_id=owner_node.id if owner_node else "",
                    token_count=self.count_tokens(text_desc),
                    metadata=block.metadata
                ))

            elif block.type == ContentType.IMAGE:
                caption = block.content or "文档图片"
                text_desc = f"[图片] {section_title}: {caption}"
                chunks.append(Chunk(
                    chunk_id=self._gen_id(doc_id, f"img_{block.page}", 0),
                    doc_id=doc_id,
                    chunk_type=ChunkType.IMAGE,
                    text_content=text_desc,
                    image_path=block.image_path,
                    page=block.page,
                    section_path=section_path,
                    section_title=section_title,
                    toc_node_id=owner_node.id if owner_node else "",
                    token_count=self.count_tokens(text_desc)
                ))

            elif block.type == ContentType.FORMULA:
                # 公式附带上下文
                context = f"[公式] {section_title}: {block.content}"
                chunks.append(Chunk(
                    chunk_id=self._gen_id(doc_id, f"formula_{block.page}", 0),
                    doc_id=doc_id,
                    chunk_type=ChunkType.FORMULA,
                    text_content=context,
                    image_path=block.image_path,
                    page=block.page,
                    section_path=section_path,
                    section_title=section_title,
                    toc_node_id=owner_node.id if owner_node else "",
                    token_count=self.count_tokens(context)
                ))

        return chunks

    def _find_owning_node(self, block: ContentBlock, toc_tree: TOCTree) -> Optional[TOCNode]:
        """根据位置找到内容块所属的 TOC 节点"""
        if not block.page:
            return None
        best = None
        for node in toc_tree.all_nodes:
            if node.page and node.page <= block.page:
                if best is None or node.page >= best.page:
                    best = node
        return best

    # ═══════════════════════════════════════════════════════════
    #  降级方案
    # ═══════════════════════════════════════════════════════════

    def fallback_chunk(self, full_text: str, doc_id: str) -> list[Chunk]:
        """
        降级: 当 TOC 抽取失败时，使用滑动窗口分块

        按段落边界 + max_chunk_tokens 切分，保留 overlap
        """
        paragraphs = re.split(r"\n\s*\n", full_text)
        chunks = []
        current_parts = []
        current_tokens = 0
        idx = 0

        dummy_node = TOCNode(id="fallback", title="", level=1)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = self.count_tokens(para)

            if current_tokens + para_tokens > self.max_chunk_tokens and current_parts:
                chunks.append(self._make_chunk(
                    text="\n\n".join(current_parts),
                    doc_id=doc_id, node=dummy_node,
                    token_count=current_tokens, sub_index=idx
                ))
                idx += 1
                if self.overlap_tokens > 0 and current_parts:
                    last = current_parts[-1]
                    current_parts = [last]
                    current_tokens = self.count_tokens(last)
                else:
                    current_parts = []
                    current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            chunks.append(self._make_chunk(
                text="\n\n".join(current_parts),
                doc_id=doc_id, node=dummy_node,
                token_count=current_tokens, sub_index=idx
            ))

        return chunks

    # ═══════════════════════════════════════════════════════════
    #  工具方法
    # ═══════════════════════════════════════════════════════════

    def _make_chunk(
        self, text: str, doc_id: str, node: TOCNode,
        token_count: int, sub_index: int = 0
    ) -> Chunk:
        return Chunk(
            chunk_id=self._gen_id(doc_id, node.id, sub_index),
            doc_id=doc_id,
            chunk_type=ChunkType.TEXT,
            text_content=text.strip(),
            section_path=node.path,
            section_title=node.title,
            toc_node_id=node.id,
            depth=node.level,
            token_count=token_count,
            page=node.page
        )

    def _gen_id(self, doc_id: str, node_id: str, idx: int) -> str:
        raw = f"{doc_id}_{node_id}_{idx}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
```

### 4.4 合并算法图解

以下面这棵 TOC 树为例，`min_tokens=200`：

```
文档 TOC 树                              各节点自身直属 token 数
─────────                               ──────────────────────
1. 引言                                   80 tokens (自身段落)
  1.1 背景                                 350 tokens ✅ 独立
  1.2 目标                                 60 tokens
  1.3 范围                                 40 tokens
2. 方法                                    30 tokens (自身段落)
  2.1 数据采集                              500 tokens → 需要二次切分
  2.2 模型设计                              250 tokens ✅ 独立
3. 结论                                    400 tokens ✅ 独立

处理过程（自底向上）:

① 叶子节点先判断:
   1.1 = 350 ≥ 200 → ✅ 独立 chunk
   1.2 = 60 < 200  → ⬆️ 冒泡给 "1. 引言"
   1.3 = 40 < 200  → ⬆️ 冒泡给 "1. 引言"
   2.1 = 500 > 512 → ✂️ 二次切分为 2 个 chunk
   2.2 = 250 ≥ 200 → ✅ 独立 chunk
   3.  = 400 ≥ 200 → ✅ 独立 chunk

② 父节点 "1. 引言" 收集冒泡:
   自身 80 + 1.2 冒泡 60 + 1.3 冒泡 40 = 180 < 200
   → 继续冒泡给根节点（或强制输出，因为已经是顶层）

③ 父节点 "2. 方法":
   自身 30 tokens，所有子节点已独立 → 30 < 200
   → 冒泡给根节点

④ 根节点收集: "1.引言"冒泡 180 + "2.方法"冒泡 30 = 210 ≥ 200
   → ✅ 合并成一个 chunk

最终结果: 6 个 chunks
  - Chunk 1: "1. 引言"(自身) + "1.2 目标" + "1.3 范围" + "2. 方法"(自身)
  - Chunk 2: "1.1 背景"
  - Chunk 3: "2.1 数据采集" (上半部分)
  - Chunk 4: "2.1 数据采集" (下半部分)
  - Chunk 5: "2.2 模型设计"
  - Chunk 6: "3. 结论"
```

---

## 五、Embedding 模块 — Qwen 多模态 Embedding

### 5.1 模型选型

**推荐模型: Qwen2.5-VL-Embedding**

| 特性 | 说明 |
|------|------|
| 类型 | 多模态 Embedding（文本 + 图片） |
| 维度 | 1536（可配置） |
| 支持语言 | 中文、英文、多语言 |
| 图片支持 | 直接编码图片为向量 |
| 最大输入 | 文本 32K tokens / 图片 1344×1344 |
| 部署方式 | vLLM / Transformers / Ollama |

### 5.2 Embedding Service 实现

```python
import torch
import numpy as np
from PIL import Image
from typing import Optional
from pathlib import Path


class QwenMultiModalEmbedding:
    """
    基于 Qwen2.5-VL-Embedding 的多模态 Embedding 服务

    三种输入模式 → 同一向量空间:
    1. 纯文本 → 文本向量
    2. 纯图片 → 图片向量
    3. 文本+图片 → 融合向量
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-Embedding",
        device: str = "cuda",
        max_text_length: int = 8192,
        batch_size: int = 32,
        dimension: int = 1536
    ):
        self.device = device
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.dimension = dimension
        self._model = None
        self._processor = None
        self.model_name = model_name

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self._model.eval()

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def processor(self):
        self._load_model()
        return self._processor

    def embed_text(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            messages = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in batch]
            embeddings = self._encode_batch(messages)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def embed_image(self, image_paths: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]
            messages = [[{"role": "user", "content": [{"type": "image", "image": p}]}] for p in batch]
            embeddings = self._encode_batch(messages)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def embed_multimodal(self, texts: list[str], image_paths: list[Optional[str]]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_images = image_paths[i:i + self.batch_size]
            messages = []
            for text, img_path in zip(batch_texts, batch_images):
                content = []
                if img_path and Path(img_path).exists():
                    content.append({"type": "image", "image": img_path})
                content.append({"type": "text", "text": text})
                messages.append([{"role": "user", "content": content}])
            embeddings = self._encode_batch(messages)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def _encode_batch(self, messages_batch: list) -> np.ndarray:
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        image_inputs = []
        for msg in messages_batch:
            for m in msg:
                for content in m.get("content", []):
                    if content.get("type") == "image":
                        image_inputs.append(Image.open(content["image"]).convert("RGB"))
        inputs = self.processor(
            text=texts, images=image_inputs if image_inputs else None,
            return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = self._pool(outputs.hidden_states[-1], inputs["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().numpy()

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
```

### 5.3 vLLM 部署方案（生产环境推荐）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-Embedding \
    --task embedding \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8100
```

```python
import httpx
import base64


class VLLMEmbeddingClient:
    """调用 vLLM 部署的 Embedding API"""

    def __init__(self, base_url: str = "http://localhost:8100"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": "Qwen/Qwen2.5-VL-Embedding", "input": texts}
        )
        data = resp.json()
        return [d["embedding"] for d in data["data"]]

    async def embed_multimodal(self, text: str, image_path: str = None) -> list[float]:
        content = []
        if image_path:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": text})
        resp = await self.client.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": "Qwen/Qwen2.5-VL-Embedding", "input": [{"type": "multimodal", "content": content}]}
        )
        data = resp.json()
        return data["data"][0]["embedding"]
```

---

## 六、向量存储模块

### 6.1 Milvus 向量存储

```python
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType


class VectorStore:
    """向量存储层 - 基于 Milvus"""

    def __init__(self, uri: str = "http://localhost:19530"):
        self.client = MilvusClient(uri=uri)
        self._ensure_collections()

    def _ensure_collections(self):
        if not self.client.has_collection("doc_chunks"):
            self.client.create_collection(
                collection_name="doc_chunks",
                dimension=1536,
                schema=self._chunk_schema(),
                index_params=self._index_params()
            )

    def _chunk_schema(self) -> CollectionSchema:
        return CollectionSchema(fields=[
            FieldSchema("chunk_id", DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema("doc_id", DataType.VARCHAR, max_length=256),
            FieldSchema("chunk_type", DataType.VARCHAR, max_length=32),
            FieldSchema("text_content", DataType.VARCHAR, max_length=65535),
            FieldSchema("section_path", DataType.VARCHAR, max_length=1024),
            FieldSchema("section_title", DataType.VARCHAR, max_length=512),
            FieldSchema("toc_node_id", DataType.VARCHAR, max_length=256),
            FieldSchema("page", DataType.INT32),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema("metadata_json", DataType.JSON),
        ])

    def _index_params(self):
        return self.client.prepare_index_params(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256}
        )

    async def upsert_chunks(self, chunks: list[dict]):
        self.client.upsert(collection_name="doc_chunks", data=chunks)

    async def search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict = None
    ) -> list[dict]:
        filter_expr = None
        if filters:
            conditions = []
            if "doc_id" in filters:
                conditions.append(f'doc_id == "{filters["doc_id"]}"')
            if "chunk_type" in filters:
                conditions.append(f'chunk_type == "{filters["chunk_type"]}"')
            if "section_path_prefix" in filters:
                prefix = filters["section_path_prefix"]
                conditions.append(f'section_path like "{prefix}%"')
            filter_expr = " and ".join(conditions) if conditions else None

        results = self.client.search(
            collection_name="doc_chunks",
            data=[query_embedding],
            limit=top_k,
            filter=filter_expr,
            output_fields=["chunk_id", "doc_id", "text_content", "chunk_type",
                          "section_path", "section_title", "toc_node_id",
                          "page", "metadata_json"],
            search_params={"metric_type": "COSINE", "params": {"ef": 128}}
        )
        return results[0]

    async def delete_doc(self, doc_id: str):
        self.client.delete(collection_name="doc_chunks", filter=f'doc_id == "{doc_id}"')
```

### 6.2 元数据存储 (PostgreSQL)

```sql
CREATE TABLE documents (
    doc_id          VARCHAR(256) PRIMARY KEY,
    filename        VARCHAR(512) NOT NULL,
    format          VARCHAR(32) NOT NULL,
    title           TEXT,
    author          TEXT,
    page_count      INT,
    file_size       BIGINT,
    file_hash       VARCHAR(64),
    storage_path    TEXT NOT NULL,
    parse_status    VARCHAR(32) DEFAULT 'pending',
    chunk_count     INT DEFAULT 0,
    toc_json        JSONB DEFAULT '{}',           -- 存储完整 TOC 树
    toc_outline     TEXT DEFAULT '',               -- 可读的大纲文本
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),
    metadata        JSONB DEFAULT '{}'
);

CREATE TABLE chunks (
    chunk_id        VARCHAR(256) PRIMARY KEY,
    doc_id          VARCHAR(256) REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_type      VARCHAR(32) NOT NULL,
    toc_node_id     VARCHAR(256),
    section_path    TEXT,
    page            INT,
    token_count     INT,
    has_image       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE knowledge_bases (
    kb_id           VARCHAR(256) PRIMARY KEY,
    name            VARCHAR(512) NOT NULL,
    description     TEXT,
    owner_id        VARCHAR(256),
    doc_count       INT DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE kb_documents (
    kb_id           VARCHAR(256) REFERENCES knowledge_bases(kb_id),
    doc_id          VARCHAR(256) REFERENCES documents(doc_id),
    PRIMARY KEY (kb_id, doc_id)
);

CREATE INDEX idx_documents_hash ON documents(file_hash);
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_chunks_toc_node ON chunks(toc_node_id);
CREATE INDEX idx_kb_docs ON kb_documents(kb_id);
```

---

## 七、Pipeline 整合

### 7.1 完整摄入 Pipeline

```python
import hashlib
import uuid
from pathlib import Path
from celery import Celery

app = Celery("doc_ingestion", broker="redis://localhost:6379/0")


class DocumentIngestionPipeline:
    """
    文档摄入 Pipeline（TOC 驱动版）

    完整流程:
    ① 文件上传 & 去重
    ② 文档解析（MinerU 等）
    ③ LLM 提取 TOC
    ④ TOC 驱动分块（min_tokens 自适应合并）
    ⑤ 多模态 Embedding
    ⑥ 写入向量数据库 + 元数据
    """

    def __init__(self, config: dict = None):
        config = config or {}

        # 文档解析
        self.format_router = FormatRouter()
        self.format_router.register(PDFParser())
        self.format_router.register(DocxParser())

        # TOC 抽取
        self.toc_extractor = LLMTOCExtractor(
            model=config.get("toc_model", "qwen2.5-72b-instruct"),
            base_url=config.get("llm_base_url", "http://localhost:8000/v1"),
        )

        # 分块
        self.chunker = TOCDrivenChunker(
            min_chunk_tokens=config.get("min_chunk_tokens", 200),
            max_chunk_tokens=config.get("max_chunk_tokens", 512),
            overlap_tokens=config.get("overlap_tokens", 64),
        )

        # Embedding
        self.embedder = VLLMEmbeddingClient(
            base_url=config.get("embedding_url", "http://localhost:8100")
        )

        # 存储
        self.vector_store = VectorStore(
            uri=config.get("milvus_uri", "http://localhost:19530")
        )

    async def ingest(self, file_path: str, kb_id: str = "default") -> dict:
        doc_id = str(uuid.uuid4())

        try:
            # ① 预处理 & 去重
            file_hash = self._compute_hash(file_path)
            if await self._is_duplicate(file_hash, kb_id):
                return {"status": "skipped", "reason": "duplicate"}

            await self._update_status(doc_id, "parsing")

            # ② 文档解析
            output_dir = f"/tmp/parse_{doc_id}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            parsed_doc = self.format_router.parse(file_path, output_dir)
            parsed_doc.doc_id = doc_id
            full_text = parsed_doc.raw_text

            # ③ LLM 提取 TOC（带降级）
            await self._update_status(doc_id, "extracting_toc")
            try:
                toc_tree = self.toc_extractor.extract_toc(full_text, doc_id)
                toc_valid = toc_tree.total_nodes >= 2
            except Exception as e:
                toc_valid = False
                toc_tree = None

            # ④ TOC 驱动分块（或降级到滑动窗口）
            await self._update_status(doc_id, "chunking")
            if toc_valid:
                chunks = self.chunker.chunk_document(
                    full_text=full_text,
                    toc_tree=toc_tree,
                    parsed_doc=parsed_doc
                )
            else:
                chunks = self.chunker.fallback_chunk(full_text, doc_id)

            # ⑤ 多模态 Embedding
            await self._update_status(doc_id, "embedding")
            records = []
            for chunk in chunks:
                inputs = chunk.embedding_inputs

                if inputs.get("image"):
                    embedding = await self.embedder.embed_multimodal(
                        text=inputs["text"], image_path=inputs["image"]
                    )
                else:
                    embeddings = await self.embedder.embed_text([inputs["text"]])
                    embedding = embeddings[0]

                records.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": doc_id,
                    "chunk_type": chunk.chunk_type.value,
                    "text_content": chunk.text_content,
                    "section_path": chunk.section_path,
                    "section_title": chunk.section_title,
                    "toc_node_id": chunk.toc_node_id,
                    "page": chunk.page or 0,
                    "embedding": embedding,
                    "metadata_json": {
                        "depth": chunk.depth,
                        "token_count": chunk.token_count,
                        "has_image": chunk.image_path is not None,
                        **chunk.metadata
                    }
                })

            # ⑥ 写入向量数据库 + 元数据
            await self.vector_store.upsert_chunks(records)
            await self._save_metadata(
                doc_id, parsed_doc, kb_id, len(chunks), file_hash,
                toc_outline=toc_tree.to_outline_str() if toc_tree else ""
            )
            await self._update_status(doc_id, "done")

            return {
                "doc_id": doc_id,
                "status": "success",
                "toc_nodes": toc_tree.total_nodes if toc_tree else 0,
                "toc_depth": toc_tree.max_depth if toc_tree else 0,
                "total_chunks": len(chunks),
                "chunking_mode": "toc_driven" if toc_valid else "fallback_sliding_window",
                "toc_outline": toc_tree.to_outline_str() if toc_tree else ""
            }

        except Exception as e:
            await self._update_status(doc_id, "failed", error=str(e))
            raise

    def _compute_hash(self, file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    async def _is_duplicate(self, file_hash: str, kb_id: str) -> bool:
        pass  # 查询 PostgreSQL

    async def _update_status(self, doc_id: str, status: str, error: str = None):
        pass  # 更新 PostgreSQL

    async def _save_metadata(self, doc_id, parsed_doc, kb_id, chunk_count, file_hash, toc_outline=""):
        pass  # 写入 PostgreSQL


# Celery 异步任务
@app.task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_document_task(self, file_path: str, kb_id: str):
    import asyncio
    pipeline = DocumentIngestionPipeline()
    return asyncio.run(pipeline.ingest(file_path, kb_id))
```

### 7.2 FastAPI 接口层

```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="RAG Document Ingestion API")


class IngestResponse(BaseModel):
    task_id: str
    doc_id: str
    status: str


class SearchRequest(BaseModel):
    query: str
    kb_id: str = "default"
    top_k: int = 10
    chunk_type: str = None
    section_filter: str = None    # 按章节路径过滤
    image: str = None


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...), kb_id: str = "default"):
    temp_path = f"/tmp/uploads/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    task = ingest_document_task.delay(temp_path, kb_id)
    return IngestResponse(task_id=task.id, doc_id="pending", status="submitted")


@app.post("/api/v1/search")
async def search(request: SearchRequest):
    embedder = VLLMEmbeddingClient()
    if request.image:
        embedding = await embedder.embed_multimodal(text=request.query, image_path=request.image)
    else:
        embeddings = await embedder.embed_text([request.query])
        embedding = embeddings[0]

    store = VectorStore()
    filters = {}
    if request.chunk_type:
        filters["chunk_type"] = request.chunk_type
    if request.section_filter:
        filters["section_path_prefix"] = request.section_filter

    results = await store.search(
        query_embedding=embedding, top_k=request.top_k, filters=filters
    )

    formatted = []
    for r in results:
        formatted.append({
            "chunk_id": r["chunk_id"],
            "text": r["text_content"],
            "section_path": r["section_path"],
            "section_title": r["section_title"],
            "page": r["page"],
            "score": r.get("score", 0),
            "source": f"[{r['section_path']}] (第{r['page']}页)"
        })
    return {"results": formatted}


@app.get("/api/v1/documents/{doc_id}/toc")
async def get_document_toc(doc_id: str):
    """查询文档的 TOC 结构"""
    pass  # 从 PostgreSQL 读取 toc_json


@app.get("/api/v1/documents/{doc_id}/status")
async def get_status(doc_id: str):
    pass


@app.delete("/api/v1/documents/{doc_id}")
async def delete_document(doc_id: str):
    store = VectorStore()
    await store.delete_doc(doc_id)
    return {"status": "deleted"}
```

---

## 八、关键设计决策 & 优化策略

### 8.1 分块策略对比（为什么选 TOC + min_tokens）

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 固定长度滑动窗口 | 简单 | 破坏语义 | 结构简单的纯文本 |
| 按段落/句子 | 保持语义 | 粒度不均 | 文章、报告 |
| 纯 TOC 叶子节点 | 结构清晰 | 短节太碎，长节太大 | 结构严谨的规范文档 |
| **TOC + min_tokens（本方案）** | **结构+密度均衡** | **需要 LLM 调用** | **通用场景** |

### 8.2 表格处理策略

```
表格 → ┬─ Markdown 文本表示 ─── 文本 Embedding（精确数值检索）
       ├─ 表格截图 ────────── 图片 Embedding（结构理解）
       └─ 自然语言描述 ────── 文本 Embedding（语义检索）
```

### 8.3 公式处理策略

```
公式 → ┬─ LaTeX 源码 ─── 嵌入上下文文本一起 Embedding
       ├─ 公式截图 ──── 图片 Embedding（视觉匹配）
       └─ 自然语言描述 ─ "爱因斯坦质能方程 E=mc²" 文本 Embedding
```

### 8.4 性能优化

| 优化点 | 方案 |
|--------|------|
| 解析加速 | MinerU GPU 推理 + 多进程批处理 |
| TOC 抽取 | LLM 推理缓存 + 短文档单次调用 |
| Embedding 吞吐 | vLLM 部署 + Dynamic Batching |
| 大文件处理 | 分页流式解析，避免内存溢出 |
| 去重 | 文件 SHA256 去重 + MinHash 近似去重 |
| 增量更新 | 基于 hash 的增量 Embedding，仅更新变化的 chunk |
| 缓存 | Redis 缓存高频查询向量 |

### 8.5 质量保障

| 环节 | 质量措施 |
|------|---------|
| TOC 抽取 | start_marker 定位命中率 ≥ 90%，否则降级 |
| 分块 | chunk token 分布监控，[min, max] 区间覆盖率 ≥ 85% |
| Embedding | 随机采样计算 chunk 间余弦相似度分布 |
| 端到端 | 构建评测集，定期 Recall@K 测试 |

---

## 九、部署架构

### 9.1 Docker Compose

```yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - milvus
      - postgres
    environment:
      - EMBEDDING_URL=http://embedding:8100
      - LLM_BASE_URL=http://toc-llm:8000/v1
      - MILVUS_URI=http://milvus:19530
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ragdb

  worker:
    build: .
    command: celery -A pipeline worker --concurrency=4
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  toc-llm:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen2.5-72B-Instruct-AWQ
      --dtype auto
      --max-model-len 16384
      --port 8000
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  embedding:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen2.5-VL-Embedding
      --task embedding
      --dtype bfloat16
      --port 8100
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  milvus:
    image: milvusdb/milvus:v2.4-latest
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pg_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  milvus_data:
  pg_data:
```

### 9.2 硬件需求估算

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| TOC LLM (Qwen-72B-AWQ) | 1× A100 (80GB) | 2× A100 (80GB) |
| Embedding 服务 | 1× A10 (24GB) | 2× A100 (80GB) |
| MinerU 解析 | 1× T4 (16GB) | 1× A10 (24GB) |
| Milvus | 8C32G + 100GB SSD | 16C64G + 500GB NVMe |
| PostgreSQL | 4C8G | 8C16G |
| API + Worker | 8C16G | 16C32G |

> **节省 GPU 方案**: TOC LLM 可使用 Qwen2.5-32B-AWQ（单卡 A100）或接入云端 API。

---

## 十、项目结构

```
rag-doc-ingestion/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── config/
│   ├── settings.py            # 配置管理
│   └── logging.yaml
├── app/
│   ├── main.py                # FastAPI 入口
│   ├── api/
│   │   ├── routes/
│   │   │   ├── ingest.py      # 文档上传 & 摄入
│   │   │   ├── search.py      # 检索接口
│   │   │   └── manage.py      # 文档/TOC 管理
│   │   └── deps.py            # 依赖注入
│   ├── parsers/
│   │   ├── base.py            # BaseParser, ParsedDocument, ContentBlock
│   │   ├── pdf_parser.py      # MinerU PDF 解析
│   │   ├── docx_parser.py     # Word 解析
│   │   ├── pptx_parser.py     # PPT 解析
│   │   ├── xlsx_parser.py     # Excel 解析
│   │   ├── image_parser.py    # 图片 OCR
│   │   └── router.py          # FormatRouter
│   ├── toc/
│   │   ├── models.py          # TOCNode, TOCTree
│   │   ├── extractor.py       # LLMTOCExtractor
│   │   └── prompts.py         # TOC 抽取 Prompt 模板
│   ├── chunking/
│   │   ├── models.py          # Chunk, ChunkType
│   │   ├── toc_chunker.py     # TOCDrivenChunker（核心）
│   │   └── fallback.py        # 滑动窗口降级分块
│   ├── embedding/
│   │   ├── qwen_local.py      # 本地推理
│   │   ├── vllm_client.py     # vLLM API 调用
│   │   └── service.py         # 统一 Embedding 服务
│   ├── storage/
│   │   ├── vector_store.py    # Milvus 操作
│   │   ├── metadata_store.py  # PostgreSQL 操作
│   │   └── object_store.py    # MinIO/S3 操作
│   ├── pipeline/
│   │   ├── ingestion.py       # 摄入 Pipeline
│   │   └── tasks.py           # Celery 任务
│   └── utils/
│       ├── hash.py
│       ├── token_counter.py
│       └── image_utils.py
├── tests/
│   ├── test_toc/              # TOC 抽取测试
│   ├── test_chunking/         # 分块测试
│   ├── test_parsers/
│   ├── test_embedding/
│   └── test_pipeline/
└── scripts/
    ├── init_db.sql
    ├── benchmark.py           # 性能测试
    └── eval_retrieval.py      # 检索质量评估
```

---

## 十一、实施路线图

| 阶段 | 周期 | 目标 |
|------|------|------|
| P0: MVP | 2 周 | PDF/DOCX 解析 + LLM TOC 抽取 + TOC 分块 + Qwen Embedding + Milvus 检索 |
| P1: 多模态 | 2 周 | 表格/图片/公式解析 + 多模态 Embedding + 按章节过滤检索 |
| P2: 工程化 | 2 周 | Celery 异步 + API 完善 + Docker 部署 + 降级方案 + 监控 |
| P3: 优化 | 持续 | Re-ranking、表格 LLM 摘要、TOC 缓存复用、评测体系 |
