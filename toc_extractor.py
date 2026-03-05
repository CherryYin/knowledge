"""
LLM TOC 抽取模块 (Table of Contents Extractor)

使用大模型从文档全文中提取结构化目录，支持：
- 有/无编号标题
- OCR / 扫描文档（无样式信息）
- 隐式逻辑结构

推荐 LLM: Qwen2.5-72B-Instruct（中文效果最好）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from doc_parser import ContentType, ParsedDocument


# ─────────────────────────────────────────────
#  数据模型
# ─────────────────────────────────────────────

class TOCNodeType(Enum):
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    APPENDIX = "appendix"
    FRONT_MATTER = "front_matter"
    BACK_MATTER = "back_matter"


@dataclass
class TOCNode:
    """TOC 树节点"""
    id: str
    title: str
    level: int
    node_type: TOCNodeType = TOCNodeType.SECTION
    start_marker: str = ""
    page: Optional[int] = None
    start_page: Optional[int] = None
    children: list = field(default_factory=list)
    parent: Optional["TOCNode"] = field(default=None, repr=False)

    # 内部定位字段
    _marker_pos: int = field(default=-1, repr=False)

    @property
    def path(self) -> str:
        """返回从根到当前节点的完整路径，如 '2. 方法 > 2.1 数据采集'"""
        parts: list[str] = []
        node: Optional[TOCNode] = self
        while node and node.level > 0:
            parts.append(node.title)
            node = node.parent
        return " > ".join(reversed(parts))

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def flatten(self) -> list["TOCNode"]:
        """前序遍历扁平化"""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result


@dataclass
class TOCTree:
    """完整的 TOC 树"""
    doc_id: str
    root: TOCNode
    total_nodes: int = 0
    max_depth: int = 0

    @property
    def all_nodes(self) -> list[TOCNode]:
        """除虚拟根节点外的所有节点（前序）"""
        return self.root.flatten()[1:]

    @property
    def leaf_nodes(self) -> list[TOCNode]:
        return [n for n in self.all_nodes if n.is_leaf]

    def get_nodes_at_level(self, level: int) -> list[TOCNode]:
        return [n for n in self.all_nodes if n.level == level]

    def to_outline_str(self) -> str:
        """输出缩进格式的大纲字符串"""
        lines = []
        for node in self.all_nodes:
            indent = "  " * (node.level - 1)
            lines.append(f"{indent}{node.title}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  LLM TOC 抽取器
# ─────────────────────────────────────────────

class LLMTOCExtractor:
    """
    使用大模型从文档全文中抽取结构化 TOC

    支持的 LLM 后端（OpenAI 兼容接口）:
    - Qwen2.5-72B / 32B（推荐，中文效果最好）
    - GPT-4o / Claude（英文文档）
    - 本地 vLLM / Ollama

    长文档策略: 分段送入 LLM → 各段独立抽取局部 TOC → 合并校验
    """

    # ── Prompts ───────────────────────────────────────────────

    SYSTEM_PROMPT = """你是一个专业的文档结构分析助手。你的任务是从给定的文档文本中提取目录结构（Table of Contents）。

## 任务要求

1. **识别所有层级的标题和章节**，包括：
   - 明确的编号标题（如 "1. 引言"、"2.1 方法"）
   - 无编号但有逻辑层次的标题（如 "背景"、"实验设置"）
   - 隐式段落主题（当文档没有明确标题时，提取关键段落的主题作为标题）

2. **准确判断层级关系**：
   - level 1: 最高级标题（章/部分）
   - level 2: 次级标题（节）
   - level 3: 三级标题（小节）
   - level 4+: 更深层级

3. **提供定位锚点**：
   - `start_marker` 必须是文档原文中**完全存在**的一段文字（通常是标题本身）
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
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_segment_tokens = max_segment_tokens
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("openai is not installed. Run: pip install openai") from exc
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    # ── 主入口 ────────────────────────────────────────────────

    def extract_toc(
        self,
        full_text: str,
        doc_id: str = "",
        parsed_doc: Optional[ParsedDocument] = None,
    ) -> TOCTree:
        """
        从完整文档文本中抽取 TOC

        流程:
        1. 检测文档是否自带目录页 → 有则优先解析
        2. 文档分段 → 每段调用 LLM 抽取局部 TOC
        3. 合并局部 TOC → 全局校验
        4. 校验 start_marker → 构建 TOCTree
        """
        existing_toc = self._detect_existing_toc(full_text)

        if existing_toc:
            toc_nodes = self._parse_existing_toc(existing_toc, full_text)
        else:
            segments = self._split_into_segments(full_text)
            if len(segments) == 1:
                toc_nodes = self._extract_from_segment(segments[0])
            else:
                partial_tocs: list[list[dict]] = []
                for i, segment in enumerate(segments):
                    partial = self._extract_from_segment(
                        segment,
                        context_hint=f"这是文档的第 {i + 1}/{len(segments)} 部分",
                    )
                    partial_tocs.append(partial)
                toc_nodes = self._merge_partial_tocs(partial_tocs)

        toc_nodes = self._validate_markers(toc_nodes, full_text)
        toc_nodes = self._infer_start_pages(toc_nodes, parsed_doc)
        return self._build_tree(toc_nodes, doc_id)

    # ── 检测已有目录 ──────────────────────────────────────────

    def _detect_existing_toc(self, text: str) -> Optional[str]:
        """检测文档中是否包含目录页"""
        patterns = [
            r"(?:table\s+of\s+contents|目\s*录|目\s*次)\s*\n",
            r"(?:contents)\s*\n.*?(?=\n\s*(?:chapter|第|1[\.\s]))",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.start()
                candidate = text[start: start + 3000]
                lines = [line.strip() for line in candidate.split("\n") if line.strip()]
                numbered = sum(1 for l in lines if re.match(r"^[\d\.\s]", l))
                if numbered >= 3:
                    return candidate
        return None

    def _parse_existing_toc(self, toc_text: str, full_text: str) -> list[dict]:
        prompt = (
            f"以下是从文档中提取的目录页内容：\n\n---\n{toc_text}\n---\n\n"
            f"以下是文档正文的前 2000 字（用于验证）：\n\n---\n{full_text[:2000]}\n---\n\n"
            "请根据目录页内容，结合正文验证，输出结构化的 TOC JSON。\n"
            "start_marker 必须能在正文中精确定位。"
        )
        return self._call_llm(prompt)

    # ── 分段 & 提取 ──────────────────────────────────────────

    def _split_into_segments(self, text: str) -> list[str]:
        """按 token 限制分段，在段落边界切分（保留 2 段重叠）"""
        max_chars = self.max_segment_tokens * 2  # 粗略字符估算

        if len(text) <= max_chars:
            return [text]

        segments: list[str] = []
        paragraphs = text.split("\n")
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > max_chars and current:
                segments.append("\n".join(current))
                overlap = current[-2:] if len(current) >= 2 else current[-1:]
                current = list(overlap)
                current_len = sum(len(p) for p in current)
            current.append(para)
            current_len += para_len

        if current:
            segments.append("\n".join(current))

        return segments

    def _extract_from_segment(self, segment_text: str, context_hint: str = "") -> list[dict]:
        prompt = (
            f"请从以下文档文本中提取目录结构。\n\n"
            f"{f'提示: {context_hint}' if context_hint else ''}\n\n"
            f"---文档内容开始---\n{segment_text}\n---文档内容结束---\n\n"
            "请提取其中所有标题和章节结构，输出 JSON。"
        )
        return self._call_llm(prompt)

    # ── 合并多段 TOC ─────────────────────────────────────────

    def _merge_partial_tocs(self, partial_tocs: list[list[dict]]) -> list[dict]:
        formatted = []
        for i, toc in enumerate(partial_tocs):
            formatted.append(f"## 片段 {i + 1}:\n{json.dumps(toc, ensure_ascii=False, indent=2)}")
        prompt = self.MERGE_PROMPT.format(partial_tocs="\n\n".join(formatted))
        return self._call_llm(prompt, raw_prompt=prompt)

    # ── LLM 调用 ─────────────────────────────────────────────

    def _call_llm(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        raw_prompt: Optional[str] = None,
    ) -> list[dict]:
        if raw_prompt:
            messages = [{"role": "user", "content": raw_prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
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
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for key in ("toc", "items", "nodes", "data", "result"):
                    if key in result and isinstance(result[key], list):
                        return result[key]
                for v in result.values():
                    if isinstance(v, list):
                        return v
        except json.JSONDecodeError:
            pass

        return self._retry_extraction(content, user_prompt)

    def _extract_json(self, text: str) -> str:
        """从 LLM 输出中剥离 Markdown 代码块，提取 JSON 字符串"""
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
        for pattern in (r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"):
            m = re.search(pattern, text)
            if m:
                return m.group(1)
        return text

    def _retry_extraction(self, failed_output: str, original_prompt: str) -> list[dict]:
        retry_prompt = "上一次输出无法解析为有效 JSON，请重新输出。只输出 JSON 数组，不要有其他文字。"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": original_prompt},
                {"role": "assistant", "content": failed_output[:500]},
                {"role": "user", "content": retry_prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        content = response.choices[0].message.content.strip()
        json_str = self._extract_json(content)
        try:
            result = json.loads(json_str)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []

    # ── 校验 start_marker ────────────────────────────────────

    def _validate_markers(self, toc_nodes: list[dict], full_text: str) -> list[dict]:
        """
        三级匹配策略校验每个节点的 start_marker:
        1. 精确匹配
        2. 标准化匹配（去多余空格）
        3. 标题关键词模糊搜索兜底
        """
        validated: list[dict] = []
        normalized_full = re.sub(r"\s+", " ", full_text)

        for node in toc_nodes:
            marker: str = node.get("start_marker", "")
            title: str = node.get("title", "")

            # 1. 精确匹配
            if marker and marker in full_text:
                node["_marker_pos"] = full_text.index(marker)
                validated.append(node)
                continue

            # 2. 标准化匹配
            norm_marker = re.sub(r"\s+", " ", marker).strip()
            if norm_marker and norm_marker in normalized_full:
                pos = normalized_full.index(norm_marker)
                node["start_marker"] = norm_marker
                node["_marker_pos"] = pos
                validated.append(node)
                continue

            # 3. 标题模糊搜索
            if title:
                pattern = rf"(?:[\d\.]+\s*)?{re.escape(title)}"
                m = re.search(pattern, full_text, re.IGNORECASE)
                if m:
                    node["start_marker"] = m.group(0)
                    node["_marker_pos"] = m.start()
                    validated.append(node)
                    continue

            # 兜底：标记为未定位，仍然保留
            node["_marker_pos"] = -1
            node["_unlocated"] = True
            validated.append(node)

        validated.sort(key=lambda n: n.get("_marker_pos", float("inf")))
        return validated

    def _infer_start_pages(
        self,
        toc_nodes: list[dict],
        parsed_doc: Optional[ParsedDocument],
    ) -> list[dict]:
        """基于解析块和 marker 位置推断每个 TOC 节点的 start_page。"""
        if not parsed_doc or not parsed_doc.blocks:
            for node in toc_nodes:
                page = node.get("page")
                node["start_page"] = int(page) if isinstance(page, int) else page
            return toc_nodes

        text_blocks = [
            b for b in parsed_doc.blocks
            if b.type in (ContentType.TEXT, ContentType.HEADING) and b.content
        ]

        spans: list[tuple[int, int, Optional[int], str]] = []
        cursor = 0
        for idx, block in enumerate(text_blocks):
            block_text = block.content
            start = cursor
            end = start + len(block_text)
            spans.append((start, end, block.page, block_text))
            cursor = end + (1 if idx < len(text_blocks) - 1 else 0)

        def find_page_by_pos(pos: int) -> Optional[int]:
            if pos < 0:
                return None
            for start, end, page, _ in spans:
                if start <= pos <= end and page is not None:
                    return page
            prev_page: Optional[int] = None
            for start, _, page, _ in spans:
                if start > pos:
                    break
                if page is not None:
                    prev_page = page
            return prev_page

        def find_page_by_text(marker: str, title: str) -> Optional[int]:
            for _, _, page, block_text in spans:
                if page is None:
                    continue
                if marker and marker in block_text:
                    return page
                if title and title in block_text:
                    return page
            return None

        for node in toc_nodes:
            page = node.get("page")
            if not isinstance(page, int):
                page = None

            if page is None:
                pos = node.get("_marker_pos", -1)
                if isinstance(pos, int):
                    page = find_page_by_pos(pos)

            if page is None:
                marker = node.get("start_marker", "") or ""
                title = node.get("title", "") or ""
                page = find_page_by_text(marker, title)

            node["start_page"] = page
            node["page"] = page

        return toc_nodes

    # ── 构建树 ───────────────────────────────────────────────

    def _build_tree(self, toc_nodes: list[dict], doc_id: str) -> TOCTree:
        """将扁平 TOC 列表构建为有序树结构"""
        root = TOCNode(id="root", title="Document Root", level=0)
        stack: list[TOCNode] = [root]
        max_depth = 0

        for d in toc_nodes:
            level = d.get("level", 1)
            max_depth = max(max_depth, level)

            try:
                node_type = TOCNodeType(d.get("node_type", "section"))
            except ValueError:
                node_type = TOCNodeType.SECTION

            node = TOCNode(
                id=d.get("id", ""),
                title=d.get("title", ""),
                level=level,
                node_type=node_type,
                start_marker=d.get("start_marker", ""),
                page=d.get("start_page", d.get("page")),
                start_page=d.get("start_page", d.get("page")),
            )
            node._marker_pos = d.get("_marker_pos", -1)

            # 找到合适的父节点（弹栈直到父层级 < 当前层级）
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
            max_depth=max_depth,
        )
