"""
TOC 驱动分块模块 (Chunking Engine)

核心策略: TOC-Driven + min_tokens 自适应合并
  1. 按 TOC 节点 start_marker 将全文切成各节内容
  2. 自底向上检查每个 TOC 节点的 token 数
     - >= min_tokens → 独立成 chunk
     - < min_tokens  → 向上合并到父节点
     - 合并后 > max_tokens → 内部滑动窗口二次切分
  3. 表格/图片/公式 → 独立 chunk，携带 TOC 路径上下文
  4. 每个 chunk 注入 section_path 前缀用于 Embedding

降级方案: 当 TOC 抽取失败时使用段落滑动窗口分块。
"""

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from doc_parser import ContentType, ParsedDocument
from toc_extractor import TOCNode, TOCNodeType, TOCTree


# ─────────────────────────────────────────────
#  数据模型
# ─────────────────────────────────────────────

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
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    section_path: str = ""           # 完整层级路径: "2. 方法 > 2.1 数据采集"
    section_title: str = ""          # 当前节标题
    toc_node_id: str = ""            # 关联的 TOC 节点 ID
    depth: int = 0                   # TOC 树深度
    metadata: dict = field(default_factory=dict)
    token_count: int = 0

    @property
    def context_enriched_text(self) -> str:
        """带层级上下文的增强文本（用于 Embedding）"""
        parts: list[str] = []
        if self.section_path:
            parts.append(f"[章节路径] {self.section_path}")
        if self.section_title:
            parts.append(f"[标题] {self.section_title}")
        parts.append(self.text_content)
        return "\n".join(parts)

    @property
    def embedding_inputs(self) -> dict:
        """返回适用于 Qwen 多模态 Embedding 的输入字典"""
        inputs: dict = {"text": self.context_enriched_text}
        if self.image_path:
            inputs["image"] = self.image_path
        return inputs


# ─────────────────────────────────────────────
#  TOC 驱动分块器
# ─────────────────────────────────────────────

class TOCDrivenChunker:
    """
    基于 TOC + min_tokens 自适应合并的分块器

    算法（自底向上）:
    ① 按 start_marker 将全文切成各节区域，再分离出每节的"自身直属文本"
    ② 后序遍历（叶 → 根）:
       - 收集节点自身文本 + 所有"冒泡"子节点文本
       - >= min_tokens 且 <= max_tokens → 独立 chunk
       - > max_tokens → 按段落 / 句子边界二次切分
       - < min_tokens → 标记为"待合并"，冒泡给父节点
    ③ 表格/图片/公式块 → 始终独立 chunk
    ④ 降级: TOC 无效时使用滑动窗口
    """

    def __init__(
        self,
        min_chunk_tokens: int = 200,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 64,
    ):
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer = None

    # ── Tokenizer（懒加载）────────────────────────────────────

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2-7B",   # 与 Qwen2.5 通用分词器兼容
                    trust_remote_code=True,
                )
            except Exception:
                # 降级: 中文字符 * 1.5 近似 token 数
                self._tokenizer = _SimpleTokenizer()
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
        parsed_doc: Optional[ParsedDocument] = None,
    ) -> list[Chunk]:
        """
        基于 TOC 树对文档进行分块

        Args:
            full_text:   文档全文
            toc_tree:    LLM 抽取的 TOC 树
            parsed_doc:  文档解析结果（可选，提供表格/图片等结构化信息）

        Returns:
            list[Chunk]
        """
        doc_id = toc_tree.doc_id

        # Step 1: 按 TOC 节点 start_marker 切分全文
        node_contents = self._split_text_by_toc(full_text, toc_tree)

        # Step 2: 分离每节自身直属文本
        node_own_text = self._compute_own_text(node_contents, toc_tree)

        # Step 3: 自底向上合并 + 二次切分 → text chunks
        text_chunks = self._merge_bottom_up(toc_tree, node_own_text, doc_id)

        # Step 4: 提取表格/图片/公式 → media chunks
        media_chunks: list[Chunk] = []
        if parsed_doc:
            media_chunks = self._extract_media_chunks(
                parsed_doc, toc_tree, node_contents, doc_id
            )

        all_chunks = text_chunks + media_chunks
        all_chunks.sort(key=lambda c: (c.page or 0, c.toc_node_id))
        self._assign_page_ranges(all_chunks)
        return all_chunks

    # ═══════════════════════════════════════════════════════════
    #  Step 1: 按 TOC start_marker 切分全文
    # ═══════════════════════════════════════════════════════════

    def _split_text_by_toc(
        self, full_text: str, toc_tree: TOCTree
    ) -> dict[str, str]:
        """返回 {node_id: 该节点及所有子节点覆盖的原始文本}"""
        all_nodes = toc_tree.all_nodes
        node_contents: dict[str, str] = {}

        positioned: list[tuple[int, TOCNode]] = []
        for node in all_nodes:
            if node._marker_pos >= 0:
                positioned.append((node._marker_pos, node))
        positioned.sort(key=lambda x: x[0])

        for i, (pos, node) in enumerate(positioned):
            content_start = pos + len(node.start_marker)
            content_end = positioned[i + 1][0] if i + 1 < len(positioned) else len(full_text)
            node_contents[node.id] = full_text[content_start:content_end].strip()

        # 第一个节点之前的前言文本
        if positioned and positioned[0][0] > 100:
            node_contents["__preamble__"] = full_text[: positioned[0][0]].strip()

        return node_contents

    # ═══════════════════════════════════════════════════════════
    #  Step 2: 分离每节自身直属文本
    # ═══════════════════════════════════════════════════════════

    def _compute_own_text(
        self, node_contents: dict[str, str], toc_tree: TOCTree
    ) -> dict[str, str]:
        """
        去掉第一个子节点之后的内容，得到该节点自身直属文本。

        例如 "2. 方法" 的管辖文本 = "方法概述...(2.1 内容)(2.2 内容)"
             "2. 方法" 的自身文本 = "方法概述..."（第一个子节点 start_marker 之前）
        """
        own_text: dict[str, str] = {}

        for node in toc_tree.all_nodes:
            full = node_contents.get(node.id, "")
            if not full:
                own_text[node.id] = ""
                continue

            if not node.children:
                own_text[node.id] = full
            else:
                first_child = node.children[0]
                if first_child.start_marker and first_child.start_marker in full:
                    cut = full.index(first_child.start_marker)
                    own_text[node.id] = full[:cut].strip()
                else:
                    own_text[node.id] = ""

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
        doc_id: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        node_status: dict[str, str] = {}  # "independent" | "bubble_up"
        bubble_text: dict[str, str] = {}

        for node in reversed(toc_tree.all_nodes):
            own = node_own_text.get(node.id, "").strip()
            text_parts: list[str] = []

            if own:
                text_parts.append(own)

            for child in node.children:
                if node_status.get(child.id) == "bubble_up":
                    bt = bubble_text.get(child.id, "")
                    if bt:
                        text_parts.append(bt)

            combined = "\n\n".join(text_parts).strip()
            token_count = self.count_tokens(combined) if combined else 0

            if token_count == 0:
                node_status[node.id] = "bubble_up"
                bubble_text[node.id] = ""
                continue

            if self.min_chunk_tokens <= token_count <= self.max_chunk_tokens:
                chunks.append(self._make_chunk(combined, doc_id, node, token_count))
                node_status[node.id] = "independent"

            elif token_count > self.max_chunk_tokens:
                chunks.extend(self._split_long_text(combined, doc_id, node))
                node_status[node.id] = "independent"

            else:  # < min_tokens → 冒泡
                bubble_content = f"{node.title}\n{combined}" if node.title else combined
                node_status[node.id] = "bubble_up"
                bubble_text[node.id] = bubble_content

        # 处理仍在冒泡的顶层节点（强制输出）
        remaining_parts: list[str] = []
        for child in toc_tree.root.children:
            if node_status.get(child.id) == "bubble_up":
                bt = bubble_text.get(child.id, "")
                if bt:
                    remaining_parts.append(bt)

        if remaining_parts:
            remaining = "\n\n".join(remaining_parts).strip()
            if remaining:
                tc = self.count_tokens(remaining)
                if tc > self.max_chunk_tokens:
                    chunks.extend(self._split_long_text(remaining, doc_id, toc_tree.root))
                else:
                    chunks.append(self._make_chunk(remaining, doc_id, toc_tree.root, tc))

        # 处理前言
        preamble = node_own_text.get("__preamble__", "").strip()
        if preamble:
            pt = self.count_tokens(preamble)
            preamble_node = TOCNode(id="0", title="前言/概述", level=1, node_type=TOCNodeType.FRONT_MATTER)
            preamble_node.parent = toc_tree.root
            if pt > self.max_chunk_tokens:
                chunks.extend(self._split_long_text(preamble, doc_id, preamble_node))
            else:
                chunks.append(self._make_chunk(preamble, doc_id, preamble_node, pt))

        return chunks

    # ═══════════════════════════════════════════════════════════
    #  二次切分（超长文本）
    # ═══════════════════════════════════════════════════════════

    def _split_long_text(
        self, text: str, doc_id: str, node: TOCNode
    ) -> list[Chunk]:
        """优先段落边界 → 句子边界 → token 硬切"""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_tokens = 0
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = self.count_tokens(para)

            if para_tokens > self.max_chunk_tokens:
                if current_parts:
                    chunks.append(self._make_chunk(
                        "\n\n".join(current_parts), doc_id, node, current_tokens, chunk_idx
                    ))
                    chunk_idx += 1
                    current_parts = []
                    current_tokens = 0
                sent_chunks = self._split_by_sentences(para, doc_id, node, chunk_idx)
                chunks.extend(sent_chunks)
                chunk_idx += len(sent_chunks)
                continue

            if current_tokens + para_tokens > self.max_chunk_tokens and current_parts:
                chunks.append(self._make_chunk(
                    "\n\n".join(current_parts), doc_id, node, current_tokens, chunk_idx
                ))
                chunk_idx += 1
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
            combined = "\n\n".join(current_parts)
            combined_tokens = self.count_tokens(combined)
            if combined_tokens < self.min_chunk_tokens and chunks:
                last_chunk = chunks[-1]
                merged = last_chunk.text_content + "\n\n" + combined
                merged_tokens = self.count_tokens(merged)
                if merged_tokens <= self.max_chunk_tokens:
                    last_chunk.text_content = merged
                    last_chunk.token_count = merged_tokens
                    return chunks
            chunks.append(self._make_chunk(combined, doc_id, node, combined_tokens, chunk_idx))

        return chunks

    def _split_by_sentences(
        self, text: str, doc_id: str, node: TOCNode, start_idx: int
    ) -> list[Chunk]:
        """按句子边界切分超长段落"""
        sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        idx = start_idx

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_tokens = self.count_tokens(sent)

            if current_tokens + sent_tokens > self.max_chunk_tokens and current:
                chunks.append(self._make_chunk(
                    "".join(current), doc_id, node, current_tokens, idx
                ))
                idx += 1
                current = []
                current_tokens = 0

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(self._make_chunk(
                "".join(current), doc_id, node, current_tokens, idx
            ))

        return chunks

    # ═══════════════════════════════════════════════════════════
    #  表格 / 图片 / 公式 独立 Chunk
    # ═══════════════════════════════════════════════════════════

    def _extract_media_chunks(
        self,
        parsed_doc: ParsedDocument,
        toc_tree: TOCTree,
        node_contents: dict[str, str],
        doc_id: str,
    ) -> list[Chunk]:
        """从解析结果中提取表格和图片，生成独立 chunks"""
        chunks: list[Chunk] = []

        for block in parsed_doc.blocks:
            if block.type not in (ContentType.TABLE, ContentType.IMAGE, ContentType.FORMULA):
                continue

            owner = self._find_owning_node(block, toc_tree)
            section_path = owner.path if owner else ""
            section_title = owner.title if owner else ""
            node_id = owner.id if owner else ""

            if block.type == ContentType.TABLE:
                text_desc = (
                    f"[表格] {section_title}\n{block.content}"
                    if block.content
                    else f"[表格] {section_title}"
                )
                chunks.append(Chunk(
                    chunk_id=self._gen_id(doc_id, f"table_{block.page}", 0),
                    doc_id=doc_id,
                    chunk_type=ChunkType.TABLE,
                    text_content=text_desc,
                    image_path=block.image_path,
                    page=block.page,
                    start_page=block.page,
                    end_page=block.page,
                    section_path=section_path,
                    section_title=section_title,
                    toc_node_id=node_id,
                    token_count=self.count_tokens(text_desc),
                    metadata=block.metadata,
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
                    start_page=block.page,
                    end_page=block.page,
                    section_path=section_path,
                    section_title=section_title,
                    toc_node_id=node_id,
                    token_count=self.count_tokens(text_desc),
                ))

            elif block.type == ContentType.FORMULA:
                context = f"[公式] {section_title}: {block.content}"
                chunks.append(Chunk(
                    chunk_id=self._gen_id(doc_id, f"formula_{block.page}", 0),
                    doc_id=doc_id,
                    chunk_type=ChunkType.FORMULA,
                    text_content=context,
                    image_path=block.image_path,
                    page=block.page,
                    start_page=block.page,
                    end_page=block.page,
                    section_path=section_path,
                    section_title=section_title,
                    toc_node_id=node_id,
                    token_count=self.count_tokens(context),
                ))

        return chunks

    def _find_owning_node(self, block, toc_tree: TOCTree) -> Optional[TOCNode]:
        """根据页码找到内容块所属的 TOC 节点（最近的、页码 ≤ 内容块页码的节点）"""
        if not block.page:
            return None
        best: Optional[TOCNode] = None
        for node in toc_tree.all_nodes:
            if node.page and node.page <= block.page:
                if best is None or node.page >= best.page:
                    best = node
        return best

    # ═══════════════════════════════════════════════════════════
    #  降级方案：滑动窗口分块
    # ═══════════════════════════════════════════════════════════

    def fallback_chunk(self, full_text: str, doc_id: str) -> list[Chunk]:
        """
        降级方案：当 TOC 抽取失败时，使用段落边界滑动窗口分块

        Args:
            full_text: 文档全文
            doc_id:    文档 ID

        Returns:
            list[Chunk]
        """
        paragraphs = re.split(r"\n\s*\n", full_text)
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_tokens = 0
        idx = 0
        dummy = TOCNode(id="fallback", title="", level=1)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            pt = self.count_tokens(para)

            if current_tokens + pt > self.max_chunk_tokens and current_parts:
                chunks.append(self._make_chunk(
                    "\n\n".join(current_parts), doc_id, dummy, current_tokens, idx
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
            current_tokens += pt

        if current_parts:
            chunks.append(self._make_chunk(
                "\n\n".join(current_parts), doc_id, dummy, current_tokens, idx
            ))

        return chunks

    # ═══════════════════════════════════════════════════════════
    #  工具方法
    # ═══════════════════════════════════════════════════════════

    def _make_chunk(
        self,
        text: str,
        doc_id: str,
        node: TOCNode,
        token_count: int,
        sub_index: int = 0,
    ) -> Chunk:
        page = node.start_page if node.start_page is not None else node.page
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
            page=page,
            start_page=page,
            end_page=page,
        )

    def _assign_page_ranges(self, chunks: list[Chunk]) -> None:
        """根据下一个 chunk 的 start_page 回填当前 chunk 的 end_page。"""
        if not chunks:
            return

        for chunk in chunks:
            if chunk.start_page is None:
                chunk.start_page = chunk.page
            if chunk.page is None:
                chunk.page = chunk.start_page

        for idx, chunk in enumerate(chunks):
            if chunk.start_page is None:
                chunk.end_page = chunk.end_page if chunk.end_page is not None else chunk.page
                continue
            if idx + 1 < len(chunks):
                next_start = chunks[idx + 1].start_page
                chunk.end_page = next_start if next_start is not None else chunk.start_page
            else:
                chunk.end_page = chunk.start_page

    @staticmethod
    def _gen_id(doc_id: str, node_id: str, idx: int) -> str:
        raw = f"{doc_id}_{node_id}_{idx}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────
#  简易后备 Tokenizer（无需 transformers）
# ─────────────────────────────────────────────

class _SimpleTokenizer:
    """
    近似 token 计数器（不依赖 transformers）
    中文字符按 1 个 token，英文单词按 1 个 token
    """

    def encode(self, text: str) -> list[int]:
        # 中文字符单独计数，英文连续字符视为一个 token
        tokens: list[int] = []
        i = 0
        while i < len(text):
            char = text[i]
            code = ord(char)
            if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
                tokens.append(code)
                i += 1
            elif char.isspace():
                i += 1
            elif char.isalnum() or char in ("_", "-"):
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] in ("_", "-")):
                    j += 1
                tokens.append(hash(text[i:j]) & 0xFFFFFF)
                i = j
            else:
                tokens.append(code)
                i += 1
        return tokens
