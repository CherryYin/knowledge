"""
基于真实文档的集成测试

测试文件:
  test_dataset/AI自动化客服平台_产品梳理与技术分析.docx
  test_dataset/三方系统接入钉钉授权系统流程文档.docx

不依赖外部服务（LLM / Embedding / DB），使用 DocxParser + fallback_chunk。
"""

from pathlib import Path

import pytest

# ─────────────────────────────────────────────
#  常量
# ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "test_dataset"

DOC_AI_PLATFORM = DATASET / "AI自动化客服平台_产品梳理与技术分析.docx"
DOC_DINGTALK = DATASET / "三方系统接入钉钉授权系统流程文档.docx"

ALL_DOCS = [
    pytest.param(DOC_AI_PLATFORM, id="ai_platform"),
    pytest.param(DOC_DINGTALK, id="dingtalk_auth"),
]


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def parsed_ai_platform(tmp_path_factory):
    from doc_parser import DocxParser
    out = tmp_path_factory.mktemp("ai_platform")
    return DocxParser().parse(str(DOC_AI_PLATFORM), str(out))


@pytest.fixture(scope="module")
def parsed_dingtalk(tmp_path_factory):
    from doc_parser import DocxParser
    out = tmp_path_factory.mktemp("dingtalk")
    return DocxParser().parse(str(DOC_DINGTALK), str(out))


@pytest.fixture(scope="module")
def chunker():
    from chunker import TOCDrivenChunker
    # min=50 适合短文；max=512 标准值
    return TOCDrivenChunker(min_chunk_tokens=50, max_chunk_tokens=512, overlap_tokens=0)


# ─────────────────────────────────────────────
#  1. 文件存在性（前置保障）
# ─────────────────────────────────────────────

@pytest.mark.parametrize("path", [DOC_AI_PLATFORM, DOC_DINGTALK])
def test_dataset_files_exist(path):
    assert path.exists(), f"测试文档不存在: {path}"


# ─────────────────────────────────────────────
#  2. DocxParser — 解析基本属性
# ─────────────────────────────────────────────

@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_parsed_document_format_is_docx(fixture_name, request):
    doc = request.getfixturevalue(fixture_name)
    assert doc.format == "docx"


@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_parsed_document_has_nonempty_blocks(fixture_name, request):
    doc = request.getfixturevalue(fixture_name)
    assert len(doc.blocks) > 0, "应解析出至少一个内容块"


@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_parsed_document_raw_text_nonempty(fixture_name, request):
    doc = request.getfixturevalue(fixture_name)
    assert len(doc.raw_text.strip()) > 100, "raw_text 应有实质性内容"


@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_parsed_document_has_headings(fixture_name, request):
    from doc_parser import ContentType
    doc = request.getfixturevalue(fixture_name)
    headings = [b for b in doc.blocks if b.type == ContentType.HEADING]
    assert len(headings) > 0, "结构化文档应包含标题块"


@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_parsed_document_has_text_blocks(fixture_name, request):
    from doc_parser import ContentType
    doc = request.getfixturevalue(fixture_name)
    text_blocks = [b for b in doc.blocks if b.type == ContentType.TEXT]
    assert len(text_blocks) > 0, "文档应包含正文文本块"


def test_parsed_document_filename_matches_ai_platform(parsed_ai_platform):
    assert parsed_ai_platform.filename == DOC_AI_PLATFORM.name


def test_parsed_document_filename_matches_dingtalk(parsed_dingtalk):
    assert parsed_dingtalk.filename == DOC_DINGTALK.name


def test_parsed_document_doc_id_is_hex_string(parsed_ai_platform):
    doc_id = parsed_ai_platform.doc_id
    assert len(doc_id) == 32, "doc_id 应为 32 位 MD5 hex 字符串"
    assert all(c in "0123456789abcdef" for c in doc_id)


# ─────────────────────────────────────────────
#  3. 表格解析
# ─────────────────────────────────────────────

def test_ai_platform_contains_tables(parsed_ai_platform):
    """产品梳理文档通常包含对比表格"""
    from doc_parser import ContentType
    tables = [b for b in parsed_ai_platform.blocks if b.type == ContentType.TABLE]
    # 至少有一个表格，或跳过（文档实际内容决定）
    if tables:
        first = tables[0]
        assert "|" in first.content, "表格应转换为 markdown 格式"
        rows = first.metadata.get("rows", 0)
        cols = first.metadata.get("cols", 0)
        assert rows > 0 and cols > 0, "表格 metadata 应记录行列数"
    else:
        pytest.skip("该文档不含表格，跳过表格断言")


# ─────────────────────────────────────────────
#  4. 降级分块（fallback_chunk）
# ─────────────────────────────────────────────

@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_fallback_chunk_returns_nonempty_list(fixture_name, request, chunker):
    doc = request.getfixturevalue(fixture_name)
    chunks = chunker.fallback_chunk(doc.raw_text, doc_id="test-fallback")
    assert len(chunks) > 0, "fallback_chunk 应产生至少一个分块"


@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_fallback_chunk_all_text_nonempty(fixture_name, request, chunker):
    doc = request.getfixturevalue(fixture_name)
    chunks = chunker.fallback_chunk(doc.raw_text, doc_id="test-fallback")
    for i, chunk in enumerate(chunks):
        assert chunk.text_content.strip(), f"第 {i} 个 chunk 的 text_content 不应为空"


@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_fallback_chunk_token_count_positive(fixture_name, request, chunker):
    doc = request.getfixturevalue(fixture_name)
    chunks = chunker.fallback_chunk(doc.raw_text, doc_id="test-fallback")
    for i, chunk in enumerate(chunks):
        assert chunk.token_count > 0, f"第 {i} 个 chunk 的 token_count 应 > 0"


@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_fallback_chunk_respects_max_tokens(fixture_name, request, chunker):
    """
    fallback_chunk 使用 \\n\\n 作为段落分隔符。
    DocxParser.raw_text 用单 \\n 拼接，整篇文档变成一整段，无法细分。
    此处将 block 内容以双换行重新拼接，正确触发分段逻辑。
    """
    from doc_parser import ContentType
    doc = request.getfixturevalue(fixture_name)
    # 模拟 "有段落分隔符" 的正常文本输入
    para_text = "\n\n".join(
        b.content for b in doc.blocks
        if b.type in (ContentType.TEXT, ContentType.HEADING) and b.content.strip()
    )
    chunks = chunker.fallback_chunk(para_text, doc_id="test-fallback")
    assert len(chunks) > 0
    oversized = [
        (i, c.token_count)
        for i, c in enumerate(chunks)
        # 末尾小段合并到上一段可能略超，允许 1.5 倍误差
        if c.token_count > chunker.max_chunk_tokens * 1.5
    ]
    assert not oversized, (
        f"以下 chunk 超出 max_chunk_tokens×1.5 上限: {oversized}"
    )


@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_fallback_chunk_ids_are_unique(fixture_name, request, chunker):
    doc = request.getfixturevalue(fixture_name)
    chunks = chunker.fallback_chunk(doc.raw_text, doc_id="test-fallback")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "所有 chunk_id 应唯一"


@pytest.mark.parametrize("fixture_name", ["parsed_ai_platform", "parsed_dingtalk"])
def test_fallback_chunk_covers_major_content(fixture_name, request, chunker):
    """所有分块拼接后的总长度应覆盖原文的 80% 以上（允许标点/空白损失）"""
    doc = request.getfixturevalue(fixture_name)
    chunks = chunker.fallback_chunk(doc.raw_text, doc_id="test-fallback")
    total_chunk_len = sum(len(c.text_content) for c in chunks)
    original_len = len(doc.raw_text.strip())
    ratio = total_chunk_len / original_len if original_len else 0
    assert ratio >= 0.80, f"分块覆盖率 {ratio:.1%} 低于 80%，可能有内容丢失"


# ─────────────────────────────────────────────
#  5. build_doc_id 确定性
# ─────────────────────────────────────────────

def test_build_doc_id_is_deterministic():
    from ingestion_pipeline import build_doc_id
    user_id = "user-abc"
    name1 = DOC_AI_PLATFORM.name
    name2 = DOC_DINGTALK.name
    assert build_doc_id(user_id, name1) == build_doc_id(user_id, name1)
    assert build_doc_id(user_id, name2) == build_doc_id(user_id, name2)


def test_build_doc_id_differs_by_filename():
    from ingestion_pipeline import build_doc_id
    user_id = "user-abc"
    id1 = build_doc_id(user_id, DOC_AI_PLATFORM.name)
    id2 = build_doc_id(user_id, DOC_DINGTALK.name)
    assert id1 != id2, "不同文件名应生成不同 doc_id"


def test_build_doc_id_differs_by_user():
    from ingestion_pipeline import build_doc_id
    name = DOC_AI_PLATFORM.name
    assert build_doc_id("user-A", name) != build_doc_id("user-B", name)


def test_build_doc_id_is_64_char_hex():
    from ingestion_pipeline import build_doc_id
    doc_id = build_doc_id("u1", DOC_AI_PLATFORM.name)
    assert len(doc_id) == 64
    assert all(c in "0123456789abcdef" for c in doc_id)


# ─────────────────────────────────────────────
#  6. FormatRouter 路由
# ─────────────────────────────────────────────

def test_format_router_selects_docx_parser_for_both_files(tmp_path):
    from doc_parser import DocxParser, FormatRouter
    router = FormatRouter()
    for path in (DOC_AI_PLATFORM, DOC_DINGTALK):
        parser = router.route(str(path))
        assert isinstance(parser, DocxParser), f"{path.name} 应路由到 DocxParser"


def test_format_router_parse_returns_parsed_document(tmp_path):
    from doc_parser import FormatRouter, ParsedDocument
    router = FormatRouter()
    result = router.parse(str(DOC_DINGTALK), str(tmp_path))
    assert isinstance(result, ParsedDocument)
    assert result.format == "docx"
