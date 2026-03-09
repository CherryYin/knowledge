from chunker import TOCDrivenChunker
from doc_parser import ContentBlock, ContentType, ParsedDocument
from toc_extractor import LLMTOCExtractor


def _build_realistic_parsed_doc() -> ParsedDocument:
    blocks = [
        ContentBlock(type=ContentType.HEADING, content="1 引言", page=1),
        ContentBlock(type=ContentType.TEXT, content="引言部分介绍研究背景与目标。", page=1),
        ContentBlock(type=ContentType.HEADING, content="2 方法", page=3),
        ContentBlock(type=ContentType.TEXT, content="方法部分说明数据处理与训练流程。", page=3),
        ContentBlock(type=ContentType.HEADING, content="3 实验", page=5),
        ContentBlock(type=ContentType.TEXT, content="实验部分给出结果对比与分析。", page=5),
    ]
    raw_text = "\n".join(
        b.content for b in blocks if b.type in (ContentType.TEXT, ContentType.HEADING)
    )
    return ParsedDocument(
        doc_id="doc-real-001",
        filename="真实样例文档.pdf",
        format="pdf",
        blocks=blocks,
        raw_text=raw_text,
    )


def _build_extractor_with_mocked_toc() -> LLMTOCExtractor:
    extractor = LLMTOCExtractor(model="mock", base_url="http://mock", api_key="mock")
    extractor._detect_existing_toc = lambda _text: None  # type: ignore[method-assign]
    extractor._split_into_segments = lambda text: [text]  # type: ignore[method-assign]
    extractor._extract_from_segment = lambda _segment, context_hint="": [  # type: ignore[method-assign]
        {
            "id": "1",
            "title": "1 引言",
            "level": 1,
            "node_type": "chapter",
            "start_marker": "1 引言",
            "page": None,
        },
        {
            "id": "2",
            "title": "2 方法",
            "level": 1,
            "node_type": "chapter",
            "start_marker": "2 方法",
            "page": None,
        },
        {
            "id": "3",
            "title": "3 实验",
            "level": 1,
            "node_type": "chapter",
            "start_marker": "3 实验",
            "page": None,
        },
    ]
    return extractor


def test_toc_extractor_infers_start_page_from_realistic_document_blocks():
    parsed_doc = _build_realistic_parsed_doc()
    extractor = _build_extractor_with_mocked_toc()

    toc_tree = extractor.extract_toc(
        full_text=parsed_doc.raw_text,
        doc_id=parsed_doc.doc_id,
        parsed_doc=parsed_doc,
    )

    start_pages = [node.start_page for node in toc_tree.all_nodes]
    assert start_pages == [1, 3, 5]
    assert [node.page for node in toc_tree.all_nodes] == [1, 3, 5]


def test_chunker_sets_end_page_from_next_chunk_start_page():
    parsed_doc = _build_realistic_parsed_doc()
    extractor = _build_extractor_with_mocked_toc()
    toc_tree = extractor.extract_toc(
        full_text=parsed_doc.raw_text,
        doc_id=parsed_doc.doc_id,
        parsed_doc=parsed_doc,
    )

    chunker = TOCDrivenChunker(min_chunk_tokens=1, max_chunk_tokens=4096, overlap_tokens=0)
    chunks = chunker.chunk_document(
        full_text=parsed_doc.raw_text,
        toc_tree=toc_tree,
        parsed_doc=None,
    )

    assert len(chunks) == 3
    assert [chunk.start_page for chunk in chunks] == [1, 3, 5]

    for idx in range(len(chunks) - 1):
        assert chunks[idx].end_page == chunks[idx + 1].start_page

    assert chunks[-1].end_page == chunks[-1].start_page
