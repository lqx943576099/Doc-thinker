import asyncio
from typing import Optional, Any, Dict, List, Tuple
import json
import tempfile
import time
import textwrap
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..schemas import QueryRequest, MultiDocumentQueryRequest
from ..state import state
from ..memory import get_session_memory_engine


router = APIRouter()

FAST_QA_TIMEOUT_SECONDS = 25
SESSION_QUERY_TIMEOUT_SECONDS = 90
FALLBACK_LLM_TIMEOUT_SECONDS = 30


def _looks_like_file_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    file_keywords = [
        "文件", "文档", "上传", "附件", "资料", "本文", "这篇", "这份", "内容", "总结", "概括",
        "图片", "图中", "图里", "这张图", "表格", "pdf",
        "file", "document", "attachment", "uploaded", "summarize", "this doc", "this file",
    ]
    return any(k in q for k in file_keywords)


def _format_thinking_process(details: Any, meta: Dict[str, Any]) -> str:
    lines: List[str] = []
    if meta:
        memory_mode = meta.get("memory_mode")
        retrieval_instruction = meta.get("retrieval_instruction")
        if memory_mode:
            lines.append(f"记忆模式: {memory_mode}")
        if retrieval_instruction:
            lines.append(f"检索指令: {retrieval_instruction}")
    if not isinstance(details, dict) or not details:
        return "\n".join(lines).strip()
    lookup = details.get("lookup")
    if isinstance(lookup, dict):
        lookup_source = lookup.get("lookup_source")
        lookup_range = lookup.get("page_range")
        if lookup_source:
            lines.append(f"内容定位: {lookup_source}")
        if lookup_range:
            lines.append(f"页码范围: {lookup_range}")
    sub_plan = details.get("sub_plan")
    if isinstance(sub_plan, dict):
        lines.append("子问题计划")
        for item in sub_plan.get("sub_questions", []) or []:
            if isinstance(item, dict):
                qid = item.get("id") or ""
                question = item.get("question") or ""
                rationale = item.get("rationale") or ""
                text = f"- {qid} {question}".strip()
                if rationale:
                    text = f"{text}\n  依据: {rationale}"
                lines.append(text)
    sub_answers = details.get("sub_answers")
    if isinstance(sub_answers, list) and sub_answers:
        lines.append("子问题回答")
        for ans in sub_answers:
            if not isinstance(ans, dict):
                continue
            qid = ans.get("id") or ""
            question = ans.get("question") or ""
            answer = ans.get("answer") or ""
            source = ans.get("source") or ans.get("routing") or ""
            confidence = ans.get("confidence")
            context = ans.get("context") or ""
            reasoning = ans.get("reasoning") or ""
            retrieval_strategy = ans.get("retrieval_strategy")
            error = ans.get("error") or ""
            head = f"- {qid} {question}".strip()
            lines.append(head)
            if answer:
                lines.append(f"  回答: {answer}")
            if source:
                lines.append(f"  路由: {source}")
            if isinstance(confidence, (int, float)):
                lines.append(f"  置信度: {confidence:.2f}")
            if context:
                lines.append(f"  证据: {context}")
            if reasoning:
                lines.append(f"  推理: {reasoning}")
            if retrieval_strategy:
                lines.append(f"  检索策略: {retrieval_strategy}")
            if error:
                lines.append(f"  错误: {error}")
    final_synthesis = details.get("final_synthesis")
    if isinstance(final_synthesis, dict) and final_synthesis:
        summary = final_synthesis.get("summary") or final_synthesis.get("final_answer")
        confidence = final_synthesis.get("confidence")
        if summary:
            lines.append("最终合成")
            lines.append(str(summary))
        if isinstance(confidence, (int, float)):
            lines.append(f"合成置信度: {confidence:.2f}")
    if not lines:
        return ""
    return "\n".join(lines).strip()


def _build_sources_from_details(details: Any, evidence: Any = None) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    sub_answers = details.get("sub_answers") if isinstance(details, dict) else None
    if isinstance(sub_answers, list):
        for ans in sub_answers:
            if not isinstance(ans, dict):
                continue
            question = ans.get("question") or ""
            answer = ans.get("answer") or ""
            context = ans.get("context") or ""
            if not (answer or context):
                continue
            content = []
            if question:
                content.append(f"问题: {question}")
            if answer:
                content.append(f"回答: {answer}")
            if context:
                content.append(f"证据: {context}")
            confidence = ans.get("confidence")
            sources.append(
                {
                    "content": "\n".join(content),
                    "confidence": confidence if isinstance(confidence, (int, float)) else 0.5,
                }
            )
    if not sources and isinstance(evidence, dict):
        raw_prompt = evidence.get("raw_prompt")
        if raw_prompt:
            snippet = raw_prompt if len(raw_prompt) <= 800 else f"{raw_prompt[:800]}..."
            sources.append({"content": snippet, "confidence": 0.4})
    return sources[:5]


def _load_content_list(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict) and "content_list" in data:
        content_list = data.get("content_list") or []
    elif isinstance(data, list):
        content_list = data
    else:
        return []
    base_dir = path.parent
    for block in content_list:
        if isinstance(block, dict) and "img_path" in block:
            img_path = Path(block["img_path"])
            if not img_path.is_absolute():
                img_path = (base_dir / img_path).resolve()
            block["img_path"] = img_path.as_posix()
    return content_list


def _extract_text_from_content_list(content_list: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for block in content_list:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
        elif block.get("type") == "table" and block.get("table_html"):
            parts.append(str(block["table_html"]))
        elif block.get("type") == "equation" and block.get("text"):
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def _count_pages_from_content_list(content_list: List[Dict[str, Any]]) -> int:
    max_page = -1
    for block in content_list:
        if not isinstance(block, dict):
            continue
        page_idx = block.get("page_idx")
        if isinstance(page_idx, int):
            max_page = max(max_page, page_idx)
    return max_page + 1 if max_page >= 0 else 0


def _find_latest_content_list(output_dir: Path, stem: str) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = list(output_dir.rglob(f"{stem}_content_list.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _render_text_to_image(text: str, output_dir: Path, name: str) -> Optional[Path]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    text = text.replace("\r", "")
    lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=80))
    if not lines:
        return None
    if len(lines) > 200:
        return None
    line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + 6
    margin = 20
    width = 1200
    height = max(200, margin * 2 + line_height * len(lines))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += line_height
    output_path = output_dir / f"{name}_quick_qa.png"
    image.save(output_path, "PNG")
    return output_path


async def _try_fast_qa(request: QueryRequest) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not request.session_id:
        return None, None
    if not state.session_manager or not state.rag_instance:
        return None, None
    if not getattr(state.rag_instance, "vision_model_func", None):
        return None, None
    files = state.session_manager.get_files(request.session_id)
    if not files:
        return None, None
    target = next((f for f in files if f.get("file_path")), None)
    if not target:
        return None, None
    file_path = Path(str(target.get("file_path")))
    if not file_path.exists():
        return None, None
    file_size = target.get("file_size")
    if not isinstance(file_size, int):
        try:
            file_size = file_path.stat().st_size
        except Exception:
            file_size = None
    file_ext = (target.get("file_ext") or file_path.suffix).lower()
    max_bytes = 300 * 1024
    max_chars = 4000
    max_pages = 2
    short_by_size = file_size is not None and file_size <= max_bytes

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    text_exts = {".txt", ".md"}

    image_path: Optional[Path] = None
    text_content = ""
    page_count = 0
    content_list = None

    if file_ext in image_exts and short_by_size:
        image_path = file_path
    else:
        if file_ext in text_exts:
            try:
                text_content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                text_content = ""
        else:
            try:
                output_dir = Path(state.rag_instance.config.parser_output_dir)
                content_list_path = _find_latest_content_list(output_dir, file_path.stem)
                if content_list_path:
                    content_list = _load_content_list(content_list_path)
                    text_content = _extract_text_from_content_list(content_list)
                    page_count = _count_pages_from_content_list(content_list)
            except Exception:
                text_content = ""

        if text_content:
            is_short = len(text_content) <= max_chars and (short_by_size or page_count <= max_pages)
            if is_short:
                session = state.session_manager.get_session(request.session_id)
                output_base = Path(session["path"]) if session and session.get("path") else Path(tempfile.mkdtemp())
                image_path = _render_text_to_image(text_content, output_base / "quick_qa", file_path.stem)

    if not image_path or not image_path.exists():
        return None, None

    prompt = f"请根据图片中的文档内容回答用户问题：{request.question}"
    try:
        answer = await state.rag_instance.vision_model_func(prompt, image_data=str(image_path))
    except Exception:
        return None, None
    if not answer:
        return None, None
    return answer, {"fast_qa": True, "file": file_path.name}


async def _get_session_rag_or_raise(session_id: Optional[str]):
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required; global knowledge is disabled",
        )
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")

    config = state.rag_instance.config
    graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
    session_rag = state.session_manager.get_session_rag(
        session_id, config, graphcore_kwargs
    )
    session_rag.llm_model_func = state.rag_instance.llm_model_func
    session_rag.embedding_func = state.rag_instance.embedding_func
    await session_rag._ensure_graphcore_initialized()
    return session_rag


async def _ingest_chat_turn(
    question: str,
    answer: str,
    session_id: Optional[str],
    timestamp: Optional[float] = None,
):
    if not state.ingestion_service:
        return
    base_text = f"User Question: {question}\nAssistant Answer: {answer}"
    text_to_ingest = base_text
    insight = None
    if state.cognitive_processor:
        try:
            insight = await state.cognitive_processor.process(text_to_ingest, source_type="chat")
            link_names = ", ".join([l.name for l in insight.potential_links[:10]])
            text_to_ingest += (
                f"\n\n[Cognitive Analysis]:\n"
                f"Summary: {insight.summary}\n"
                f"Concepts: {', '.join(insight.concepts)}\n"
                f"Potential Links: {link_names}\n"
                f"Reasoning: {insight.reasoning}"
            )
        except Exception:
            insight = None
    if insight:
        metadata = {
            "summary": insight.summary,
            "reasoning": insight.reasoning,
            "key_points": insight.key_points,
            "concepts": insight.concepts,
            "entities": [e.dict() for e in insight.entities],
            "relations": [r.dict() for r in insight.relations],
            "inferred_relations": [r.dict() for r in insight.inferred_relations],
            "hypotheses": insight.hypotheses,
            "action_items": insight.action_items,
        }
        memory_engine = get_session_memory_engine(session_id)
        if memory_engine:
            try:
                relation_triples = []
                for r in list(insight.relations or []) + list(insight.inferred_relations or []):
                    relation_triples.append((getattr(r, "source", ""), getattr(r, "relation", "related_to"), getattr(r, "target", "")))
                await memory_engine.add_observation(
                    summary=insight.summary or base_text[:500],
                    key_points=list(insight.key_points or []),
                    concepts=list(insight.concepts or []),
                    entity_ids=[getattr(e, "name", str(e)) for e in (insight.entities or [])],
                    relation_triples=relation_triples,
                    source_type="chat",
                    session_id=session_id,
                    existing_insight=insight,
                    timestamp=timestamp,
                )
                memory_engine.save()
            except Exception:
                pass
    if session_id:
        try:
            await state.ingestion_service.ingest_text(text_to_ingest, session_id=session_id)
        except Exception:
            pass


@router.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    print(f"DEBUG: Received query: {request.question}, session_id: {request.session_id}")
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not request.session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required; global knowledge is disabled",
        )
    session_rag = await _get_session_rag_or_raise(request.session_id)
    effective_memory_mode = "session"

    # 1. Simple Intent Check (Identity/Greeting)
    identity_keywords = ["你是谁", "who are you", "your name", "介绍一下自己", "你好", "hello", "hi"]
    is_identity_query = any(k in request.question.lower() for k in identity_keywords) and len(request.question) < 20

    if request.session_id:
        try:
            state.session_manager.add_message(request.session_id, "user", request.question)
        except Exception:
            pass

    answer = ""
    sources: List[Dict[str, Any]] = []
    thinking_process = None
    answer_mode = "rag"
    
    try:
        # 2. Fast Path for Identity Questions
        if is_identity_query:
            # ... (keep existing identity logic)
            system_prompt = ("你是由 WhiteCat 团队开发的智能知识与内容系统，用于帮助用户管理和理解文档信息，并进行检索、对比和结构化分析。")
            llm_resp = await state.rag_instance.llm_model_func(f"{system_prompt}\n\nUser: {request.question}\nAssistant:")
            answer = llm_resp if llm_resp else "抱歉，我暂时无法回答这个问题。"
        elif request.session_id and _looks_like_file_question(request.question):
            try:
                fast_answer, fast_meta = await asyncio.wait_for(
                    _try_fast_qa(request), timeout=FAST_QA_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                fast_answer, fast_meta = None, None
            if fast_answer:
                answer = fast_answer
                answer_mode = "fast_qa"
                thinking_process = "快问快答"
                sources = [{"content": f"文档: {fast_meta.get('file', '')}", "confidence": 0.6}] if fast_meta else []
        
        if not answer and request.enable_thinking:
            print(f"DEBUG: Using session-only thinking mode for query: {request.question}")
            context_prefix = ""
            analogies: List[Tuple[Any, float, Optional[str]]] = []
            memory_engine = get_session_memory_engine(request.session_id)
            if memory_engine:
                try:
                    analogies = await memory_engine.retrieve_analogies(
                        request.question, top_k=5, then_spread=True, spread_top_k=3
                    )
                    if analogies:
                        lines = ["[Session Memory] 与当前问题相关的历史片段:"]
                        for ep, score, hint in analogies[:5]:
                            lines.append(f"- {ep.summary[:200]}...")
                            if hint:
                                lines.append(f"  关联: {hint}")
                        context_prefix += "\n".join(lines) + "\n\n"
                        # 神经可塑性：记录 episode 与 episode 实体的共激活关系
                        ep_ids = [ep.episode_id for ep, _, _ in analogies]
                        kg_ids = getattr(state, "kg_entity_ids", set())
                        ent_ids = []
                        for ep, _, _ in analogies:
                            ent_ids.extend(ep.entity_ids or [])
                        ent_ids = [e for e in dict.fromkeys(ent_ids) if e and e in kg_ids]
                        if ep_ids or ent_ids:
                            try:
                                memory_engine.record_co_activation(ep_ids, ent_ids)
                                memory_engine.save()
                            except Exception:
                                pass
                except Exception:
                    pass
            session_result = ""
            try:
                session_result = await asyncio.wait_for(
                    session_rag.aquery(
                        query=request.question,
                        mode=request.mode,
                        enable_rerank=request.enable_rerank,
                    ),
                    timeout=SESSION_QUERY_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                session_result = ""
            except Exception:
                session_result = ""
            context_prefix += f"[Session Context]\n{session_result}\n\n"
            if request.retrieval_instruction:
                context_prefix += f"[Retrieval Instruction]\n{request.retrieval_instruction}\n\n"
            answer = session_result
            answer_mode = "session_thinking"
            thinking_details = {
                "memory_hits": len(analogies),
                "mode": request.mode,
            }
            thinking_process = _format_thinking_process(
                thinking_details,
                {
                    "memory_mode": effective_memory_mode,
                    "retrieval_instruction": request.retrieval_instruction,
                },
            )
            if hasattr(session_rag, "get_last_query_evidence"):
                sources = _build_sources_from_details({}, session_rag.get_last_query_evidence())

        # 3. Session-only RAG Logic (no global sharing)
        elif not answer:
            try:
                answer = await asyncio.wait_for(
                    session_rag.aquery(
                        query=request.question,
                        mode=request.mode,
                        enable_rerank=request.enable_rerank,
                    ),
                    timeout=SESSION_QUERY_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                answer = ""

        # 4. Fallback if RAG returns nothing useful
        if not answer:
            answer = "抱歉，我未能找到相关信息。"

        negative_responses = ["i don't know", "不知道", "没有找到", "sorry", "抱歉", "无法回答"]
        if not is_identity_query and any(n in (answer or "").lower() for n in negative_responses) and len(answer or "") < 100:
            fallback_prompt = (
                f"用户提出了问题：'{request.question}'。"
                "RAG 未从文档中检索到足够信息。请作为通用助手给出清晰回答；"
                "如果确实无法确定，请明确说明并建议补充文档。"
            )
            llm_resp = None
            try:
                llm_resp = await asyncio.wait_for(
                    state.rag_instance.llm_model_func(fallback_prompt),
                    timeout=FALLBACK_LLM_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                llm_resp = None
            if llm_resp:
                answer = llm_resp

        if request.session_id:
            try:
                state.session_manager.add_message(request.session_id, "assistant", answer)
            except Exception:
                pass

        chat_turn_ts = time.time()
        background_tasks.add_task(
            _ingest_chat_turn,
            request.question,
            answer,
            request.session_id,
            chat_turn_ts,
        )

        if not sources and hasattr(session_rag, "get_last_query_evidence"):
            evidence = session_rag.get_last_query_evidence()
            sources = _build_sources_from_details({}, evidence)

        return {
            "answer": answer,
            "query": request.question,
            "mode": request.mode,
            "session_id": request.session_id,
            "memory_mode": effective_memory_mode,
            "thinking_process": thinking_process,
            "sources": sources,
            "answer_mode": answer_mode
        }
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "api_key" in err_msg or "authentication" in err_msg:
            raise HTTPException(
                status_code=401, 
                detail="API key 配置错误或失效，请检查环境变量 LLM_BINDING_API_KEY。"
            )
        elif "403" in err_msg:
            raise HTTPException(status_code=403, detail="API 访问被拒绝，请检查权限。")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/text")
async def query_text(request: QueryRequest, background_tasks: BackgroundTasks):
    """Alias for /query to match frontend expectations"""
    return await query(request, background_tasks)


@router.post("/query/multi-document")
async def query_multi_document(request: MultiDocumentQueryRequest):
    if not state.rag_instance:
        raise HTTPException(status_code=500, detail="Service not initialized")
    session_rag = await _get_session_rag_or_raise(request.session_id)
    try:
        result = await session_rag.aquery_multi_document_enhanced(
            query=request.question,
            mode=request.mode,
            enable_rerank=request.enable_rerank,
        )
        return {
            "answer": result["answer"],
            "query": request.question,
            "mode": request.mode,
            "related_documents": result["related_documents"],
            "extracted_entities": result["extracted_entities"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


