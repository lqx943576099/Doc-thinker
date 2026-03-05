from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body

from ..schemas import EntityRelationshipRequest, RelationshipRequest
from ..state import state


router = APIRouter()


async def _get_session_rag_or_raise(session_id: Optional[str]):
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(
            session_id, config, graphcore_kwargs
        )
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        return session_rag
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")


@router.post("/config")
async def update_config(payload: Dict[str, Any] = Body(...)):
    """Update system configuration"""
    if not state.rag_instance:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    config_type = payload.get("type")
    config_data = payload.get("data", {})
    
    try:
        if config_type == "kg":
            # Update Knowledge Graph configuration
            if "kg-storage" in config_data:
                state.rag_instance.config.knowledge_graph_storage_type = config_data["kg-storage"]
            
            if "kg-path" in config_data:
                state.rag_instance.config.knowledge_graph_path = config_data["kg-path"]
            
            if "entity-threshold" in config_data:
                state.rag_instance.config.entity_disambiguation_threshold = float(config_data["entity-threshold"])
            
            if "rel-threshold" in config_data:
                state.rag_instance.config.relationship_validation_threshold = float(config_data["rel-threshold"])
                
            if "enable-auto-validation" in config_data:
                state.rag_instance.config.enable_auto_validation = config_data["enable-auto-validation"] == "on"
            
            # New dual mode parameters
            if "graph-construction-mode" in config_data:
                state.rag_instance.config.graph_construction_mode = config_data["graph-construction-mode"]
                # Also update orchestrator if it exists
                if hasattr(state, "orchestrator") and state.orchestrator:
                    if hasattr(state.orchestrator, "hyper_system") and state.orchestrator.hyper_system:
                        state.orchestrator.hyper_system.graph_construction_mode = config_data["graph-construction-mode"]
            
            if "spacy-model" in config_data:
                state.rag_instance.config.spacy_model = config_data["spacy-model"]
                # Also update orchestrator if it exists
                if hasattr(state, "orchestrator") and state.orchestrator:
                    if hasattr(state.orchestrator, "hyper_system") and state.orchestrator.hyper_system:
                        state.orchestrator.hyper_system.spacy_model = config_data["spacy-model"]

            return {"success": True, "message": "Knowledge graph configuration updated"}
            
        elif config_type == "ui":
            # UI config might not be directly updateable in backend RAG instance
            # but we could store it if needed
            return {"success": True, "message": "UI configuration received (not all fields are persistent)"}
            
        elif config_type == "api":
            # API config might require restart to take effect
            return {"success": True, "message": "API configuration received (restart may be required)"}
            
        else:
            return {"success": False, "message": f"Unknown configuration type: {config_type}"}
            
    except Exception as e:
        return {"success": False, "message": f"Error updating configuration: {str(e)}"}


@router.get("/knowledge-graph/stats-all")
async def get_all_graph_stats():
    """Return node/edge counts for all session graphs."""
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    result: Dict[str, Any] = {"sessions": {}}
    for s in state.session_manager.list_sessions():
        sid = s.get("id")
        if not sid:
            continue
        try:
            config = state.rag_instance.config
            graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
            session_rag = state.session_manager.get_session_rag(sid, config, graphcore_kwargs)
            session_rag.llm_model_func = state.rag_instance.llm_model_func
            session_rag.embedding_func = state.rag_instance.embedding_func
            await session_rag._ensure_graphcore_initialized()
            SG = session_rag.graphcore.chunk_entity_relation_graph
            snd = await SG.get_all_nodes()
            sed = await SG.get_all_edges()
            result["sessions"][sid] = {
                "nodes": len(snd),
                "edges": len(sed),
                "title": s.get("title", "unknown"),
            }
            if hasattr(SG, "_graphml_xml_file"):
                result["sessions"][sid]["path"] = str(getattr(SG, "_graphml_xml_file", ""))
        except Exception as e:
            result["sessions"][sid] = {"error": str(e), "title": s.get("title", "unknown")}
    return result


@router.get("/knowledge-graph/data")
async def get_graph_data(session_id: Optional[str] = None):
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(session_id, config, graphcore_kwargs)
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        target_rag = session_rag
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session graph not found: {e}")

    try:
        if not target_rag.graphcore:
            await target_rag._ensure_graphcore_initialized()
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()

        nodes = []
        edges = []
        max_nodes = 1000  # 提高上限，支持 500+ 节点图谱
        # 优先保留扩展节点（黄色）：is_expanded=1 或 source_id=llm_expansion
        def _is_expanded(n: dict) -> bool:
            ie = n.get("is_expanded")
            if ie is not None and ie != "":
                if ie == 1 or ie == "1" or str(ie).strip() == "1":
                    return True
            sid = str(n.get("source_id") or "").strip()
            return sid == "llm_expansion"
        expanded_nodes = [n for n in nodes_data if _is_expanded(n)]
        other_nodes = [n for n in nodes_data if not _is_expanded(n)]
        nodes_to_use = expanded_nodes + other_nodes[: max(0, max_nodes - len(expanded_nodes))]
        for node_info in nodes_to_use:
            node_id = node_info.get("id") or node_info.get("entity_id") or ""
            if not node_id:
                continue
            is_expanded = _is_expanded(node_info)
            nodes.append(
                {
                    "id": node_id,
                    "label": node_id,
                    "type": node_info.get("entity_type", "unknown"),
                    "size": 20,
                    "color": "#FFD700" if is_expanded else "#3498db",
                    "is_expanded": is_expanded,
                }
            )

        node_ids = set(n["id"] for n in nodes)
        for edge_info in edges_data:
            u = edge_info["source"]
            v = edge_info["target"]
            if u in node_ids and v in node_ids:
                edges.append(
                    {
                        "id": f"{u}-{v}",
                        "source": u,
                        "target": v,
                        "label": edge_info.get("keywords", "related"),
                        "color": "#95a5a6",
                        "width": 1,
                    }
                )

        expanded_in_response = sum(1 for x in nodes if x.get("is_expanded"))
        meta = {
            "total_nodes": len(nodes_data),
            "total_edges": len(edges_data),
            "session_id": session_id,
            "nodes_returned": len(nodes),
            "expanded_in_response": expanded_in_response,
        }
        if hasattr(G, "_graphml_xml_file"):
            meta["graph_file"] = str(getattr(G, "_graphml_xml_file", ""))
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": meta,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract graph data: {str(e)}")


@router.post("/knowledge-graph/expand")
async def expand_knowledge_graph(payload: Dict[str, Any] = Body(default={})):
    """Expand a session knowledge graph with LLM-generated candidate nodes."""
    session_id = payload.get("session_id")
    angle_indices = payload.get("angle_indices")
    apply = payload.get("apply", True)
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(
            session_id, config, graphcore_kwargs
        )
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        target_rag = session_rag
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")
    try:
        if not target_rag.graphcore:
            await target_rag._ensure_graphcore_initialized()
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get graph: {e}")

    llm_fn = getattr(target_rag, "llm_model_func", None)
    if not llm_fn:
        raise HTTPException(status_code=500, detail="LLM not available")
    embed_fn = getattr(target_rag, "embedding_func", None)
    if embed_fn and hasattr(embed_fn, "func"):
        embed_fn = embed_fn.func

    try:
        from docthinker.kg_expansion import KGExpander

        expander = KGExpander(
            llm_func=llm_fn,
            embedding_func=embed_fn,
            min_per_angle=15,
            semantic_dedup_threshold=1.0,
        )
        result = await expander.expand(
            nodes_data,
            edges_data,
            angle_indices=angle_indices if angle_indices is not None else [0, 1, 5],
            apply_to_graph=G if apply else None,
            session_id=session_id,
        )
        return {"success": True, **result}
    except Exception as e:
        err = str(e)
        raise HTTPException(status_code=500, detail=err or "Expansion failed")


@router.get("/knowledge-graph/debug-expanded")
async def debug_expanded_nodes(session_id: Optional[str] = None):
    """Return diagnostics for expanded nodes in a session graph."""
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(
            session_id, config, graphcore_kwargs
        )
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        target_rag = session_rag
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")

    try:
        if not target_rag.graphcore:
            await target_rag._ensure_graphcore_initialized()
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()

        def _is_expanded_node(n: dict) -> bool:
            ie = n.get("is_expanded")
            if ie is not None and ie != "":
                if ie == 1 or ie == "1" or str(ie).strip() == "1":
                    return True
            return str(n.get("source_id") or "").strip() == "llm_expansion"

        expanded = [
            {"id": n.get("id") or n.get("entity_id"), "is_expanded": n.get("is_expanded")}
            for n in nodes_data
            if _is_expanded_node(n)
        ]
        total = len(nodes_data)
        storage_info = {}
        if hasattr(G, "_graphml_xml_file"):
            storage_info["graph_file"] = getattr(G, "_graphml_xml_file", "N/A")

        return {
            "expanded_count": len(expanded),
            "total_nodes": total,
            "expanded_sample": expanded[:20],
            "storage_info": storage_info,
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/stats")
async def get_knowledge_graph_stats(session_id: Optional[str] = None):
    target_rag = await _get_session_rag_or_raise(session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()
        entity_types = sorted(
            {
                str(n.get("entity_type") or "unknown")
                for n in nodes_data
            }
        )
        relationship_types = sorted(
            {
                str(e.get("keywords") or e.get("description") or "related")
                for e in edges_data
            }
        )
        return {
            "session_id": session_id,
            "total_entities": len(nodes_data),
            "total_relationships": len(edges_data),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-graph/entity")
async def add_entity(request: EntityRelationshipRequest):
    target_rag = await _get_session_rag_or_raise(request.session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        props = dict(request.properties or {})
        props.setdefault("entity_type", request.entity_type)
        props.setdefault("source_id", request.document_id)
        await G.upsert_node(request.entity_name, props)
        await G.index_done_callback()
        return {
            "status": "success",
            "entity": {
                "id": request.entity_name,
                "name": request.entity_name,
                "type": props.get("entity_type", request.entity_type),
                "properties": props,
                "document_ids": [request.document_id] if request.document_id else [],
                "session_id": request.session_id,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-graph/relationship")
async def add_relationship(request: RelationshipRequest):
    target_rag = await _get_session_rag_or_raise(request.session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        if not await G.has_node(request.source_entity) or not await G.has_node(request.target_entity):
            raise HTTPException(status_code=404, detail="Source or target entity not found")
        props = dict(request.properties or {})
        props.setdefault("keywords", request.relationship_type)
        props.setdefault("description", request.relationship_type)
        props.setdefault("source_id", request.document_id)
        await G.upsert_edge(request.source_entity, request.target_entity, props)
        await G.index_done_callback()
        return {
            "status": "success",
            "relationship": {
                "id": f"{request.source_entity}-{request.target_entity}",
                "source_id": request.source_entity,
                "target_id": request.target_entity,
                "type": request.relationship_type,
                "properties": props,
                "document_ids": [request.document_id] if request.document_id else [],
                "session_id": request.session_id,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/knowledge-graph/entity/{entity_name}")
async def update_entity(entity_name: str, properties: Dict[str, Any], session_id: Optional[str] = None):
    target_rag = await _get_session_rag_or_raise(session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        if await G.has_node(entity_name):
            await G.upsert_node(entity_name, properties)
            await G.index_done_callback()
            return {"status": "success", "message": f"Entity {entity_name} updated", "session_id": session_id}
        raise HTTPException(status_code=404, detail="Entity not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge-graph/relationship")
async def delete_relationship(source: str, target: str, session_id: Optional[str] = None):
    target_rag = await _get_session_rag_or_raise(session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        if await G.has_edge(source, target):
            await G.remove_edges([(source, target)])
            await G.index_done_callback()
            return {"status": "success", "message": f"Relationship {source}->{target} deleted", "session_id": session_id}
        raise HTTPException(status_code=404, detail="Relationship not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats")
async def memory_stats():
    """Memory engine status summary."""
    if not getattr(state, "memory_engine", None) or state.memory_engine is None:
        return {"enabled": False, "episodes": 0, "edges": 0}
    try:
        episodes = state.memory_engine.episode_store.all_episodes()
        edges = state.memory_engine.graph.get_all_edges()
        return {
            "enabled": True,
            "episodes": len(episodes),
            "edges": len(edges),
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@router.get("/memory/graph-data")
async def memory_graph_data():
    """Graph payload for memory visualization."""
    if not getattr(state, "memory_engine", None) or state.memory_engine is None:
        return {"nodes": [], "edges": [], "metadata": {"source": "memory", "enabled": False}}
    try:
        graph = state.memory_engine.graph
        episodes = state.memory_engine.episode_store.all_episodes()
        nodes = []
        edges = []
        type_color = {"episode": "#3498db", "entity": "#2ecc71", "chunk": "#e67e22"}
        type_label = {"episode": "episode", "entity": "entity", "chunk": "chunk"}
        for nid, nd in graph.get_all_nodes():
            ntype = nd.get("type", "episode")
            label = nid
            if ntype == "episode" and nid in episodes:
                summary_text = episodes[nid].summary or ""
                summary = (summary_text or nid)[:30]
                label = summary + "..." if len(summary_text) > 30 else (summary_text or nid)
            nodes.append(
                {
                    "id": nid,
                    "label": label,
                    "type": type_label.get(ntype, ntype),
                    "size": 20,
                    "color": type_color.get(ntype, "#95a5a6"),
                }
            )
        for e in graph.get_all_edges():
            edges.append(
                {
                    "id": f"{e.source_id}-{e.edge_type.value}-{e.target_id}",
                    "source": e.source_id,
                    "target": e.target_id,
                    "label": e.edge_type.value,
                    "type": e.edge_type.value,
                    "color": "#9b59b6",
                    "width": max(1, int(e.weight * 3)),
                }
            )
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "source": "memory",
                "enabled": True,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }
    except Exception as e:
        return {"nodes": [], "edges": [], "metadata": {"source": "memory", "enabled": True, "error": str(e)}}


@router.post("/memory/consolidate")
async def memory_consolidate(recent_n: int = 50, run_llm: bool = True):
    """Trigger one memory consolidation pass."""
    if not getattr(state, "memory_engine", None) or state.memory_engine is None:
        raise HTTPException(status_code=501, detail="Memory engine not initialized")
    try:
        result = await state.memory_engine.consolidate(
            recent_n=recent_n,
            run_llm=run_llm,
        )
        try:
            dp = state.memory_engine.decay_and_prune(
                decay_factor=0.9,
                max_age_days=30.0,
                min_weight=0.05,
            )
            result["decayed"] = dp.get("decayed", 0)
            result["pruned"] = dp.get("pruned", 0)
        except Exception:
            pass
        state.memory_engine.save()
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/decay-prune")
async def memory_decay_prune(
    decay_factor: float = 0.9,
    max_age_days: float = 30.0,
    min_weight: float = 0.05,
):
    """Run memory edge decay and pruning."""
    if not getattr(state, "memory_engine", None) or state.memory_engine is None:
        raise HTTPException(status_code=501, detail="Memory engine not initialized")
    try:
        result = state.memory_engine.decay_and_prune(
            decay_factor=decay_factor,
            max_age_days=max_age_days,
            min_weight=min_weight,
        )
        state.memory_engine.save()
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
