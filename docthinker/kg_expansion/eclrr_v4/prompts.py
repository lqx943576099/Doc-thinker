"""Strict JSON prompts and prompt-budget planning for ECLRR-v4."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .models import ECLRRConfig, EvidencePackage, Proposal
from .clue_semantics import SYMMETRIC_RELATIONS

RELATION_FAMILIES = (
    "kinship",
    "romantic",
    "friendship",
    "causation",
    "influence",
    "collaboration",
    "dependency",
    "ownership",
    "location",
    "participation",
    "identity",
    "communication",
    "protection",
    "opposition",
    "succession",
    "affiliation",
    "temporal",
    "other_specific",
)

CANONICAL_RELATIONS = (
    "parent_child",
    "siblings",
    "spouses",
    "love",
    "liking",
    "jealousy",
    "hatred",
    "trust",
    "friendship",
    "collaboration",
    "participation",
    "influence",
    "causation",
    "teacher_student",
    "enemies",
    "colleagues",
    "protection",
    "communication",
    "dependency",
    "ownership",
    "succession",
    "identity",
    "affiliation",
    "location",
    "temporal_relation",
    "opposition",
)

GENERATOR_INSTRUCTION = """You are the Closure Relation Generator in ECLRR-v4.
Use only the supplied EvidencePackage objects. Each item is one non-adjacent endpoint
pair selected from a loop-free 3-8 hop backbone chain in the original fact graph.
Your job is to propose a formal closure relation between the item's source and target
when the supplied original chunk quotes support it. The closure relation plus the
existing backbone subpath will form a chord/closed loop, but the graph topology itself
is never evidence for the new relation.

You only propose. You do not approve graph writes, and you must not describe any
proposal as already confirmed or written. Do not use outside knowledge, common sense,
node-name background knowledge, fuzzy edges, promoted edges, legacy synthetic edges,
or unlisted evidence. Use the backbone path only to delimit the endpoints, disambiguate
entities, preserve the closure_backbone_path, and provide evidence that the existing
subpath is real.

Allowed support_mode values:
- explicit_direct: one or more supplied quotes directly state the endpoint relation.
- multi_chunk_composed: multiple supplied quotes jointly support the endpoint relation
  without external knowledge and without relying on mere path transitivity.

Never propose a relation based only on co-occurrence, neighboring paragraphs, shared
events, shared locations, shared organizations, shared middle nodes, entity type match,
or the fact that the path connects the endpoints. If the evidence only says A relates
to B and B relates to C, do not infer A relates to C unless the supplied quotes also
support that endpoint relation as an explicit or composed claim. Keep time, place,
role, condition, negation, and uncertainty limits in the description.

Choose one relation_family and one canonical_relation from the supplied ontologies.
Copy the canonical relation spelling exactly into relation when possible. Generic
values such as related, associated, connected, 联系, 相关, or 某种关系 are forbidden.
Preserve direction from the evidence. Relations listed in symmetric_relation_ontology
must use direction=undirected. Cite every primary path evidence_id and every direct
endpoint evidence_id supplied in required_evidence_refs so the deterministic gate can
write the full evidence chain back into the promoted edge.

Output strict JSON only, with no Markdown or trailing explanation:
{"proposals":[{"review_id":"...","source":"...","target":"...",
"relation":"...","relation_family":"...","direction":"source_to_target|target_to_source|undirected",
"description":"...","support_mode":"explicit_direct|multi_chunk_composed",
"evidence_refs":["ev-..."]}]}
Return {"proposals":[]} when no formal closure relation is justified. For every
proposal, evidence_refs MUST exactly equal that item's required_evidence_refs.
"""

JUDGE_INSTRUCTION = """You are the independent Closure Relation Judge in ECLRR-v4.
Re-read the original EvidencePackage and the Generator Proposal. Judge whether the
supplied chunk quotes support a formal closure relation between the source and target.
The backbone path proves the existing subpath and provides context, but path topology
alone is not evidence for the new endpoint relation. Accept or revise only when the
proposal is supported by explicit_direct evidence or by a complete multi_chunk_composed
argument from the supplied quotes. Reject relation generalization, invented facts,
unsupported direction, missing hop evidence, missing required evidence refs,
co-occurrence-only claims, topology-only claims, and chains that contradict the
proposed relation. Scores are decision scores, not probabilities. Output strict JSON
only, with no Markdown or trailing explanation:
{"decisions":[{"review_id":"...","decision":"accept|revise|reject",
"evidence_coverage":0,"semantic_composability":0,"relation_direction":0,
"uncertainty_calibration":0,"total":0,"reason_codes":[],
"revised_relation":null,"revised_relation_family":null,
"verified_evidence_refs":["ev-..."]}]}
Ranges: evidence_coverage 0-4, semantic_composability 0-3,
relation_direction 0-2, uncertainty_calibration 0-1, total 0-10.
Use revise only when the evidence supports a more precise relation than the Proposal.
When evidence_coverage is 4, verified_evidence_refs MUST exactly equal the supplied
required_evidence_refs. The primary 3-8 hop evidence must cover every hop of the
closure_backbone_path; direct_endpoint_evidence, when present, is additional direct
relation evidence. For semantic_composability, score whether the quoted evidence forms
a coherent, closed, non-contradictory relation claim without relying on transitive
relation algebra. Directional relations must follow the actor-to-affected order in the
evidence; undirected relationships may receive full direction credit when the Proposal
marks direction=undirected. Prefer revise to the closest supplied canonical relation
over a bespoke synonym. A revised relation must exactly match the canonical relation
ontology.
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def generator_prompt(packages: list[EvidencePackage], detail: int = 0) -> str:
    include_alternates = detail == 0
    include_context = detail <= 1
    payload = {
        "relation_family_ontology": list(RELATION_FAMILIES),
        "canonical_relation_ontology": list(CANONICAL_RELATIONS),
        "symmetric_relation_ontology": sorted(SYMMETRIC_RELATIONS),
        "items": [
            package.to_prompt_dict(
                include_alternates=include_alternates,
                include_context=include_context,
            )
            for package in packages
        ],
    }
    if detail >= 2:
        for item in payload["items"]:
            item["supporting_paths"] = []
    return GENERATOR_INSTRUCTION + "\nINPUT=" + _json(payload)


def plan_generator_batches(
    packages: list[EvidencePackage],
    config: ECLRRConfig,
) -> list[tuple[list[EvidencePackage], str]]:
    """Fit batches by dropping optional evidence, never primary evidence."""

    def is_contiguous_slice(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
        width = len(shorter)
        return any(
            longer[index : index + width] == shorter
            for index in range(len(longer) - width + 1)
        )

    longest_paths = sorted(
        (package.review_item.primary_path.nodes for package in packages),
        key=lambda nodes: (-len(nodes), nodes),
    )
    chain_by_review: dict[str, tuple[str, ...]] = {}
    for package in packages:
        nodes = package.review_item.primary_path.nodes
        chain_by_review[package.review_item.review_id] = next(
            (
                candidate
                for candidate in longest_paths
                if is_contiguous_slice(nodes, candidate)
            ),
            nodes,
        )

    batches: list[tuple[list[EvidencePackage], str]] = []
    cursor = 0
    while cursor < len(packages):
        upper = cursor
        chains: set[tuple[str, ...]] = set()
        while upper < len(packages) and upper - cursor < config.max_generator_items:
            chain = chain_by_review[packages[upper].review_item.review_id]
            if chain not in chains and len(chains) >= config.max_generator_chains:
                break
            chains.add(chain)
            upper += 1
        accepted: tuple[list[EvidencePackage], str] | None = None
        while upper > cursor:
            candidate = packages[cursor:upper]
            for detail in range(3):
                prompt = generator_prompt(candidate, detail=detail)
                if len(prompt) <= config.max_prompt_chars:
                    accepted = (candidate, prompt)
                    break
            if accepted:
                break
            upper -= 1
        if accepted is None:
            single = [packages[cursor]]
            prompt = generator_prompt(single, detail=2)
            if len(prompt) > config.max_prompt_chars:
                raise ValueError(
                    f"primary_evidence_exceeds_prompt_budget:{packages[cursor].review_item.review_id}"
                )
            accepted = (single, prompt)
        batches.append(accepted)
        cursor += len(accepted[0])
    return batches


def judge_prompt(package: EvidencePackage, proposal: Proposal, detail: int = 0) -> str:
    payload = {
        "relation_family_ontology": list(RELATION_FAMILIES),
        "canonical_relation_ontology": list(CANONICAL_RELATIONS),
        "symmetric_relation_ontology": sorted(SYMMETRIC_RELATIONS),
        "evidence_package": package.to_prompt_dict(
            include_alternates=detail == 0,
            include_context=detail <= 1,
        ),
        "proposal": asdict(proposal),
    }
    if detail >= 2:
        payload["evidence_package"]["supporting_paths"] = []
    return JUDGE_INSTRUCTION + "\nINPUT=" + _json(payload)


def fit_judge_prompt(
    package: EvidencePackage, proposal: Proposal, config: ECLRRConfig
) -> str:
    for detail in range(3):
        prompt = judge_prompt(package, proposal, detail=detail)
        if len(prompt) <= config.max_prompt_chars:
            return prompt
    raise ValueError(
        f"judge_primary_evidence_exceeds_prompt_budget:{proposal.review_id}"
    )
