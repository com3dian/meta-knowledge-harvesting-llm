"""
Copied from lightRAG repo
"""

from __future__ import annotations

import asyncio
import traceback
import json
import re
import os
from typing import Any, AsyncIterator, Callable
import networkx
from collections import Counter, defaultdict
from datetime import datetime

from utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    normalize_extracted_info,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    use_llm_func_with_cache,
    save_knowledge_graph_to_pickle,
)
from base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from prompt import GRAPH_FIELD_SEP, PROMPTS
from cheatsheet import CHEATSHEETS
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


def chunking_by_token_size(
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = decode_tokens_by_tiktoken(
                            _tokens[start : start + max_token_size],
                            model_name=tiktoken_model,
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = decode_tokens_by_tiktoken(
                tokens[start : start + max_token_size], model_name=tiktoken_model
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["summary_to_max_tokens"]

    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")

    # Use LLM function with cache
    summary = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        max_tokens=summary_max_tokens,
        cache_type="extract",
    )
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None

    # Clean and validate entity name
    entity_name = clean_str(record_attributes[1]).strip('"')
    if not entity_name.strip():
        logger.warning(
            f"Entity extraction error: empty entity name in: {record_attributes}"
        )
        return None

    # Normalize entity name
    entity_name = normalize_extracted_info(entity_name, is_entity=True)

    # Clean and validate entity type
    entity_type = clean_str(record_attributes[2]).strip('"')
    if not entity_type.strip() or entity_type.startswith('("'):
        logger.warning(
            f"Entity extraction error: invalid entity type in: {record_attributes}"
        )
        return None

    # Clean and validate description
    entity_description = clean_str(record_attributes[3])
    entity_description = normalize_extracted_info(entity_description)

    if not entity_description.strip():
        logger.warning(
            f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
        )
        return None

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=chunk_key,
        file_path=file_path,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1])
    target = clean_str(record_attributes[2])

    # Normalize source and target entity names
    source = normalize_extracted_info(source, is_entity=True)
    target = normalize_extracted_info(target, is_entity=True)

    edge_description = clean_str(record_attributes[3])
    edge_description = normalize_extracted_info(edge_description)

    edge_keywords = clean_str(record_attributes[4]).strip('"').strip("'")
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1].strip('"').strip("'"))
        if is_float_regex(record_attributes[-1])
        else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        file_path=file_path,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
):
    """Get existing nodes from knowledge graph use name, if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_file_paths.extend(
            split_string_by_multi_markers(already_node["file_path"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    file_path = GRAPH_FIELD_SEP.join(
        set([dp["file_path"] for dp in nodes_data] + already_file_paths)
    )

    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    num_fragment = description.count(GRAPH_FIELD_SEP) + 1
    num_new_fragment = len(set([dp["description"] for dp in nodes_data]))

    if num_fragment > 1:
        if num_fragment >= force_llm_summary_on_merge:
            status_message = f"LLM merge N: {entity_name} | {num_new_fragment}+{num_fragment-num_new_fragment}"
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            description = await _handle_entity_relation_summary(
                entity_name,
                description,
                global_config,
                pipeline_status,
                pipeline_status_lock,
                llm_response_cache,
            )
        else:
            status_message = f"Merge N: {entity_name} | {num_new_fragment}+{num_fragment-num_new_fragment}"
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)

    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        file_path=file_path,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
):
    """
    Get existing edges from knowledge graph use source and target, if exists, merge data, else create, then upsert.
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 0.0 if missing
            already_weights.append(already_edge.get("weight", 0.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    split_string_by_multi_markers(
                        already_edge["source_id"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get file_path with empty string default if missing or None
            if already_edge.get("file_path") is not None:
                already_file_paths.extend(
                    split_string_by_multi_markers(
                        already_edge["file_path"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.append(already_edge["description"])

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    # Process edges_data with None checks
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(
            set(
                [dp["description"] for dp in edges_data if dp.get("description")]
                + already_description
            )
        )
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(
            set(
                [dp["keywords"] for dp in edges_data if dp.get("keywords")]
                + already_keywords
            )
        )
    )
    source_id = GRAPH_FIELD_SEP.join(
        set(
            [dp["source_id"] for dp in edges_data if dp.get("source_id")]
            + already_source_ids
        )
    )
    file_path = GRAPH_FIELD_SEP.join(
        set(
            [dp["file_path"] for dp in edges_data if dp.get("file_path")]
            + already_file_paths
        )
    )

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "entity_id": need_insert_id,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": "UNKNOWN",
                    "file_path": file_path,
                },
            )

    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    num_fragment = description.count(GRAPH_FIELD_SEP) + 1
    num_new_fragment = len(
        set([dp["description"] for dp in edges_data if dp.get("description")])
    )

    if num_fragment > 1:
        if num_fragment >= force_llm_summary_on_merge:
            status_message = f"LLM merge E: {src_id} - {tgt_id} | {num_new_fragment}+{num_fragment-num_new_fragment}"
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            description = await _handle_entity_relation_summary(
                f"({src_id}, {tgt_id})",
                description,
                global_config,
                pipeline_status,
                pipeline_status_lock,
                llm_response_cache,
            )
        else:
            status_message = f"Merge E: {src_id} - {tgt_id} | {num_new_fragment}+{num_fragment-num_new_fragment}"
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)

    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    entity_cheatsheet_dict: dict,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    output_prefix: str = "",
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    cheatsheet_knowledge_graph_inst: networkx.Graph = None,
    write_result_to_txt: bool = False,
) -> str | None:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = CHEATSHEETS["entity_extraction"]
    special_interest = ""
    for entity_type, entity_description in entity_cheatsheet_dict.items():
        special_interest += f"{entity_type}: {entity_description}\n"

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        special_interest=special_interest,
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    fill_nightly_prompt = CHEATSHEETS["entity_fill_nightly_extraction"]
    # we can use the same context_base here

    continue_prompt = CHEATSHEETS["entity_continue_extraction"].format(**context_base)
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    processed_chunks = 0
    total_chunks = len(ordered_chunks)
    total_entities_count = 0
    total_relations_count = 0

    # Get lock manager from shared storage
    from database_impl.shared_storage import get_graph_db_lock

    graph_db_lock = get_graph_db_lock(enable_logging=False)

    # Use the global use_llm_func_with_cache function from utils.py

    async def _process_extraction_result(
        result: str, chunk_key: str, file_path: str = "unknown_source"
    ):
        """
        Process a single extraction result (either initial or gleaning)
        Args:
            result (str): The extraction result to process
            chunk_key (str): The chunk key for source tracking
            file_path (str): The file path for citation
        Returns:
            tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
        """
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        records = split_string_by_multi_markers(
            result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, file_path
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key, file_path
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        return maybe_nodes, maybe_edges

    async def _nightly_inference_on_initial_result(
        maybe_nodes: dict[str, list[dict]],
        maybe_edges: dict[tuple[str, str], list[dict]],
        entity_types: list[str],
        chunk_key: str,
        cheatsheet_knowledge_graph_inst: networkx.Graph,
        file_path: str = "unknown_source",
    ):
        """
        Process a single extraction result (either initial or gleaning) by using domain knowledge to find nightly
        inference nodes and relationships
        Args:
            maybe_nodes (dict[str, list[dict]]): The extracted entities
            maybe_edges (dict[tuple[str, str], list[dict]]): The extracted relationships
            chunk_key (str): The chunk key for source tracking
            file_path (str): The file path for citation
        Returns:
            nightly_nodes (dict[str, list[dict]]): The nightly inference entities
            nightly_edges (dict[tuple[str, str], list[dict]]): The nightly inference relationships
        """
        nightly_nodes = defaultdict(list)
        nightly_edges = defaultdict(list)

        # Skip nightly inference if no cheatsheet knowledge graph is provided
        if cheatsheet_knowledge_graph_inst is None:
            return nightly_nodes, nightly_edges

        # Use domain knowledge to find nightly inference nodes and relationships
        for entity_name, entity_info_dict_list in maybe_nodes.items():
            for entity_info_dict in entity_info_dict_list:
                # Extract entity information
                entity_type = entity_info_dict['entity_type']
                description = entity_info_dict['description']
                if entity_type not in entity_types:
                    continue
                neighbors = list(cheatsheet_knowledge_graph_inst.neighbors(entity_type))
                for neighbor in neighbors:
                    if neighbor not in entity_types:
                        continue

                    # Add nightly inference node
                    nightly_nodes[neighbor].append(
                        {
                            "entity_name": "<Nightly Entity Name>",
                            "entity_type": neighbor,
                            "description": "<Nightly inference>",
                            "reference": entity_info_dict['description'],
                            "source_id": chunk_key,
                            "file_path": file_path,
                        }
                    )

                    nightly_edge_data = cheatsheet_knowledge_graph_inst.get_edge_data(entity_type, neighbor)
                    nightly_edges[(entity_name, "<Nightly Entity Name>")].append(
                        dict(
                            src_id=entity_name,
                            tgt_id="<Nightly Entity Name>",
                            description="<Nightly Inference>",
                            keywords="<Nightly inference>",
                            reference=entity_info_dict['description'],
                            source_id=chunk_key,
                            file_path=file_path,
                        )
                    )

        return nightly_nodes, nightly_edges
    
    async def nightly_kg_to_text(nightly_nodes, nightly_edges):
        """
        Convert nightly nodes and edges into formatted text strings using tuple and record delimiters
        Args:
            nightly_nodes (dict[str, list[dict]]): The nightly inference entities
            nightly_edges (dict[tuple[str, str], list[dict]]): The nightly inference relationships
        Returns:
            str: Formatted text containing nodes and edges in the specified format
        """
        nightly_nodes_text = []
        nightly_edges_text = []

        # Process nodes
        for entity_name, entity_info_list in nightly_nodes.items():
            for entity_info in entity_info_list:
                node_str = (
                    f'("entity"'
                    f'{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}'
                    f'{entity_info["entity_name"]}'
                    f'{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}'
                    f'{entity_info["entity_type"]}'
                    f'{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}'
                    f'{entity_info["description"]}'
                    f'{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}'
                    f'{entity_info["source"]}'
                    f'{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}'
                    f'{entity_info["reference"]})'
                )
                nightly_nodes_text.append(node_str)

        # Process edges
        for edge_key, edge_info_list in nightly_edges.items():
            for edge_info in edge_info_list:
                edge_str = (
                    f'("relationship"'
                    f'{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}'
                    f'{edge_info["src_id"]}'
                    f'{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}'
                    f'{edge_info["tgt_id"]}'
                    f'{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}'
                    f'{edge_info["description"]}'
                    f'{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}'
                    f'{edge_info["keywords"]}'
                    f'{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}'
                    f'{edge_info["reference"]})'
                )
                nightly_edges_text.append(edge_str)

        # Combine nodes and edges with record delimiters
        all_records = nightly_nodes_text + nightly_edges_text
        return f'{PROMPTS["DEFAULT_RECORD_DELIMITER"]}'.join(all_records)

    async def fill_nightly_nodes_edges_with_llm(
            nightly_nodes: dict[str, list[dict]],
            nightly_edges: dict[tuple[str, str], list[dict]],
            input_text: str,
            chunk_key: str,
            file_path: str = "unknown_source",
        ):
        """
        Fill nightly nodes and edges with LLM
        Args:
            nightly_nodes (dict[str, list[dict]]): The nightly inference entities
            nightly_edges (dict[tuple[str, str], list[dict]]): The nightly inference relationships
            source_texts (str): The source texts for LLM
        Returns:
            maybe_nodes (dict[str, list[dict]]): The filled nightly inference entities
            maybe_edges (dict[tuple[str, str], list[dict]]): The filled nightly inference relationships
        """
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        hint_prompt = fill_nightly_prompt.format(
            **context_base, input_text="{input_text}", nightly_entities_and_relationships="{nightly_entities_and_relationships}"
        ).format(**context_base,
                 input_text=input_text,
                 nightly_entities_and_relationships=nightly_kg_to_text(nightly_nodes, nightly_edges))

        final_result = await use_llm_func_with_cache(
            hint_prompt,
            use_llm_func,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
        )

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

        # Process initial extraction with file path
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result, chunk_key, file_path
        )

        return maybe_nodes, maybe_edges

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")

        # Get initial extraction
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await use_llm_func_with_cache(
            hint_prompt,
            use_llm_func,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
        )

        """
        process initial extraction result @Lu

        """

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

        # Process initial extraction with file path
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result, chunk_key, file_path
        )

        nightly_nodes, nightly_edges = await _nightly_inference_on_initial_result(
            maybe_nodes, maybe_edges, chunk_key, file_path, cheatsheet_knowledge_graph_inst
        )

        glean_nodes, glean_edges = await fill_nightly_nodes_edges_with_llm(nightly_nodes, nightly_edges, content, chunk_key, file_path)
        # Merge results - only add entities and edges with new names
        for entity_name, entities in glean_nodes.items():
            if entity_name not in maybe_nodes:  # Only accetp entities with new name in gleaning stage
                maybe_nodes[entity_name].extend(entities)
            
        for edge_key, edges in glean_edges.items():
            if edge_key not in maybe_edges:  # Only accetp edges with new name in gleaning stage
                maybe_edges[edge_key].extend(edges)

        # Process additional gleaning results
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func_with_cache(
                continue_prompt,
                use_llm_func,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
            )

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)

            # Process gleaning result separately with file path
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result, chunk_key, file_path
            )

            nightly_nodes, nightly_edges = await _nightly_inference_on_initial_result(
                glean_nodes, glean_edges, chunk_key, file_path, cheatsheet_knowledge_graph_inst
            )

            glean_nodes = glean_nodes | nightly_nodes
            glean_edges = glean_edges | nightly_edges

            # Merge results - only add entities and edges with new names
            for entity_name, entities in glean_nodes.items():
                if (
                    entity_name not in maybe_nodes
                ):  # Only accetp entities with new name in gleaning stage
                    maybe_nodes[entity_name].extend(entities)
            for edge_key, edges in glean_edges.items():
                if (
                    edge_key not in maybe_edges
                ):  # Only accetp edges with new name in gleaning stage
                    maybe_edges[edge_key].extend(edges)

            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func_with_cache(
                if_loop_prompt,
                use_llm_func,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chk {processed_chunks}/{total_chunks}: extracted {entities_count} Ent + {relations_count} Rel"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Handle all chunks in parallel and collect results
    tasks = [_process_single_content(c) for c in ordered_chunks]
    chunk_results = await asyncio.gather(*tasks)

    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    for maybe_nodes, maybe_edges in chunk_results:
        # Collect nodes
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)

    # Centralized processing of all nodes and edges
    entities_data = []
    relationships_data = []

    # Use graph database lock to ensure atomic merges and updates
    if write_result_to_txt:
        output_file_name = save_knowledge_graph_to_pickle(all_nodes,
                                                          all_edges,
                                                          output_prefix,
                                                          write_result_to_txt=write_result_to_txt)
        
    async with graph_db_lock:
        # Process and update all entities at once
        for entity_name, entities in all_nodes.items():
            entity_data = await _merge_nodes_then_upsert(
                entity_name,
                entities,
                knowledge_graph_inst,
                global_config,
                pipeline_status,
                pipeline_status_lock,
                llm_response_cache,
            )
            entities_data.append(entity_data)

        # Process and update all relationships at once
        for edge_key, edges in all_edges.items():
            edge_data = await _merge_edges_then_upsert(
                edge_key[0],
                edge_key[1],
                edges,
                knowledge_graph_inst,
                global_config,
                pipeline_status,
                pipeline_status_lock,
                llm_response_cache,
            )
            relationships_data.append(edge_data)

        # Update vector databases with all collected data
        if entity_vdb is not None and entities_data:
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "entity_name": dp["entity_name"],
                    "entity_type": dp["entity_type"],
                    "content": f"{dp['entity_name']}\n{dp['description']}",
                    "source_id": dp["source_id"],
                    "file_path": dp.get("file_path", "unknown_source"),
                }
                for dp in entities_data
            }
            await entity_vdb.upsert(data_for_vdb)

        if relationships_vdb is not None and relationships_data:
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "keywords": dp["keywords"],
                    "content": f"{dp['src_id']}\t{dp['tgt_id']}\n{dp['keywords']}\n{dp['description']}",
                    "source_id": dp["source_id"],
                    "file_path": dp.get("file_path", "unknown_source"),
                }
                for dp in relationships_data
            }
            await relationships_vdb.upsert(data_for_vdb)

    # Update total counts
    total_entities_count = len(entities_data)
    total_relations_count = len(relationships_data)

    log_message = f"Extracted {total_entities_count} entities + {total_relations_count} relationships (total)"
    logger.info(log_message)
    if pipeline_status is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)
    
    return output_file_name

async def _link_relationships_across_entities(
    all_nodes: dict[str, list[dict]],
    all_edges: dict[tuple[str, str], list[dict]],
    embedding_func: Callable,
    cheatsheet_knowledge_graph_inst: networkx.Graph = None,
):
    """
    Link relationships across entities
    """
    linkage_edges = []
    linkage_nodes = []

    for edge_key, edge_list in all_edges.items():
        for edge in edge_list:
            similarity_score = 0
            similarity_src_node_id = None
            similarity_tgt_node_id = None
            if edge["src_id"] in all_nodes and edge["tgt_id"] in all_nodes:
                src_entity_type = all_nodes[edge["src_id"]][0]["entity_type"]
                tgt_entity_type = all_nodes[edge["tgt_id"]][0]["entity_type"]
                if not cheatsheet_knowledge_graph_inst.has_edge(src_entity_type, tgt_entity_type):
                    continue

            if edge["src_id"] in all_nodes:
                node_id = 0
                for node in all_nodes[edge["src_id"]]:
                    node_description = node["description"]
                    edge_description = edge['description']
                    node_embedding = await embedding_func([node_description])
                    edge_embedding = await embedding_func([edge_description])

                    # Calculate cosine similarity between node and edge embeddings
                    similarity = cosine_similarity(node_embedding, edge_embedding)
                    if similarity > similarity_score:
                        similarity_score = similarity
                        similarity_src_node_id = node_id

                    node_id += 1
            
            similarity_score = 0
            if edge["tgt_id"] in all_nodes:
                node_id = 0
                for node in all_nodes[edge["tgt_id"]]:
                    node_description = node["description"]
                    edge_description = edge["description"]
                    node_embedding = await embedding_func([node_description])
                    edge_embedding = await embedding_func([edge_description])

                    # Calculate cosine similarity between node and edge embeddings
                    similarity = cosine_similarity(node_embedding, edge_embedding)
                    if similarity > similarity_score:
                        similarity_score = similarity
                        similarity_tgt_node_id = node_id
                        
                    node_id += 1

            if (similarity_src_node_id is not None) and (similarity_tgt_node_id is not None):
                linkage_edge = dict(
                    src_id=edge["src_id"] + "_" + str(similarity_src_node_id),
                    tgt_id=edge["tgt_id"] + "_" + str(similarity_tgt_node_id),
                    description=edge["description"],
                    type=(src_entity_type, tgt_entity_type),
                    keywords=edge["keywords"],
                    source_id=edge["source_id"],
                    file_path=edge["file_path"],
                )
                linkage_edges.append(linkage_edge)

                linkage_node_src = dict(
                    entity_name=edge["src_id"] + "_" + str(similarity_src_node_id),
                    entity_type=all_nodes[edge["src_id"]][similarity_src_node_id]["entity_type"],
                    description=all_nodes[edge["src_id"]][similarity_src_node_id]["description"],
                    source_id=edge["source_id"],
                    file_path=edge["file_path"],
                )
                linkage_node_tgt = dict(
                    entity_name=edge["tgt_id"] + "_" + str(similarity_tgt_node_id),    
                    entity_type=all_nodes[edge["tgt_id"]][similarity_tgt_node_id]["entity_type"],
                    description=all_nodes[edge["tgt_id"]][similarity_tgt_node_id]["description"],
                    source_id=edge["source_id"],
                    file_path=edge["file_path"],
                )
                linkage_nodes.append(linkage_node_src)
                linkage_nodes.append(linkage_node_tgt)

    return linkage_nodes, linkage_edges    

async def build_structured_knowledge_graph(
    all_nodes: dict[str, list[dict]],
    all_edges: dict[tuple[str, str], list[dict]],
    embedding_func: Callable,
):
    """
    """
    linkage_nodes, linkage_edges = await _link_relationships_across_entities(
        all_nodes, all_edges, embedding_func)
    networkx_graph = networkx.Graph()
    networkx_graph.add_nodes_from(linkage_nodes)
    networkx_graph.add_edges_from(linkage_edges)
    return networkx_graph