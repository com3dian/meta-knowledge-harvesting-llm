import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from rapidfuzz import fuzz
from nltk.tokenize import sent_tokenize
import itertools


def compare_knowledge_graphs(claim_graph, reference_graph):
    """
    Compare claim graph with reference graph to determine entailment relationship.
    
    Args:
        claim_graph: NetworkX graph from claim
        reference_graph: NetworkX graph from reference
    
    Returns:
        str: "Entailment", "Contradiction", or "Unverifiable"
    """
    # Get nodes from both graphs
    claim_nodes = set(claim_graph.nodes())
    reference_nodes = set(reference_graph.nodes())
    
    # Check if all claim nodes exist in reference graph
    nodes_match = claim_nodes.issubset(reference_nodes)
    
    if not nodes_match:
        # If nodes don't match, it's unverifiable
        return "unver"
    
    # If nodes match, check edges with their connected nodes
    claim_edges_with_nodes = set()
    reference_edges_with_nodes = set()
    
    # Extract edges with their node information for claim graph
    for edge in claim_graph.edges():
        source, target = edge
        # Create a tuple that represents the edge with its nodes
        claim_edges_with_nodes.add((source, target))
        # Also add the reverse direction to handle undirected graphs
        claim_edges_with_nodes.add((target, source))
    
    # Extract edges with their node information for reference graph
    for edge in reference_graph.edges():
        source, target = edge
        # Create a tuple that represents the edge with its nodes
        reference_edges_with_nodes.add((source, target))
        # Also add the reverse direction to handle undirected graphs
        reference_edges_with_nodes.add((target, source))
    
    # Check if all claim edges (with their nodes) exist in reference graph
    edges_match = all(edge in reference_edges_with_nodes for edge in claim_edges_with_nodes)
    
    if edges_match:
        # Both nodes and edges match - entailment
        return "entail"
    else:
        # Nodes match but edges don't - contradiction
        return "contra"

def compare_knowledge_graphs_similarity(claim_node_list, reference_node_list, if_plot=False):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = model.encode(claim_node_list, convert_to_tensor=True)
    embeddings2 = model.encode(reference_node_list, convert_to_tensor=True)
    similarity_matrix = np.inner(embeddings1, embeddings2) / (
        np.linalg.norm(embeddings1, axis=1, keepdims=True) * np.linalg.norm(embeddings2, axis=1)
    )

    if if_plot:
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=True, xticklabels=reference_node_list, yticklabels=claim_node_list, cmap="YlGnBu")
        plt.xlabel("List 2")
        plt.ylabel("List 1")
        plt.title("Similarity Matrix Heatmap")
        plt.show()
    
    return similarity_matrix


def get_src_tgt_dict(similarity_matrix, claim_node_list, reference_node_list):
    src_tgt_dict = dict()
    for i, row in enumerate(similarity_matrix):
        # Fit KMeans with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        row_reshaped = row.reshape(-1, 1)
        if len(row_reshaped) <= 1:
            threshold = 0.25 - 0.25*1/len(similarity_matrix)
            continue
        kmeans.fit(row_reshaped)
        centers = np.sort(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)  # Midpoint between the two centers
        threshold = max(threshold, 0.25 - 0.25*1/len(similarity_matrix))

        # Find values above threshold
        anomalies = row[row > threshold]
        src_tgt_dict[claim_node_list[i]] = []
        for index, sim in enumerate(row > threshold):
            if sim:
                src_tgt_dict[claim_node_list[i]].append(reference_node_list[index])

    return src_tgt_dict

def find_best_span(target_chunk, paragraph, max_window_size=5):
    sentences = sent_tokenize(paragraph)
    best_score = -1
    best_span = None

    for window_size in range(1, min(max_window_size, len(sentences)) + 1):
        for i in range(len(sentences) - window_size + 1):
            window = ' '.join(sentences[i:i+window_size])
            score = fuzz.ratio(target_chunk, window)
            if score > best_score:
                best_score = score
                best_span = window

    return best_span, best_score

def get_map_src_tgt_dict(src_tgt_dict, claim_edges):
    combinations = list(itertools.product(src_tgt_dict.keys(), repeat=2))
    ans = dict()
    for key_pair in claim_edges.keys():
        if key_pair[0] not in src_tgt_dict or key_pair[1] not in src_tgt_dict:
            continue
        tgt_list_1 = src_tgt_dict[key_pair[0]]
        tgt_list_2 = src_tgt_dict[key_pair[1]]
        combinations = list(itertools.product(tgt_list_1, tgt_list_2)) + list(itertools.product(tgt_list_2, tgt_list_1))
        ans[key_pair] = combinations
    return ans

def find_best_span_for_all_evidences(reference_edges,
                                     claim_reference_edge_map,
                                     paragraph,
                                     max_window_size=5):
    sentences = sent_tokenize(paragraph)

    best_span_list = []
    for _, reference_edge_list in claim_reference_edge_map.items():
        evidence_edges = set(reference_edge_list) & set(reference_edges.keys())

        for evidence_edge in evidence_edges:
            reference_edge = reference_edges[evidence_edge]

            best_score = -1
            best_span = None

            for window_size in range(1, min(max_window_size, len(sentences)) + 1):
                for i in range(len(sentences) - window_size + 1):
                    window = ' '.join(sentences[i:i+window_size])
                    score = fuzz.ratio(reference_edge[0]["description"], window)
                    if score > best_score:
                        best_score = score
                        best_span = (i, i+window_size)
            best_span_list.append(best_span)
    
    intervals = sorted(set(best_span_list), key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

    evidence_sentences = [' '.join(sentences[merged[i][0]:merged[i][1]]) for i in range(len(merged))]
    return ' '.join(evidence_sentences)