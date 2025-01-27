import os
import random
from collections import defaultdict
import argparse

def get_relation_distribution(edges):
    """Calculate the distribution of relation types in the edge set."""
    relation_counts = defaultdict(int)
    for edge in edges:
        _, relation, _ = edge.split('\t')
        relation_counts[relation] += 1
    return relation_counts

def stratified_split(edges, split_ratio, relation_counts=None):
    """
    Split edges while maintaining relation type proportions.
    
    Parameters:
    - edges: List of edges to split
    - split_ratio: Proportion of edges to select
    - relation_counts: Optional pre-calculated relation counts
    
    Returns:
    - selected_edges: List of selected edges maintaining relation proportions
    - remaining_edges: List of unselected edges
    """
    if relation_counts is None:
        relation_counts = get_relation_distribution(edges)
    
    # Group edges by relation type
    relation_groups = defaultdict(list)
    for edge in edges:
        _, relation, _ = edge.split('\t')
        relation_groups[relation].append(edge)
    
    selected_edges = []
    remaining_edges = []
    
    # For each relation type, select proportional number of edges
    for relation, count in relation_counts.items():
        edges_for_relation = relation_groups[relation]
        n_select = int(len(edges_for_relation) * split_ratio)
        
        # Shuffle edges for this relation
        random.shuffle(edges_for_relation)
        
        # Split edges for this relation
        selected_edges.extend(edges_for_relation[:n_select])
        remaining_edges.extend(edges_for_relation[n_select:])
    
    return selected_edges, remaining_edges

def create_stratified_transductive_splits(all_edges, training_edges, validation_ratio=0.15, test_ratio=0.15):
    """Creates transductive splits while maintaining relation type proportions."""
    # Get training nodes
    training_nodes = set()
    for edge in training_edges:
        node1, _, node2 = edge.split('\t')
        training_nodes.add(node1)
        training_nodes.add(node2)
    
    # Filter edges for transductive setting
    filtered_edges = [edge for edge in all_edges 
                     if edge.split('\t')[0] in training_nodes 
                     and edge.split('\t')[2] in training_nodes]
    non_training_edges = list(set(filtered_edges) - set(training_edges))
    
    # Calculate relation distribution in non-training edges
    relation_dist = get_relation_distribution(non_training_edges)
    
    # Create validation split
    validation_edges, remaining = stratified_split(
        non_training_edges, 
        validation_ratio / (1 - validation_ratio),
        relation_dist
    )
    
    # Create test split
    test_edges, _ = stratified_split(
        remaining,
        test_ratio / (1 - validation_ratio),
        relation_dist
    )
    
    return validation_edges, test_edges

def create_stratified_semi_inductive_splits(all_edges, training_edges, new_nodes_edges, 
                                          validation_ratio=0.15, test_ratio=0.15):
    """Creates semi-inductive splits while maintaining relation type proportions."""
    # Get relation distribution in new_nodes_edges
    relation_dist = get_relation_distribution(new_nodes_edges)
    
    # Split new edges for validation and test
    total_new = len(new_nodes_edges)
    new_validation_edges, remaining = stratified_split(
        new_nodes_edges,
        validation_ratio,
        relation_dist
    )
    new_test_edges, _ = stratified_split(
        remaining,
        test_ratio / (1 - validation_ratio),
        relation_dist
    )
    
    # Get remaining edges (not in training or new_nodes_edges)
    remaining_edges = list(set(all_edges) - set(training_edges) - set(new_nodes_edges))
    
    # Split remaining edges proportionally
    existing_validation_edges, remaining = stratified_split(
        remaining_edges,
        validation_ratio
    )
    existing_test_edges, _ = stratified_split(
        remaining,
        test_ratio / (1 - validation_ratio)
    )
    
    # Combine splits
    validation_edges = existing_validation_edges + new_validation_edges
    test_edges = existing_test_edges + new_test_edges
    
    return validation_edges, test_edges

def split_patient_kg(biomedical_kg_path, patient_kg_path, output_dir):
    """
    Split both biomedical and patient-integrated KGs while maintaining relation distributions.
    This implementation follows a specific split structure for fine-tuning:
    - Foundation model training: 80% training, 10% validation, 10% test of biomedical KG
    - Fine-tuning: 
        * All foundation model training triples (8T/10)
        * 90% of patient-specific triples for training (9P/10)
        * Original validation triples (T/10) plus 10% patient-specific triples (P/10) for validation
    
    Parameters:
    - biomedical_kg_path: Path to the biomedical KG file
    - patient_kg_path: Path to the patient-integrated KG file
    - output_dir: Directory to save the splits
    """
    # Read KGs
    with open(biomedical_kg_path, 'r') as f:
        biomedical_edges = [line.strip() for line in f]
    with open(patient_kg_path, 'r') as f:
        patient_edges = [line.strip() for line in f]
    
    # Verify patient KG is a superset of biomedical KG
    if not set(biomedical_edges).issubset(set(patient_edges)):
        raise ValueError("Biomedical KG must be a subset of patient KG")
    
    # Split biomedical KG for foundation model (T total triples)
    bio_train_edges, remaining = stratified_split(biomedical_edges, 0.8)  # 8T/10
    bio_valid_edges, bio_test_edges = stratified_split(remaining, 0.5)    # T/10 each
    
    # Identify patient-specific edges (P total triples)
    patient_specific_edges = list(set(patient_edges) - set(biomedical_edges))
    
    # Split patient-specific edges
    patient_train_edges, patient_valid_edges = stratified_split(
        patient_specific_edges, 0.9  # 9P/10 for training, P/10 for validation
    )
    
    # Create directories
    os.makedirs(os.path.join(output_dir, 'foundation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'fine_tuning'), exist_ok=True)
    
    # Save foundation model splits
    write_edges(bio_train_edges, os.path.join(output_dir, 'foundation/train.txt'))
    write_edges(bio_valid_edges, os.path.join(output_dir, 'foundation/valid.txt'))
    write_edges(bio_test_edges, os.path.join(output_dir, 'foundation/test.txt'))
    
    # Analyze relation distributions
    def get_relation_stats(edges):
        """Get counts and percentages for each relation type."""
        relation_counts = defaultdict(int)
        total = len(edges)
        for edge in edges:
            _, relation, _ = edge.split('\t')
            relation_counts[relation] += 1
        return {rel: (count, count/total) for rel, count in relation_counts.items()}

    # Get relation distributions for both edge sets
    bio_train_stats = get_relation_stats(bio_train_edges)
    patient_train_stats = get_relation_stats(patient_train_edges)
    
    # Combine edges while maintaining relation proportions
    def combine_edges_with_stratification(edges1, edges2, stats1, stats2):
        """Combine two sets of edges while maintaining relation distributions."""
        # Group edges by relation
        relations1 = defaultdict(list)
        relations2 = defaultdict(list)
        
        for edge in edges1:
            _, rel, _ = edge.split('\t')
            relations1[rel].append(edge)
        for edge in edges2:
            _, rel, _ = edge.split('\t')
            relations2[rel].append(edge)
        
        # Combine while maintaining proportions
        combined = []
        all_relations = set(relations1.keys()) | set(relations2.keys())
        
        for rel in all_relations:
            rel_edges1 = relations1[rel]
            rel_edges2 = relations2[rel]
            
            # Keep all edges from both sets but ensure proper mixing
            random.shuffle(rel_edges1)
            random.shuffle(rel_edges2)
            combined.extend(rel_edges1)
            combined.extend(rel_edges2)
        
        return combined

    # Create stratified combinations for fine-tuning
    fine_tuning_train = combine_edges_with_stratification(
        bio_train_edges, patient_train_edges,
        bio_train_stats, patient_train_stats
    )
    
    fine_tuning_valid = combine_edges_with_stratification(
        bio_valid_edges, patient_valid_edges,
        get_relation_stats(bio_valid_edges),
        get_relation_stats(patient_valid_edges)
    )
    
    # Save fine-tuning splits
    write_edges(fine_tuning_train, os.path.join(output_dir, 'fine_tuning/train.txt'))
    write_edges(fine_tuning_valid, os.path.join(output_dir, 'fine_tuning/valid.txt'))
    write_edges(bio_test_edges, os.path.join(output_dir, 'fine_tuning/test.txt'))
    
    # Print detailed relation distribution statistics
    print("\nRelation Distribution Statistics:")
    print("\nBiomedical Training Set:")
    for rel, (count, percent) in bio_train_stats.items():
        print(f"  {rel}: {count} edges ({percent*100:.2f}%)")
    
    print("\nPatient-specific Training Set:")
    for rel, (count, percent) in patient_train_stats.items():
        print(f"  {rel}: {count} edges ({percent*100:.2f}%)")
    
    print("\nCombined Fine-tuning Training Set:")
    final_stats = get_relation_stats(fine_tuning_train)
    for rel, (count, percent) in final_stats.items():
        print(f"  {rel}: {count} edges ({percent*100:.2f}%)")
    
    # Print statistics
    print("\nSplit Statistics:")
    print(f"Foundation Model:")
    print(f"  Training: {len(bio_train_edges)} edges")
    print(f"  Validation: {len(bio_valid_edges)} edges")
    print(f"  Test: {len(bio_test_edges)} edges")
    print(f"\nPatient-specific edges:")
    print(f"  Total: {len(patient_specific_edges)} edges")
    print(f"  Training: {len(patient_train_edges)} edges")
    print(f"  Validation: {len(patient_valid_edges)} edges")
    print(f"\nFine-tuning splits:")
    print(f"  Training: {len(fine_tuning_train)} edges")
    print(f"  Validation: {len(fine_tuning_valid)} edges")
    print(f"  Test: {len(bio_test_edges)} edges")

def write_edges(edges, filepath):
    """Write edges to a file."""
    with open(filepath, 'w') as f:
        for edge in edges:
            f.write(f"{edge}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split knowledge graphs for personalized drug repurposing")
    parser.add_argument('--biomedical_kg', required=True, help="biomedical_kg.txt")
    parser.add_argument('--patient_kg', required=True, help="patient_KGs_G20/patient_1272434_kg.txt")
    parser.add_argument('--output_dir', required=True, help="outputs")
    
    args = parser.parse_args()
    split_patient_kg(args.biomedical_kg, args.patient_kg, args.output_dir)