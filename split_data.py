import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

def load_jsonl(file_path):
    """Load JSONL file into list"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    """Save list to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def analyze_split(data, split_name):
    """Analyze entity distribution in a split"""
    entity_counts = Counter()
    total_entities = 0
    
    for item in data:
        for entity in item.get('aligned_entities', []):
            entity_counts[entity['label']] += 1
            total_entities += 1
    
    return entity_counts, total_entities

def main():
    
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'processed' / 'training_data_bio.jsonl'
    output_dir = base_dir / 'data' / 'processed'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SPLITTING DATA INTO TRAIN/VAL/TEST")
    print("=" * 80)
    
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        print("\nPlease run bio_tagging.py first!")
        return
    

    print(f"\nLoading data from: {input_file}")
    data = load_jsonl(input_file)
    print(f"✓ Loaded {len(data)} items")
  
    print("\nSplitting data...")
    
    train_data, temp_data = train_test_split(
        data, 
        test_size=0.3, 
        random_state=42,
        shuffle=True
    )
    
   
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=42,
        shuffle=True
    )
    
    
    print(f"\n{'='*80}")
    print("SPLIT SUMMARY")
    print(f"{'='*80}")
    print(f"Total items: {len(data)}")
    print(f"\nTrain: {len(train_data):>4} items ({len(train_data)/len(data)*100:>5.1f}%)")
    print(f"Val:   {len(val_data):>4} items ({len(val_data)/len(data)*100:>5.1f}%)")
    print(f"Test:  {len(test_data):>4} items ({len(test_data)/len(data)*100:>5.1f}%)")
    
    print(f"\n{'='*80}")
    print("ENTITY DISTRIBUTION PER SPLIT")
    print(f"{'='*80}")
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    all_entity_types = set()
    split_stats = {}
    
    for split_name, split_data in splits.items():
        entity_counts, total = analyze_split(split_data, split_name)
        split_stats[split_name] = (entity_counts, total)
        all_entity_types.update(entity_counts.keys())
   
    print(f"\n{'Entity Type':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 80)
    
    for entity_type in sorted(all_entity_types):
        train_count = split_stats['train'][0].get(entity_type, 0)
        val_count = split_stats['val'][0].get(entity_type, 0)
        test_count = split_stats['test'][0].get(entity_type, 0)
        total = train_count + val_count + test_count
        
        print(f"{entity_type:<25} {train_count:>8} {val_count:>8} {test_count:>8} {total:>8}")
    
    print("-" * 80)
    print(f"{'TOTAL':<25} {split_stats['train'][1]:>8} {split_stats['val'][1]:>8} {split_stats['test'][1]:>8} {sum(s[1] for s in split_stats.values()):>8}")

    print(f"\n{'='*80}")
    print("SAVING SPLITS")
    print(f"{'='*80}")
    
    for split_name, split_data in splits.items():
        output_path = output_dir / f'{split_name}.jsonl'
        save_jsonl(split_data, output_path)
        print(f"✓ Saved {split_name}.jsonl ({len(split_data)} items)")
    
    split_info = {
        'total_items': len(data),
        'train_items': len(train_data),
        'val_items': len(val_data),
        'test_items': len(test_data),
        'train_entities': split_stats['train'][1],
        'val_entities': split_stats['val'][1],
        'test_entities': split_stats['test'][1],
        'entity_distribution': {
            entity_type: {
                'train': split_stats['train'][0].get(entity_type, 0),
                'val': split_stats['val'][0].get(entity_type, 0),
                'test': split_stats['test'][0].get(entity_type, 0)
            }
            for entity_type in all_entity_types
        }
    }
    
    info_path = output_dir / 'split_info.json'
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Saved split_info.json")
    


if __name__ == '__main__':
    main()