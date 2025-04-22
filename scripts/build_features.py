#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified feature extraction script that reduces feature count and improves execution speed.
"""

import os
import sys
import argparse
import logging
import json
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import random

def setup_logger():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def load_story_segments(file_path: str, max_segments: int = 50) -> List[str]:
    """
    Load story segments, limiting to the specified maximum quantity.
    
    Args:
        file_path: Path to the file containing story segments
        max_segments: Maximum number of segments to load
        
    Returns:
        List of story segments
    """
    segments = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by separator
    raw_segments = content.split("\n\n===\n\n")
    
    # Only keep specified number of segments
    for i, segment in enumerate(raw_segments):
        if segment.strip() and i < max_segments:
            segments.append(segment.strip())
        if i >= max_segments:
            break
    
    logger.info(f"Loaded {len(segments)} segments (from {file_path})")
    return segments

def simple_tokenize(text: str) -> List[str]:
    """
    Simple tokenization function, not dependent on NLTK
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    # Replace punctuation with space+punctuation+space to ensure they are separate tokens
    for punct in '.,;:!?()[]{}"\'':
        text = text.replace(punct, ' ' + punct + ' ')
    
    # Split by spaces and filter out empty tokens
    return [token for token in text.lower().split() if token.strip()]

def build_vocabulary(segments: List[str], min_count: int = 2, max_words: int = 1000) -> Dict[str, int]:
    """
    Build vocabulary from story segments with limits on vocabulary size.
    
    Args:
        segments: List of text segments
        min_count: Minimum occurrence count for vocabulary items
        max_words: Maximum vocabulary size
        
    Returns:
        Dictionary mapping words to their frequencies
    """
    word_counts = Counter()
    
    for segment in segments:
        # Tokenize
        words = simple_tokenize(segment.lower())
        word_counts.update(words)
    
    # Filter by minimum count
    vocab = {word: count for word, count in word_counts.items() 
             if count >= min_count}
    
    # If vocabulary is too large, only keep the most common max_words words
    if len(vocab) > max_words:
        vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:max_words])
    
    logger.info(f"Built vocabulary containing {len(vocab)} unique words")
    return vocab

def extract_ngrams(segments: List[str], n: int = 3, max_contexts: int = 1000) -> Dict[Tuple[str, ...], Counter]:
    """
    Extract n-grams from story segments with a limit on context quantity.
    
    Args:
        segments: List of text segments
        n: Size of n-grams to extract
        max_contexts: Maximum number of contexts to retain
        
    Returns:
        Dictionary mapping n-gram contexts to next token distributions
    """
    ngram_counts = defaultdict(Counter)
    
    for segment in segments:
        # Tokenize
        tokens = ['<START>'] + simple_tokenize(segment.lower()) + ['<END>']
        
        # Extract n-grams
        for i in range(len(tokens) - n + 1):
            context = tuple(tokens[i:i+n-1])  # n-1 tokens as context
            next_token = tokens[i+n-1]        # nth token as target
            ngram_counts[context][next_token] += 1
    
    # If too many contexts, only keep the most common max_contexts
    if len(ngram_counts) > max_contexts:
        # Calculate total frequency for each context
        context_freqs = {context: sum(counts.values()) for context, counts in ngram_counts.items()}
        # Sort and truncate to the most common max_contexts
        top_contexts = sorted(context_freqs.items(), key=lambda x: x[1], reverse=True)[:max_contexts]
        top_context_keys = [context for context, _ in top_contexts]
        
        # Only keep these contexts
        ngram_counts = {context: ngram_counts[context] for context in top_context_keys}
    
    logger.info(f"Extracted {len(ngram_counts)} unique {n}-gram contexts")
    return ngram_counts

def extract_basic_entities(segments: List[str], max_entities: int = 20) -> Dict[str, List[str]]:
    """
    Extract basic entities from story segments using simple heuristic methods.
    
    Args:
        segments: List of text segments
        max_entities: Maximum number of entities per type
        
    Returns:
        Dictionary mapping entity types to entity lists
    """
    characters = [
        "Detective", "Inspector", "Officer", "Sergeant", "Doctor", "Professor",
        "Mr. Holmes", "Watson", "Lady", "Gentleman", "Witness", "Victim",
        "Suspect", "Client", "Bartender", "Driver", "Stranger", "Boss"
    ]
    
    locations = [
        "London", "Paris", "New York", "Street", "Alley", "House", "Mansion",
        "Office", "Police Station", "Bar", "Speakeasy", "Hotel", "Restaurant",
        "Apartment", "Warehouse", "Basement", "Attic", "Garden", "Park"
    ]
    
    # Randomly select some characters and locations
    selected_characters = random.sample(characters, min(max_entities, len(characters)))
    selected_locations = random.sample(locations, min(max_entities, len(locations)))
    
    logger.info(f"Extracted {len(selected_characters)} characters and {len(selected_locations)} locations")
    
    return {
        'characters': selected_characters,
        'locations': selected_locations
    }

def extract_simple_contexts(segments: List[str], max_pairs: int = 50) -> List[Dict[str, str]]:
    """
    Extract simple context-continuation pairs for training sequence models.
    
    Args:
        segments: List of text segments
        max_pairs: Maximum number of pairs
        
    Returns:
        List of dictionaries containing context and continuation pairs
    """
    pairs = []
    
    for segment in segments:
        if len(segment) < 200:  # Skip segments that are too short
            continue
        
        # Simply split each segment into first half and second half
        mid = len(segment) // 2
        context = segment[:mid]
        continuation = segment[mid:]
        
        pairs.append({
            'context': context,
            'continuation': continuation
        })
        
        # If we already have enough pairs, stop
        if len(pairs) >= max_pairs:
            break
    
    logger.info(f"Extracted {len(pairs)} context-continuation pairs")
    return pairs

def prepare_simple_choice_data(segments: List[str], max_choices: int = 30) -> List[Dict[str, str]]:
    """
    Prepare simplified training data for choice generation.
    
    Args:
        segments: List of text segments
        max_choices: Maximum number of choice data pairs
        
    Returns:
        List of dictionaries containing contexts and choices
    """
    template_choices = [
        "Go to the bar",
        "Talk to the detective",
        "Search for clues",
        "Follow the suspect",
        "Examine the evidence",
        "Call for backup",
        "Hide behind the door",
        "Confront the witness",
        "Look for a weapon",
        "Wait for the signal"
    ]
    
    choice_data = []
    
    for i, segment in enumerate(segments):
        if len(segment) < 100:  # Skip segments that are too short
            continue
        
        # Use the first 100 characters of the segment as context
        context = segment[:100]
        
        # Generate 3-5 choices
        num_choices = random.randint(3, 5)
        choices = random.sample(template_choices, min(num_choices, len(template_choices)))
        
        choice_data.append({
            'context': context,
            'choices': choices
        })
        
        # Stop once we reach the maximum number
        if len(choice_data) >= max_choices:
            break
    
    logger.info(f"Prepared {len(choice_data)} context-choice pairs")
    return choice_data

def main(args):
    """Main feature extraction function."""
    logger.info(f"Building features with input path: {args.input_dir}")
    
    # Load story segments
    segments_file = os.path.join(args.input_dir, 'processed/training_segments.txt')
    segments = load_story_segments(segments_file, max_segments=args.max_segments)
    
    # Output directory
    output_dir = os.path.join(args.input_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Build vocabulary
    vocab = build_vocabulary(segments, min_count=args.min_count, max_words=args.max_words)
    vocab_file = os.path.join(output_dir, 'vocabulary.json')
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    
    # 2. Extract n-grams for n-gram model
    ngram_data = extract_ngrams(segments, n=args.ngram_size, max_contexts=args.max_contexts)
    ngram_file = os.path.join(output_dir, f'{args.ngram_size}gram_model.pkl')
    with open(ngram_file, 'wb') as f:
        pickle.dump(dict(ngram_data), f)
    
    # 3. Extract entities
    entities = extract_basic_entities(segments, max_entities=args.max_entities)
    entities_file = os.path.join(output_dir, 'entities.json')
    with open(entities_file, 'w', encoding='utf-8') as f:
        json.dump(entities, f, indent=2)
    
    # 4. Extract context-continuation pairs for neural model
    pairs = extract_simple_contexts(segments, max_pairs=args.max_pairs)
    pairs_file = os.path.join(output_dir, 'context_continuations.json')
    with open(pairs_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2)
    
    # 5. Prepare choice data
    choice_data = prepare_simple_choice_data(segments, max_choices=args.max_choices)
    choice_file = os.path.join(output_dir, 'choice_data.json')
    with open(choice_file, 'w', encoding='utf-8') as f:
        json.dump(choice_data, f, indent=2)
    
    logger.info("Feature extraction complete!")
    logger.info(f"Files saved to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from story text data')
    parser.add_argument('--input_dir', type=str, default='data',
                        help='Directory containing input data')
    parser.add_argument('--min_count', type=int, default=2,
                        help='Minimum count for words in vocabulary')
    parser.add_argument('--ngram_size', type=int, default=3,
                        help='Size of n-grams to extract')
    parser.add_argument('--max_segments', type=int, default=50,
                        help='Maximum number of segments to process')
    parser.add_argument('--max_words', type=int, default=1000,
                        help='Maximum number of words in vocabulary')
    parser.add_argument('--max_contexts', type=int, default=1000,
                        help='Maximum number of contexts in n-gram model')
    parser.add_argument('--max_entities', type=int, default=20,
                        help='Maximum number of entities per type')
    parser.add_argument('--max_pairs', type=int, default=50,
                        help='Maximum number of context-continuation pairs')
    parser.add_argument('--max_choices', type=int, default=30,
                        help='Maximum number of context-choice pairs')
    
    args = parser.parse_args()
    main(args)