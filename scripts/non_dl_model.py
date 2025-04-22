#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
N-gram model for text generation in interactive fiction.
This model uses statistical n-gram patterns for text generation.
"""

import os
import pickle
import random
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Ensure required NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class NGramStoryGenerator:
    """
    N-gram based story generator.
    This is our classical ML approach without using neural networks.
    """
    def __init__(self, n: int = 3):
        self.n = n
        self.model = defaultdict(Counter)
        self.start_tokens = Counter()
        self.vocab = set()
        
    def train(self, texts: List[str]) -> None:
        """Train the n-gram model on a collection of texts."""
        for text in texts:
            tokens = ['<START>'] + word_tokenize(text.lower()) + ['<END>']
            self.vocab.update(tokens)
            
            # Track starting n-grams
            start_ngram = tuple(tokens[:self.n-1])
            self.start_tokens[start_ngram] += 1
            
            # Build n-gram model
            for i in range(len(tokens) - self.n + 1):
                current_ngram = tuple(tokens[i:i+self.n-1])
                next_token = tokens[i+self.n-1]
                self.model[current_ngram][next_token] += 1
    
    def _get_next_token(self, current_ngram: Tuple[str, ...]) -> str:
        """Choose the next token based on n-gram probabilities."""
        if current_ngram in self.model:
            candidates = self.model[current_ngram]
            total = sum(candidates.values())
            # Convert counts to probabilities and sample
            token_probs = {token: count/total for token, count in candidates.items()}
            tokens, probs = zip(*token_probs.items())
            return np.random.choice(tokens, p=probs)
        else:
            # Backoff: return random token from vocabulary if n-gram not found
            return random.choice(list(self.vocab - {'<START>', '<END>'}))
    
    def generate(self, context: str, num_words: int = 50) -> str:
        """Generate text continuation using the n-gram model."""
        if context:
            # Start with the last n-1 tokens from context
            tokens = word_tokenize(context.lower())[-self.n+1:]
            current_ngram = tuple(tokens)
        else:
            # Start with a random starting n-gram
            if self.start_tokens:
                start_options, counts = zip(*self.start_tokens.items())
                current_ngram = start_options[np.random.choice(len(start_options), p=[c/sum(counts) for c in counts])]
            else:
                # Fallback if no start tokens are available
                current_ngram = tuple(['<START>'] * (self.n-1))
        
        generated = list(current_ngram)
        
        # Continue generating until we reach num_words or <END>
        for _ in range(num_words):
            next_token = self._get_next_token(current_ngram)
            
            if next_token == '<END>':
                break
                
            generated.append(next_token)
            
            # Update current n-gram
            current_ngram = tuple(generated[-self.n+1:])
        
        # Filter out special tokens and join
        result = ' '.join([token for token in generated if token not in {'<START>'}])
        return result
    
    def generate_choices(self, context: str, num_choices: int = 3) -> List[str]:
        """Generate possible next actions using the n-gram model."""
        choices = []
        
        for _ in range(num_choices):
            # Generate a short continuation that could be a choice
            continuation = self.generate(context, num_words=10)
            words = continuation.split()
            
            # Format as an action (imperative verb phrase)
            if words:
                # Capitalize first word and ensure it's a verb-like beginning
                verb_starters = ["Go", "Talk", "Look", "Find", "Take", "Ask", "Run", "Hide", "Fight", "Use", "Examine", "Search"]
                
                if not words[0].capitalize() in verb_starters:
                    choice = random.choice(verb_starters) + " " + " ".join(words[:5])
                else:
                    choice = words[0].capitalize() + " " + " ".join(words[1:5])
                
                choices.append(choice)
        
        # Ensure uniqueness
        return list(dict.fromkeys(choices))[:num_choices]
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        model_data = {
            'n': self.n,
            'model': dict(self.model),
            'start_tokens': dict(self.start_tokens),
            'vocab': list(self.vocab)
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str) -> 'NGramStoryGenerator':
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(n=model_data['n'])
        model.model = defaultdict(Counter)
        for ngram, counter in model_data['model'].items():
            model.model[ngram] = Counter(counter)
        
        model.start_tokens = Counter(model_data['start_tokens'])
        model.vocab = set(model_data['vocab'])
        
        return model

    def evaluate(self, test_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Evaluate the n-gram model.
        
        Args:
            test_data: List of (context, reference) pairs for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'perplexity': 0.0,
            'diversity': 0.0,
            'response_time': 0.0
        }
        
        # Calculate perplexity
        log_probabilities = []
        generations = []
        import time
        total_time = 0
        
        for context, reference in test_data:
            # Calculate perplexity on reference text
            tokens = word_tokenize(reference.lower())
            context_tokens = word_tokenize(context.lower())[-self.n+1:]
            
            log_prob = 0
            count = 0
            
            for i in range(len(tokens)):
                if i >= self.n - 1:
                    current_ngram = tuple(tokens[i-(self.n-1):i])
                    next_token = tokens[i]
                    
                    if current_ngram in self.model and next_token in self.model[current_ngram]:
                        prob = self.model[current_ngram][next_token] / sum(self.model[current_ngram].values())
                        log_prob += np.log(prob)
                        count += 1
            
            if count > 0:
                log_probabilities.append(log_prob / count)
                
            # Generate text and measure time
            start_time = time.time()
            generated = self.generate(context)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            generations.append(generated)
        
        # Calculate perplexity
        if log_probabilities:
            avg_log_prob = sum(log_probabilities) / len(log_probabilities)
            metrics['perplexity'] = np.exp(-avg_log_prob)
        
        # Calculate diversity (unique words / total words)
        all_words = []
        for text in generations:
            all_words.extend(text.lower().split())
            
        if all_words:
            metrics['diversity'] = len(set(all_words)) / len(all_words)
        
        # Calculate average response time
        if test_data:
            metrics['response_time'] = total_time / len(test_data)
            
        return metrics

# Factory function
def get_model(model_path: Optional[str] = None) -> NGramStoryGenerator:
    """Get an n-gram model instance."""
    if model_path and os.path.exists(model_path):
        return NGramStoryGenerator.load(model_path)
    
    # Create and minimally train a new model
    model = NGramStoryGenerator()
    model.train([
        "You enter a magical forest. The trees whisper ancient secrets.",
        "The detective examines the clues carefully. Something doesn't add up.",
        "Starlight illuminates the alien landscape. Your ship's sensors detect movement.",
        "The ancient tome reveals forgotten knowledge. You begin to understand the mystery."
    ])
    return model

if __name__ == "__main__":
    # Example usage
    model = NGramStoryGenerator()
    
    # Simple training data for testing
    training_texts = [
        "You enter the magical forest and see a glowing mushroom circle.",
        "The ancient castle stands tall on the misty mountain peak.",
        "A suspicious figure in a dark cloak watches you from the shadows.",
        "The spaceship's engines roar to life as you prepare for takeoff.",
        "You find a mysterious letter hidden under an old painting.",
    ]
    model.train(training_texts)
    
    print("Example generation:")
    print(model.generate("You enter the magical"))
    
    print("\nExample choices:")
    for choice in model.generate_choices("You enter the magical"):
        print(f"- {choice}")
    
    # Save and load example
    model.save("models/ngram_model.pkl")
    loaded_model = NGramStoryGenerator.load("models/ngram_model.pkl")
    print("\nLoaded model generation:")
    print(loaded_model.generate("You enter the magical"))