"""
Evaluation script for comparing the three models:
1. Naive model (baseline)
2. Non-deep learning model (n-gram)
3. Deep learning model (transformer)

This script implements various metrics to evaluate text generation quality.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
import pickle
import json
import re

# Add parent directory to path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure we have the required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Import the models
try:
    # Import from the naive_model module
    from scripts.naive_model import NaiveStoryGenerator, get_naive_model
except ImportError as e:
    print(f"Error importing naive model: {e}")
    # Define a fallback naive model
    class NaiveStoryGenerator:
        def generate(self, context, num_words=50):
            return f"[Naive model output for: {context[:20]}...]"
        
        def generate_choices(self, context, num_choices=3):
            return [f"Naive choice {i}" for i in range(num_choices)]
    
    def get_naive_model(model_path=None):
        return NaiveStoryGenerator()

try:
    # Import from the non_dl_model module
    from scripts.non_dl_model import NGramStoryGenerator, get_model as get_ngram_model
except ImportError as e:
    print(f"Error importing n-gram model: {e}")
    # Define a fallback n-gram model
    class NGramStoryGenerator:
        def generate(self, context, num_words=50):
            return f"[N-gram model output for: {context[:20]}...]"
        
        def generate_choices(self, context, num_choices=3):
            return [f"N-gram choice {i}" for i in range(num_choices)]
    
    def get_ngram_model(model_path=None):
        return NGramStoryGenerator()

try:
    # Import from the deep_learning_model module
    from scripts.deep_learning_model import TransformerStoryGenerator, get_transformer_model
except ImportError as e:
    print(f"Error importing transformer model: {e}")
    # Define a fallback transformer model
    class TransformerStoryGenerator:
        def generate(self, context, num_words=50):
            return f"[Transformer model output for: {context[:20]}...]"
        
        def generate_choices(self, context, num_choices=3):
            return [f"Transformer choice {i}" for i in range(num_choices)]
    
    def get_transformer_model(model_path=None):
        return TransformerStoryGenerator()


class ModelEvaluator:
    """
    Class to evaluate and compare different story generation models.
    """
    def __init__(self, test_data_path=None, reference_texts=None):
        """
        Initialize the evaluator with test data or reference texts.
        
        Args:
            test_data_path (str): Path to test data JSON file with context-reference pairs
            reference_texts (list): List of reference texts for comparison
        """
        self.test_data = []
        if test_data_path and os.path.exists(test_data_path):
            try:
                with open(test_data_path, 'r', encoding='utf-8') as f:
                    self.test_data = json.load(f)
                print(f"Loaded {len(self.test_data)} test examples from {test_data_path}")
            except Exception as e:
                print(f"Error loading test data: {e}")
        
        self.reference_texts = reference_texts if reference_texts else []
        self.models = {}
        self.results = {}
        
    def add_model(self, name, model):
        """Add a model to be evaluated."""
        self.models[name] = model
        print(f"Added model: {name}")
        
    def _calculate_perplexity(self, model, text):
        """
        Calculate perplexity for a given model and text.
        Lower perplexity means better prediction of the text.
        """
        # For NGramStoryGenerator, use its built-in evaluation
        if isinstance(model, NGramStoryGenerator) and hasattr(model, 'evaluate'):
            try:
                # Create a mock test set with a single example
                mock_test = [(text[:len(text)//2], text[len(text)//2:])]
                metrics = model.evaluate(mock_test)
                return metrics.get('perplexity', float('inf'))
            except Exception as e:
                print(f"Error using n-gram model's perplexity: {e}")
        
        # For Transformer model, use token probabilities
        if hasattr(model, 'model') and hasattr(model, 'tokenizer'):
            try:
                import torch
                
                # Tokenize the text
                inputs = model.tokenizer(text, return_tensors="pt")
                
                # Move to the same device as the model
                if hasattr(model, 'device'):
                    device = model.device
                else:
                    device = next(model.model.parameters()).device
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get model output with loss calculation enabled
                with torch.no_grad():
                    outputs = model.model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
                
                loss = outputs.loss
                
                # Perplexity is exp(loss)
                return torch.exp(loss).item()
            except Exception as e:
                print(f"Error calculating transformer perplexity: {e}")
        
        # Generic fallback: simple n-gram perplexity
        try:
            # Tokenize the text
            tokens = nltk.word_tokenize(text.lower())
            n = 3  # Use trigrams
            
            # Count n-grams in the text
            text_ngrams = list(ngrams(tokens, n))
            ngram_counts = Counter(text_ngrams)
            
            # Calculate log probability
            log_prob = 0
            valid_ngrams = 0
            
            for i in range(len(tokens) - n + 1):
                current_ngram = tuple(tokens[i:i+n])
                if current_ngram in ngram_counts:
                    prob = ngram_counts[current_ngram] / len(text_ngrams)
                    log_prob += np.log(prob)
                    valid_ngrams += 1
            
            if valid_ngrams > 0:
                perplexity = np.exp(-log_prob / valid_ngrams)
                return perplexity
            else:
                return float('inf')
        except Exception as e:
            print(f"Error calculating fallback perplexity: {e}")
            return float('inf')
    
    def _calculate_bleu(self, generated_text, reference_texts):
        """
        Calculate BLEU score comparing generated text against reference texts.
        Higher score means better alignment with reference texts.
        """
        if not reference_texts:
            return 0.0
            
        try:
            # Tokenize texts
            generated_tokens = nltk.word_tokenize(generated_text.lower())
            reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference_texts]
            
            # Calculate BLEU with smoothing
            smoothie = SmoothingFunction().method1
            weights = [(1.0,), (0.5, 0.5), (0.33, 0.33, 0.33), (0.25, 0.25, 0.25, 0.25)]
            bleu_scores = []
            
            for weight in weights:
                score = sentence_bleu(reference_tokens, generated_tokens, 
                                    weights=weight, smoothing_function=smoothie)
                bleu_scores.append(score)
                
            # Return average of BLEU-1, BLEU-2, BLEU-3, and BLEU-4
            return np.mean(bleu_scores)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def _calculate_vocabulary_diversity(self, text):
        """
        Calculate vocabulary diversity (unique words / total words).
        Higher value means more diverse vocabulary.
        """
        try:
            tokens = nltk.word_tokenize(text.lower())
            if not tokens:
                return 0.0
                
            unique_tokens = set(tokens)
            return len(unique_tokens) / len(tokens)
        except Exception as e:
            print(f"Error calculating vocabulary diversity: {e}")
            return 0.0
    
    def _measure_response_time(self, model, prompt, n_trials=3):
        """
        Measure average response time for model generation.
        Lower is better.
        """
        times = []
        for _ in range(n_trials):
            start_time = time.time()
            _ = model.generate(prompt)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return np.mean(times)
    
    def evaluate_all(self, prompts=None, verbose=True):
        """
        Evaluate all models on all metrics using the provided prompts.
        
        Args:
            prompts (list): List of prompt texts to generate responses from
            verbose (bool): Whether to print progress
        
        Returns:
            dict: Dictionary of evaluation results
        """
        # If no prompts are provided but we have test data, use contexts from test data
        if not prompts and self.test_data:
            prompts = [item['context'] for item in self.test_data if 'context' in item]
        
        # If still no prompts, use some default examples
        if not prompts:
            prompts = [
                "You enter the speakeasy and notice the bartender watching you suspiciously.",
                "The machine bears down on the building. You grip the piano, bracing for impact.",
                "A mysterious figure in a dark coat approaches you in the alleyway.",
                "The police officer examines your forged documents with narrowed eyes."
            ]
            
        if verbose:
            print(f"Evaluating {len(self.models)} models on {len(prompts)} prompts...")
            
        for model_name, model in self.models.items():
            if verbose:
                print(f"Evaluating {model_name}...")
                
            # Initialize results dictionary for this model
            self.results[model_name] = {
                'perplexity': [],
                'bleu': [],
                'vocabulary_diversity': [],
                'response_time': []
            }
            
            for prompt in prompts:
                # Generate text from the model
                generated_text = model.generate(prompt)
                
                # Find reference texts if any
                refs = []
                if self.test_data:
                    for item in self.test_data:
                        if item.get('context') == prompt and 'continuation' in item:
                            refs.append(item['continuation'])
                
                if not refs and self.reference_texts:
                    refs = self.reference_texts
                
                # Calculate metrics
                perplexity = self._calculate_perplexity(model, generated_text)
                bleu = self._calculate_bleu(generated_text, refs) if refs else 0.0
                vocab_diversity = self._calculate_vocabulary_diversity(generated_text)
                response_time = self._measure_response_time(model, prompt, n_trials=2)
                
                # Store results
                self.results[model_name]['perplexity'].append(perplexity)
                self.results[model_name]['bleu'].append(bleu)
                self.results[model_name]['vocabulary_diversity'].append(vocab_diversity)
                self.results[model_name]['response_time'].append(response_time)
                
                if verbose:
                    print(f"  Prompt: {prompt[:30]}...")
                    print(f"  Generated: {generated_text[:50]}...")
                    print(f"  Metrics: perplexity={perplexity:.2f}, bleu={bleu:.2f}, " 
                          f"diversity={vocab_diversity:.2f}, time={response_time:.3f}s")
        
        # Calculate average metrics for each model
        self._calculate_average_metrics()
        
        if verbose:
            self.print_results()
            
        return self.results
    
    def _calculate_average_metrics(self):
        """Calculate average metrics across all prompts for each model."""
        for model_name in self.results:
            model_results = self.results[model_name]
            # Create a list of metrics to avoid modifying dict during iteration
            metrics = list(model_results.keys())
            for metric in metrics:
                # Only calculate averages for list values (individual results)
                if isinstance(model_results[metric], list) and model_results[metric]:
                    values = model_results[metric]
                    model_results[f'avg_{metric}'] = np.mean(values)
    
    def print_results(self):
        """Print the evaluation results in a formatted table."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Print header
        metrics = ['avg_perplexity', 'avg_bleu', 'avg_vocabulary_diversity', 'avg_response_time']
        header = "Model".ljust(20)
        for metric in metrics:
            header += metric.replace('avg_', '').ljust(20)
        print(header)
        print("-"*80)
        
        # Print results for each model
        for model_name in self.results:
            row = model_name.ljust(20)
            for metric in metrics:
                value = self.results[model_name].get(metric, 0.0)
                row += f"{value:.4f}".ljust(20)
            print(row)
        
        print("="*80)
    
    def plot_comparative_results(self, save_path=None):
        """
        Generate comparative bar charts for all models.
        
        Args:
            save_path (str): Path to save the plot image, if None, display only
        """
        if not self.results:
            print("No results to plot")
            return
            
        metrics = ['avg_perplexity', 'avg_bleu', 'avg_vocabulary_diversity', 'avg_response_time']
        model_names = list(self.results.keys())
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            metric_values = [self.results[model][metric] for model in model_names]
            axes[i].bar(model_names, metric_values)
            axes[i].set_title(metric.replace('avg_', '').replace('_', ' ').title())
            axes[i].set_ylabel('Value')
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            
            # For perplexity, lower is better
            if 'perplexity' in metric:
                axes[i].set_title(metric.replace('avg_', '').replace('_', ' ').title() + " (Lower is Better)")
            # For response time, lower is better
            elif 'response_time' in metric:
                axes[i].set_title(metric.replace('avg_', '').replace('_', ' ').title() + " (Lower is Better)")
            # For other metrics, higher is better
            else:
                axes[i].set_title(metric.replace('avg_', '').replace('_', ' ').title() + " (Higher is Better)")
            
            # Rotate x-axis labels if there are many models
            if len(model_names) > 3:
                axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def export_results(self, output_path):
        """
        Export evaluation results to a CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
        """
        if not self.results:
            print("No results to export")
            return
            
        # Prepare data for DataFrame
        data = []
        for model_name in self.results:
            row = {'model': model_name}
            for metric, values in self.results[model_name].items():
                if isinstance(values, list):
                    continue  # Skip the lists of individual prompt results
                row[metric] = values
            data.append(row)
            
        # Create and save DataFrame
        df = pd.DataFrame(data)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

def load_test_data(path="data/processed/context_continuations.json"):
    """
    Load test data for evaluation.
    
    Args:
        path (str): Path to the test data JSON file
    
    Returns:
        list: List of test data examples
    """
    if not os.path.exists(path):
        print(f"Test data file not found: {path}")
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If it's a dict with keys, convert to list
        if isinstance(data, dict):
            data = [{"context": k, "continuation": v} for k, v in data.items()]
        
        print(f"Loaded {len(data)} test examples from {path}")
        return data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

def main():
    """Main function to run the evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate story generation models")
    parser.add_argument("--test-data", default="data/processed/context_continuations.json", 
                      help="Path to test data JSON file")
    parser.add_argument("--output-dir", default="data/outputs",
                      help="Directory to save output files")
    parser.add_argument("--naive-model", default="models/naive_model.pkl",
                      help="Path to naive model pickle file")
    parser.add_argument("--ngram-model", default="models/ngram_model.pkl",
                      help="Path to n-gram model pickle file")
    parser.add_argument("--transformer-model", default="models/transformer",
                      help="Path to transformer model directory")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data if available
    test_data = load_test_data(args.test_data)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(reference_texts=[item.get('continuation', '') for item in test_data])
    
    print("Loading models...")
    
    # Load the naive model
    try:
        naive_model = get_naive_model(args.naive_model)
        evaluator.add_model("Naive Model", naive_model)
    except Exception as e:
        print(f"Error loading naive model: {e}")
    
    # Load the n-gram model
    try:
        ngram_model = get_ngram_model(args.ngram_model)
        evaluator.add_model("N-gram Model", ngram_model)
    except Exception as e:
        print(f"Error loading n-gram model: {e}")
    
    # Load the transformer model
    try:
        transformer_model = get_transformer_model(args.transformer_model)
        evaluator.add_model("Transformer Model", transformer_model)
    except Exception as e:
        print(f"Error loading transformer model: {e}")
    
    # Run evaluation
    print("Running evaluation...")
    prompts = [item.get('context', '') for item in test_data[:5]]  # Use first 5 examples to avoid long runtime
    evaluator.evaluate_all(prompts=prompts)
    
    # Generate plots
    plot_path = os.path.join(args.output_dir, "model_comparison.png")
    evaluator.plot_comparative_results(save_path=plot_path)
    
    # Export results
    results_path = os.path.join(args.output_dir, "evaluation_results.csv")
    evaluator.export_results(results_path)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()