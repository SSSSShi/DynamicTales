import os
import torch
from typing import List, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class StoryGenerator:
    def __init__(self):
        pass
    
    def generate(self, context: str, num_words: int = 50) -> str:
        """Generate story continuation based on context."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_choices(self, context: str, num_choices: int = 3) -> List[str]:
        """Generate possible next actions based on context."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        raise NotImplementedError("Subclasses must implement this method")
    
    @classmethod
    def load(cls, path: str) -> 'StoryGenerator':
        """Load model from disk."""
        raise NotImplementedError("Subclasses must implement this method")

class TransformerStoryGenerator(StoryGenerator):
    """
    GPT-2 based story generator.
    This is our deep learning approach using transformer neural networks.
    """
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        # Load pre-trained model and tokenizer
        if model_name:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            self.tokenizer = None
            self.model = None
        
        # Special token for continuation
        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def fine_tune(self, texts: List[str], epochs: int = 3, batch_size: int = 2, 
                  learning_rate: float = 5e-5, output_dir: str = None) -> None:
        """Fine-tune the GPT-2 model on story texts."""
        # Prepare dataset
        encoded_texts = [self.tokenizer.encode(text) for text in texts]
        max_length = max(len(tokens) for tokens in encoded_texts)
        
        # Pad sequences
        padded_texts = []
        attention_masks = []
        
        for tokens in encoded_texts:
            padded = tokens + [self.tokenizer.pad_token_id] * (max_length - len(tokens))
            padded_texts.append(padded)
            
            # Mask: 1 for real tokens, 0 for padding
            mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
            attention_masks.append(mask)
        
        # Convert to PyTorch tensors
        input_ids = torch.tensor(padded_texts)
        attention_masks = torch.tensor(attention_masks)
        
        # Prepare training arguments
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            
            # Process in batches
            for i in range(0, len(input_ids), batch_size):
                batch_input_ids = input_ids[i:i+batch_size].to(device)
                batch_masks = attention_masks[i:i+batch_size].to(device)
                
                # Forward pass
                outputs = self.model(batch_input_ids, 
                                    attention_mask=batch_masks,
                                    labels=batch_input_ids)
                
                loss = outputs.loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if i % 10 == 0:
                    print(f"Batch {i}, Loss: {loss.item()}")
        
        # Save the fine-tuned model if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")
    
    def generate(self, context: str, num_words: int = 50) -> str:
        """Generate story continuation using GPT-2."""
        # Make sure we're using the same device throughout
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(device)
        
        # Generate sequence
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + num_words,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=self.tokenizer.pad_token_id,
            no_repeat_ngram_size=3  # Avoid repeating n-grams
        )
        
        # Decode only the newly generated tokens (not including the input)
        continuation = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return continuation
    
    def generate_choices(self, context: str, num_choices: int = 3) -> List[str]:
        """Generate diverse choices using GPT-2."""
        # Make sure we're using the same device throughout
        device = next(self.model.parameters()).device
        
        # Prompting technique for generating choices
        prompt = context + "\nWhat would you like to do? Choose from:\n1."
        
        choices = []
        
        # Generate multiple continuations with different sampling
        for _ in range(num_choices + 2):  # Generate extra choices in case of duplicates
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 10,  # Short continuation for choice
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=1.0,  # Higher temperature for diversity
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Extract generated text and format as a choice
            continuation = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Clean up and format
            choice = continuation.split("\n")[0].strip()
            
            # Ensure it starts with an action verb
            if choice and not choice[0].isupper() and len(choice.split()) > 1:
                # Capitalize first word
                words = choice.split()
                choice = words[0].capitalize() + " " + " ".join(words[1:])
            
            if choice and len(choice) > 3:
                choices.append(choice)
        
        # Remove duplicates and return requested number
        unique_choices = list(dict.fromkeys(choices))  # Preserve order
        return unique_choices[:num_choices]
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def load(cls, path: str) -> 'TransformerStoryGenerator':
        """Load model from disk."""
        model = cls(model_name=None)  # Initialize with no pre-trained model
        
        model.tokenizer = GPT2Tokenizer.from_pretrained(path)
        model.model = GPT2LMHeadModel.from_pretrained(path)
        
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        
        return model

# Function to load training data
def load_training_data(data_path):
    """Load context-continuation pairs for fine-tuning."""
    import json
    with open(data_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    # Format data for training
    texts = []
    for pair in pairs:
        # Combine context and continuation for full story segments
        full_text = pair['context'] + " " + pair['continuation']
        texts.append(full_text)
    
    return texts

# Factory function to get the model
def get_transformer_model(model_path: Optional[str] = None) -> StoryGenerator:
    """Factory function to get a transformer model instance."""
    if model_path:
        return TransformerStoryGenerator.load(model_path)
    return TransformerStoryGenerator()

# Usage example
if __name__ == "__main__":
    import os
    
    print("Testing transformer model:")
    transformer_model = TransformerStoryGenerator()
    
    # Path to your processed data
    data_path = "data/processed/context_continuations.json"
    
    # Check if the file exists
    if os.path.exists(data_path):
        print(f"Loading training data from {data_path}")
        training_texts = load_training_data(data_path)
        
        try:
            # Fine-tune the model (use smaller values for epochs)
            print("Fine-tuning model on story data...")
            transformer_model.fine_tune(
                texts=training_texts,
                epochs=100,  
                batch_size=1, 
                learning_rate=5e-5,
                output_dir="models/transformer"
            )
        except RuntimeError as e:
            # If CUDA error occurs, try running on CPU only
            if "cuda" in str(e).lower():
                print("GPU error encountered. Trying CPU-only mode...")
                # Force CPU mode
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                transformer_model = TransformerStoryGenerator()
                transformer_model.fine_tune(
                    texts=training_texts,
                    epochs=1,
                    batch_size=1,
                    learning_rate=5e-5,
                    output_dir="models/transformer"
                )
            else:
                # Re-raise if it's not a CUDA error
                raise
    else:
        print(f"Warning: Training data not found at {data_path}")
        print("Using pre-trained model without fine-tuning")
    
    # Test the model
    context = "You enter the speakeasy and notice the bartender watching you suspiciously."
    try:
        print("\nGenerating continuation:")
        continuation = transformer_model.generate(context)
        print(f"Continuation: {continuation}")
        
        print("\nGenerating choices:")
        choices = transformer_model.generate_choices(context)
        for i, choice in enumerate(choices, 1):
            print(f"{i}. {choice}")
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Try using the model without fine-tuning or with CPU-only mode.")