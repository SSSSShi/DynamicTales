import os
import random
import pickle
from typing import List

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

class NaiveStoryGenerator(StoryGenerator):
    """
    A simple template-based story generator.
    This is our naive baseline model that doesn't use machine learning.
    """
    def __init__(self):
        super().__init__()
        self.settings = ["speakeasy", "street", "alleyway", "police station", "warehouse"]
        self.characters = ["bartender", "cop", "gangster", "musician", "patron"]
        self.actions = ["talks to", "argues with", "fights", "escapes from", "helps"]
        self.emotions = ["happy", "sad", "angry", "surprised", "afraid"]
        self.templates = [
            "You {action} the {character} in the {setting}. They seem {emotion}.",
            "The {character} {action} you while looking {emotion}. You're still in the {setting}.",
            "While in the {setting}, you notice a {character} who seems {emotion}.",
            "A {emotion} {character} approaches you in the {setting}.",
            "You find yourself in the {setting} with a {emotion} {character}."
        ]
        self.choice_templates = [
            "Talk to the {character}",
            "Leave the {setting}",
            "Offer help to the {character}",
            "Hide from the {character}",
            "Look for clues in the {setting}",
            "Confront the {character}",
            "Find another {character}",
            "Examine the {setting} more closely"
        ]
    
    def generate(self, context: str, num_words: int = 50) -> str:
        """Generate a simple templated story segment."""
        template = random.choice(self.templates)
        return template.format(
            action=random.choice(self.actions),
            character=random.choice(self.characters),
            setting=random.choice(self.settings),
            emotion=random.choice(self.emotions)
        )
    
    def generate_choices(self, context: str, num_choices: int = 3) -> List[str]:
        """Generate simple choices using templates."""
        choices = []
        templates = random.sample(self.choice_templates, num_choices)
        
        for template in templates:
            choices.append(template.format(
                character=random.choice(self.characters),
                setting=random.choice(self.settings)
            ))
        
        return choices
    
    def save(self, path: str) -> None:
        """Save model attributes to disk."""
        model_data = {
            'settings': self.settings,
            'characters': self.characters,
            'actions': self.actions,
            'emotions': self.emotions,
            'templates': self.templates,
            'choice_templates': self.choice_templates
        }
        
        # Create the directory only if path contains a directory
        directory = os.path.dirname(path)
        if directory:  # Check if directory is not empty
            os.makedirs(directory, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str) -> 'NaiveStoryGenerator':
        """Load model attributes from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls()
        model.settings = model_data['settings']
        model.characters = model_data['characters']
        model.actions = model_data['actions']
        model.emotions = model_data['emotions']
        model.templates = model_data['templates']
        model.choice_templates = model_data['choice_templates']
        
        return model

# Factory function to get the model
def get_naive_model(model_path=None):
    """Get an instance of the naive story generator."""
    if model_path:
        return NaiveStoryGenerator.load(model_path)
    return NaiveStoryGenerator()

# Usage example
if __name__ == "__main__":
    # Example of using the naive model
    print("Testing naive model:")
    naive_model = NaiveStoryGenerator()
    context = "You enter the speakeasy and"
    continuation = naive_model.generate(context)
    print(f"Continuation: {continuation}")
    
    print("\nGenerated choices:")
    choices = naive_model.generate_choices(context)
    for i, choice in enumerate(choices, 1):
        print(f"{i}. {choice}")
    
    try:
        # Try saving to current directory instead of a nested path
        print("\nSaving model...")
        naive_model.save("models/naive_model.pkl")
        print("Model saved successfully!")
        
        # Try loading the model
        print("\nLoading model...")
        loaded_model = NaiveStoryGenerator.load("models/naive_model.pkl")
        print("Model loaded successfully!")
        
        # Test the loaded model
        print("\nTesting loaded model:")
        loaded_continuation = loaded_model.generate(context)
        print(f"Continuation: {loaded_continuation}")
    except Exception as e:
        print(f"Error: {e}")