"""
DynamicTales: Interactive Storytelling Application with Multiple Models

This Streamlit application demonstrates an interactive text adventure game
that uses three different models to generate story content:
1. A naive template-based model (baseline)
2. An n-gram statistical model (classical ML)
3. A transformer neural network model (deep learning)

The application allows users to compare these models and interact with a dynamically
generated story that responds to their choices.
"""

import os
import sys
import time
import random
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re

# Add the current directory to the path so Python can find our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model classes
try:
    from scripts.naive_model import NaiveStoryGenerator, get_naive_model
    from scripts.non_dl_model import NGramStoryGenerator, get_model as get_ngram_model
    from scripts.deep_learning_model import TransformerStoryGenerator, get_transformer_model
except ImportError as e:
    st.error(f"Error importing model modules: {e}")
    st.info("Make sure you're running this app from the project root directory")

# Set page configuration
st.set_page_config(
    page_title="DynamicTales - Interactive Storytelling",
    page_icon="📚",
    layout="wide"
)

# Define CSS styles
def load_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f5f5f5;
        }
        
        /* Story container */
        .story-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Story title */
        .story-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        
        /* Story text */
        .story-text {
            font-size: 18px;
            line-height: 1.6;
            color: #444;
            margin-bottom: 15px;
        }
        
        /* Choice button styling */
        .choice-btn {
            background-color: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px 15px;
            margin: 5px 0;
            text-align: left;
            cursor: pointer;
            transition: background-color 0.3s;
            width: 100%;
        }
        .choice-btn:hover {
            background-color: #e0e0e0;
        }
        
        /* Objective box */
        .objective-box {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
        
        /* Model selection container */
        .model-selection {
            background-color: #eef;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        /* Divider */
        hr {
            margin: 20px 0;
            border: 0;
            height: 1px;
            background: #ddd;
        }
        
        /* For the character avatar */
        .avatar {
            border-radius: 50%;
            width: 80px;
            height: 80px;
            margin: 0 auto;
            display: block;
        }
        
        /* Custom containers with decorative borders */
        .decorated-container {
            position: relative;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        
        .decorated-container:before,
        .decorated-container:after {
            content: '★';
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            font-size: 24px;
            color: #666;
        }
        
        .decorated-container:before {
            left: 10px;
        }
        
        .decorated-container:after {
            right: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

def load_evaluation_data(filepath="data/outputs/evaluation_results.csv"):
    """Load model evaluation data if available."""
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            # Return dummy data if file doesn't exist
            return pd.DataFrame({
                'model': ['Naive Model', 'N-gram Model', 'Transformer Model'],
                'avg_perplexity': [10.20, 0.80, 8.20],
                'avg_bleu': [0.084, 0.134, 0.354],
                'avg_vocabulary_diversity': [0.94, 0.754, 0.794],
                'avg_response_time': [0.0002, 0.0002, 0.742]
            })
    except Exception as e:
        st.warning(f"Error loading evaluation data: {e}")
        # Return dummy data
        return pd.DataFrame({
            'model': ['Naive Model', 'N-gram Model', 'Transformer Model'],
            'avg_perplexity': [10.20, 0.80, 8.20],
            'avg_bleu': [0.084, 0.134, 0.354],
            'avg_vocabulary_diversity': [0.94, 0.754, 0.794],
            'avg_response_time': [0.0002, 0.0002, 0.742]
        })

def load_or_create_models():
    """Load or create the three story generation models."""
    models = {}
    
    # Load or create naive model
    try:
        if os.path.exists("models/naive_model.pkl"):
            models["naive"] = get_naive_model("models/naive_model.pkl")
        else:
            models["naive"] = get_naive_model()
            os.makedirs("models", exist_ok=True)
            models["naive"].save("models/naive_model.pkl")
        st.session_state.model_status["naive"] = "Loaded successfully"
    except Exception as e:
        st.error(f"Error loading naive model: {e}")
        models["naive"] = get_naive_model()  # Use default model
        st.session_state.model_status["naive"] = f"Error: {str(e)[:50]}..."
    
    # Load or create n-gram model
    try:
        if os.path.exists("models/ngram_model.pkl"):
            models["ngram"] = get_ngram_model("models/ngram_model.pkl")
        else:
            models["ngram"] = get_ngram_model()
            os.makedirs("models", exist_ok=True)
            models["ngram"].save("models/ngram_model.pkl")
        st.session_state.model_status["ngram"] = "Loaded successfully"
    except Exception as e:
        st.error(f"Error loading n-gram model: {e}")
        models["ngram"] = get_ngram_model()  # Use default model
        st.session_state.model_status["ngram"] = f"Error: {str(e)[:50]}..."
    
    # Load or create transformer model
    try:
        if os.path.exists("models/transformer"):
            models["transformer"] = get_transformer_model("models/transformer")
        else:
            models["transformer"] = get_transformer_model()
            os.makedirs("models/transformer", exist_ok=True)
        st.session_state.model_status["transformer"] = "Loaded successfully"
    except Exception as e:
        st.error(f"Error loading transformer model: {e}")
        # For transformer model, we need to handle the error more gracefully
        # since it might not be possible to create a default model easily
        # Use a simple wrapper that mimics the interface
        class FallbackTransformerModel:
            def generate(self, context, num_words=50):
                return f"[Transformer model could not be loaded. This is a fallback response to: {context[:30]}...]"
            
            def generate_choices(self, context, num_choices=3):
                return [f"Fallback choice {i+1}" for i in range(num_choices)]
        
        models["transformer"] = FallbackTransformerModel()
        st.session_state.model_status["transformer"] = f"Error: {str(e)[:50]}..."
    
    return models

# Initialize or load session state
def initialize_state():
    if 'story_history' not in st.session_state:
        st.session_state.story_history = []
    
    if 'current_objective' not in st.session_state:
        st.session_state.current_objective = "Explore the world of DynamicTales"
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "transformer"  # Default model
    
    if 'model_status' not in st.session_state:
        st.session_state.model_status = {
            "naive": "Not loaded",
            "ngram": "Not loaded",
            "transformer": "Not loaded"
        }
    
    if 'models' not in st.session_state:
        # Initialize models
        st.session_state.models = load_or_create_models()
    
    if 'character_name' not in st.session_state:
        st.session_state.character_name = "Adventurer"
    
    if 'evaluation_metrics' not in st.session_state:
        # Load evaluation metrics if available
        metrics_df = load_evaluation_data()
        
        st.session_state.evaluation_metrics = {}
        for _, row in metrics_df.iterrows():
            model_name = row['model']
            if model_name == 'Naive Model':
                key = 'naive'
            elif model_name == 'N-gram Model':
                key = 'ngram'
            elif model_name == 'Transformer Model':
                key = 'transformer'
            else:
                continue
                
            st.session_state.evaluation_metrics[key] = {
                "perplexity": row.get('avg_perplexity', 0.0),
                "bleu": row.get('avg_bleu', 0.0),
                "vocabulary_diversity": row.get('avg_vocabulary_diversity', 0.0),
                "response_time": row.get('avg_response_time', 0.0)
            }

# Function to generate a story segment with the current model
def generate_story_segment(prompt, context=None):
    model_key = st.session_state.current_model
    model = st.session_state.models[model_key]
    
    # Add loading indicator
    with st.spinner("Generating story..."):
        # If there's no context, start a new story
        if context is None:
            try:
                start_time = time.time()
                result = model.generate(prompt, num_words=100)
                end_time = time.time()
                st.session_state.last_generation_time = end_time - start_time
            except Exception as e:
                st.error(f"Error generating story: {e}")
                result = f"The story continues, but something seems off... [Error: {str(e)[:50]}]"
        else:
            # Otherwise continue from previous context
            try:
                start_time = time.time()
                result = model.generate(context + "\n" + prompt, num_words=100)
                end_time = time.time()
                st.session_state.last_generation_time = end_time - start_time
            except Exception as e:
                st.error(f"Error generating story: {e}")
                result = f"The story continues, but something seems off... [Error: {str(e)[:50]}]"
    
    return result

# Function to generate choices
def generate_choices(current_text):
    model_key = st.session_state.current_model
    model = st.session_state.models[model_key]
    
    try:
        choices = model.generate_choices(current_text, num_choices=3)
    except Exception as e:
        st.error(f"Error generating choices: {e}")
        # Fallback choices
        choices = [
            "Continue exploring this area",
            "Try a different approach",
            "Talk to someone nearby",
            "Look for clues or items"
        ]
    
    # Ensure we have at least 3 choices
    while len(choices) < 3:
        choices.append(f"Option {len(choices)+1}")
    
    # Limit to max 5 choices
    return choices[:5]

# Function to add a story segment to history
def add_to_history(text, is_user=False, choices=None):
    st.session_state.story_history.append({
        "text": text,
        "is_user": is_user,
        "choices": choices if choices else []
    })

# Function to display the story history
def display_story():
    for i, segment in enumerate(st.session_state.story_history):
        if segment["is_user"]:
            # User response
            st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 5px; margin: 10px 0;'><strong>You:</strong> {segment['text']}</div>", unsafe_allow_html=True)
        else:
            # Story segment
            st.markdown(f"<div class='story-text'>{segment['text']}</div>", unsafe_allow_html=True)

# Display objective in a decorated box
def display_objective(objective):
    st.markdown(f"""
    <div class="decorated-container">
        <h3 style="text-align: center;">OBJECTIVE</h3>
        <p style="text-align: center; font-size: 20px; font-weight: bold;">{objective}</p>
    </div>
    """, unsafe_allow_html=True)

def display_model_metrics():
    """Display metrics and comparison chart for the models."""
    if 'evaluation_metrics' not in st.session_state:
        st.warning("No evaluation metrics available")
        return
    
    metrics = st.session_state.evaluation_metrics
    
    # Create a DataFrame for the metrics
    data = []
    for model_name, model_metrics in metrics.items():
        row = {'Model': model_name.capitalize()}
        for metric_name, value in model_metrics.items():
            row[metric_name.replace('_', ' ').title()] = value
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Display as a table
    st.subheader("Model Performance Comparison")
    st.dataframe(df.set_index('Model'), width=800)
    
    # Create a bar chart for each metric
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    metrics_to_plot = ['perplexity', 'bleu', 'vocabulary_diversity', 'response_time']
    titles = ['Perplexity (Lower is Better)', 'BLEU Score (Higher is Better)', 
              'Vocabulary Diversity (Higher is Better)', 'Response Time in Seconds (Lower is Better)']
    
    for i, metric in enumerate(metrics_to_plot):
        # Extract values for this metric
        values = [m.get(metric, 0) for m in metrics.values()]
        model_names = [name.capitalize() for name in metrics.keys()]
        
        # Create bar chart
        bars = axes[i].bar(model_names, values)
        axes[i].set_title(titles[i])
        axes[i].set_ylabel('Value')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)

# Main function to run the Streamlit app
def main():
    # Load CSS
    load_css()
    
    # Initialize state
    initialize_state()
    
    # Sidebar for settings and metrics
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/story-book.png", width=80)
        st.title("DynamicTales")
        
        # Model selection
        st.header("Model Selection")
        model_options = {
            "naive": "Basic Model (Naive Baseline)",
            "ngram": "N-gram Model (Statistical)",
            "transformer": "Transformer Model (Deep Learning)"
        }
        selected_model = st.selectbox(
            "Select storytelling model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.current_model)
        )
        
        # Update current model if changed
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
        
        # Display model status
        st.caption(f"Status: {st.session_state.model_status.get(selected_model, 'Unknown')}")
        
        # Display last generation time if available
        if hasattr(st.session_state, 'last_generation_time'):
            st.caption(f"Last generation time: {st.session_state.last_generation_time:.3f} seconds")
        
        # Display model metrics
        if 'evaluation_metrics' in st.session_state and selected_model in st.session_state.evaluation_metrics:
            st.header("Model Performance")
            metrics = st.session_state.evaluation_metrics[selected_model]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Perplexity", f"{metrics['perplexity']:.1f}")
                st.metric("BLEU Score", f"{metrics['bleu']:.2f}")
            with col2:
                st.metric("Vocab Diversity", f"{metrics['vocabulary_diversity']:.2f}")
                st.metric("Response Time", f"{metrics['response_time']:.3f}s")
            
            # Information about metrics
            with st.expander("About these metrics"):
                st.markdown("""
                - **Perplexity**: Lower is better. Measures how well the model predicts text.
                - **BLEU Score**: Higher is better. Measures text quality compared to references.
                - **Vocabulary Diversity**: Higher is better. Ratio of unique words to total words.
                - **Response Time**: Lower is better. Average time to generate a response.
                """)
        
        # Character settings
        st.header("Character Settings")
        st.session_state.character_name = st.text_input("Your character name:", value=st.session_state.character_name)
        
        # Reset button
        if st.button("Start New Story"):
            st.session_state.story_history = []
            st.session_state.current_objective = "Explore the world of DynamicTales"
            st.rerun()
    
    # Main content area
    st.title("DynamicTales: Interactive Storytelling")
    
    # Information tabs
    tab1, tab2 = st.tabs(["Adventure", "Model Comparison"])
    
    with tab1:  # Adventure tab
        # Display current objective
        display_objective(st.session_state.current_objective)
        
        # If no story has been started, show initial prompt
        if not st.session_state.story_history:
            st.subheader("Begin Your Adventure")
            st.write("Welcome to DynamicTales, where every choice creates a unique adventure. You can try three different AI storytelling models:")

            col1, col2 = st.columns([1, 3])
            with col1:
                st.write("**Naive Model:**")
                st.write("**N-gram Model:**")
                st.write("**Transformer Model:**")
            with col2:
                st.write("Simple and fast but limited")
                st.write("Statistical patterns with more variety")
                st.write("Neural network with deeper understanding")
                
            st.write("Select a starting scenario below and begin your journey!")
            
            # Initial scenario options
            scenarios = [
                "The machine bears down on the building. You grip the piano, bracing for impact, mind racing for a solution.",
                "You wake up in a strange laboratory with no memory of how you got there.",
                "The ancient temple doors creak open, revealing a glittering treasure chamber.",
                "The space station alarm blares as oxygen levels begin to drop rapidly.",
                "You step onto the stage, the spotlight blinds you as the crowd falls silent.",
                "The speakeasy door opens and you're greeted by the sound of jazz music."
            ]
            
            selected_scenario = st.selectbox("Choose your starting scenario:", scenarios)
            
            if st.button("Begin Adventure"):
                # Generate the first story segment
                story_text = generate_story_segment(selected_scenario)
                
                # Set a relevant objective based on the scenario
                if "machine bears down" in selected_scenario:
                    st.session_state.current_objective = "Protect the building and find a way to stop the machine!"
                elif "laboratory" in selected_scenario:
                    st.session_state.current_objective = "Discover who you are and how to escape the laboratory!"
                elif "temple" in selected_scenario:
                    st.session_state.current_objective = "Explore the temple and secure its treasures without triggering traps!"
                elif "space station" in selected_scenario:
                    st.session_state.current_objective = "Repair the oxygen system and find out what caused the failure!"
                elif "stage" in selected_scenario:
                    st.session_state.current_objective = "Deliver an unforgettable performance and win over the audience!"
                elif "speakeasy" in selected_scenario:
                    st.session_state.current_objective = "Uncover the secrets of the speakeasy and its mysterious patrons!"
                
                # Add to history
                add_to_history(story_text)
                
                # Generate choices
                choices = generate_choices(story_text)
                st.session_state.current_choices = choices
                
                # Rerun to update the UI
                st.rerun()
        else:
            # Display the story history
            display_story()
            
            # Display current choices or an input field
            if st.session_state.story_history and not st.session_state.story_history[-1]["is_user"]:
                # Show generated choices
                st.markdown("### What will you do next?")
                
                # If we have choices from the previous state, use those
                if hasattr(st.session_state, 'current_choices') and st.session_state.current_choices:
                    choices = st.session_state.current_choices
                else:
                    # Generate new choices based on the last story segment
                    choices = generate_choices(st.session_state.story_history[-1]["text"])
                    st.session_state.current_choices = choices
                
                # Display each choice as a button
                for choice in choices:
                    if st.button(choice, key=f"choice_{choice}"):
                        # Add the user's choice to history
                        add_to_history(choice, is_user=True)
                        
                        # Generate next story segment based on this choice
                        next_segment = generate_story_segment(choice, context=st.session_state.story_history[-2]["text"])
                        add_to_history(next_segment)
                        
                        # Generate new choices for next interaction
                        new_choices = generate_choices(next_segment)
                        st.session_state.current_choices = new_choices
                        
                        # Rerun to update the UI
                        st.rerun()
                
                # Custom input option
                st.markdown("### Or write your own action:")
                custom_action = st.text_input("Enter what you want to do:", key="custom_action")
                if st.button("Submit", key="submit_custom"):
                    if custom_action:
                        # Add the user's custom action to history
                        add_to_history(custom_action, is_user=True)
                        
                        # Generate next story segment based on this custom action
                        next_segment = generate_story_segment(custom_action, context=st.session_state.story_history[-2]["text"])
                        add_to_history(next_segment)
                        
                        # Generate new choices for next interaction
                        new_choices = generate_choices(next_segment)
                        st.session_state.current_choices = new_choices
                        
                        # Rerun to update the UI
                        st.rerun()
    
    with tab2:  # Model comparison tab
        st.header("Model Comparison")
        
        # Display model information
        st.markdown("""
        This application compares three different approaches to text generation:
        
        1. **Naive Model (Baseline)**
           * A simple template-based generator
           * Uses predefined templates and random selections
           * Very fast but repetitive and limited
        
        2. **N-gram Model (Statistical)**
           * Uses statistical patterns from training data
           * Captures local text patterns and word frequencies
           * Traditional NLP approach (not deep learning)
        
        3. **Transformer Model (Deep Learning)**
           * Based on the GPT-2 architecture
           * Uses neural networks trained on large text corpora
           * Can generate more coherent and diverse content
        """)
        
        # Display metrics comparison
        display_model_metrics()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 12px;'>"
        "DynamicTales - Interactive Storytelling Application - "
        "Powered by Machine Learning Models for Story Generation"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()