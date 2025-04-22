#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare datasets for the interactive fiction generator.
This script downloads and processes text data from various sources
to train our story generation models.
"""

import os
import sys
import argparse
import logging
import json
import random
import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import zipfile
import io

# Ensure required NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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

def extract_story_segments(text: str, min_length: int = 100, max_length: int = 1000) -> List[str]:
    """
    Extract story segments from text that can serve as training examples.
    
    Args:
        text: The input text to process
        min_length: Minimum character length for a segment
        max_length: Maximum character length for a segment
        
    Returns:
        List of story segments suitable for training
    """
    # Split into paragraphs
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    segments = []
    current_segment = ""
    
    for para in paragraphs:
        # Skip very short paragraphs or those with special chars/formatting
        if len(para) < 20 or re.search(r'[*_]{2,}|\d+\.\s|\d+\)', para):
            continue
            
        if len(current_segment) + len(para) < max_length:
            # Add to current segment
            current_segment += para + "\n\n"
        else:
            # If current segment is long enough, add it to segments
            if len(current_segment) >= min_length:
                segments.append(current_segment.strip())
            # Start a new segment
            current_segment = para + "\n\n"
    
    # Add the last segment if it's long enough
    if len(current_segment) >= min_length:
        segments.append(current_segment.strip())
    
    return segments

def download_project_gutenberg_text(book_id: int, output_dir: str) -> str:
    """
    Download a text from Project Gutenberg.
    
    Args:
        book_id: The Project Gutenberg ID of the book to download
        output_dir: Directory to save the downloaded text
        
    Returns:
        Path to the downloaded file
    """
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    alternate_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    
    output_path = os.path.join(output_dir, f"gutenberg_{book_id}.txt")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_path):
        logger.info(f"File already exists at {output_path}")
        return output_path
    
    try:
        logger.info(f"Downloading book {book_id} from Project Gutenberg")
        response = requests.get(url)
        if response.status_code != 200:
            logger.info(f"Trying alternate URL: {alternate_url}")
            response = requests.get(alternate_url)
        
        response.raise_for_status()
        
        # Save the content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        logger.info(f"Downloaded and saved to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error downloading book {book_id}: {e}")
        return None

def clean_gutenberg_text(file_path: str) -> str:
    """
    Clean Project Gutenberg text by removing header and footer.
    
    Args:
        file_path: Path to the downloaded text file
        
    Returns:
        Cleaned text content
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Find the start of the actual text (after header)
    # Typical markers include "*** START OF THIS PROJECT GUTENBERG EBOOK"
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK"
    ]
    
    for marker in start_markers:
        if marker in content:
            content = content.split(marker, 1)[1]
            break
    
    # Find the end of the actual text (before footer)
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "End of Project Gutenberg's",
        "End of the Project Gutenberg"
    ]
    
    for marker in end_markers:
        if marker in content:
            content = content.split(marker, 1)[0]
            break
    
    return content.strip()

def download_literary_works(output_dir: str, num_books: int = 5) -> List[str]:
    """
    Download classic literary works texts.
    
    Args:
        output_dir: Directory to save downloaded texts
        num_books: Number of books to download
        
    Returns:
        List of file paths for downloaded texts
    """
    # Classic literary works IDs from Project Gutenberg
    # These are public domain books covering various genres and styles
    literary_book_ids = {
        "Alice in Wonderland": 11,  # Alice's Adventures in Wonderland by Lewis Carroll
        "Sherlock Holmes": 1661,    # The Adventures of Sherlock Holmes by Arthur Conan Doyle
        "The Odyssey": 1727,        # The Odyssey by Homer
        "Pride and Prejudice": 1342, # Pride and Prejudice by Jane Austen
        "Frankenstein": 84,         # Frankenstein by Mary Shelley
        "The Time Machine": 35,     # The Time Machine by H.G. Wells
        "The Picture of Dorian Gray": 174, # The Picture of Dorian Gray by Oscar Wilde
        "A Tale of Two Cities": 98,  # A Tale of Two Cities by Charles Dickens
        "Moby Dick": 2701,          # Moby Dick by Herman Melville
        "The War of the Worlds": 36, # The War of the Worlds by H.G. Wells
        "Dracula": 345,             # Dracula by Bram Stoker
        "The Count of Monte Cristo": 1184, # The Count of Monte Cristo by Alexandre Dumas
        "Great Expectations": 1400,  # Great Expectations by Charles Dickens
        "Jane Eyre": 1260,          # Jane Eyre by Charlotte Brontë
        "The Jungle Book": 35997,   # The Jungle Book by Rudyard Kipling
    }
    
    # Select a subset of books if num_books is less than total
    if num_books < len(literary_book_ids):
        selected_titles = random.sample(list(literary_book_ids.keys()), num_books)
        selected_ids = [literary_book_ids[title] for title in selected_titles]
    else:
        selected_ids = list(literary_book_ids.values())
    
    # Download and process each book
    downloaded_files = []
    for book_id in selected_ids:
        file_path = download_project_gutenberg_text(book_id, output_dir)
        if file_path:
            downloaded_files.append(file_path)
    
    return downloaded_files

def download_detective_fiction(output_dir: str, num_books: int = 5) -> List[str]:
    """
    Download detective and noir fiction texts.
    
    Args:
        output_dir: Directory to save downloaded texts
        num_books: Number of books to download
        
    Returns:
        List of file paths for downloaded texts
    """
    # Detective/noir fiction book IDs from Project Gutenberg
    # These are public domain books in the mystery/detective genre
    detective_book_ids = [
        1661,  # The Adventures of Sherlock Holmes by Arthur Conan Doyle
        2097,  # The Hound of the Baskervilles by Arthur Conan Doyle
        863,   # The Moonstone by Wilkie Collins
        155,   # The Murders in the Rue Morgue by Edgar Allan Poe
        1438,  # A Study In Scarlet by Arthur Conan Doyle
        558,   # The Sign of the Four by Arthur Conan Doyle
        32887, # The Mysterious Affair at Styles by Agatha Christie
        244,   # A Study in Scarlet by Arthur Conan Doyle
        537,   # Frankenstein; Or, The Modern Prometheus by Mary Shelley
        1695   # The Man Who Was Thursday by G.K. Chesterton
    ]
    
    # Select a subset of books
    selected_ids = random.sample(detective_book_ids, min(num_books, len(detective_book_ids)))
    
    # Download and process each book
    downloaded_files = []
    for book_id in selected_ids:
        file_path = download_project_gutenberg_text(book_id, output_dir)
        if file_path:
            downloaded_files.append(file_path)
    
    return downloaded_files

def prepare_training_data(input_files: List[str], output_file: str, 
                         segment_min_length: int = 100, 
                         segment_max_length: int = 1000) -> int:
    """
    Process input files and prepare training data for story generation.
    
    Args:
        input_files: List of input text files
        output_file: Path to save the processed training data
        segment_min_length: Minimum character length for a story segment
        segment_max_length: Maximum character length for a story segment
        
    Returns:
        Number of segments extracted
    """
    all_segments = []
    
    for file_path in input_files:
        logger.info(f"Processing {file_path}")
        
        # Clean the text
        cleaned_text = clean_gutenberg_text(file_path)
        
        # Extract story segments
        segments = extract_story_segments(
            cleaned_text, 
            min_length=segment_min_length, 
            max_length=segment_max_length
        )
        
        logger.info(f"Extracted {len(segments)} segments from {file_path}")
        all_segments.extend(segments)
    
    # Save segments to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in all_segments:
            f.write(segment + "\n\n===\n\n")  # Separator between segments
    
    logger.info(f"Saved {len(all_segments)} segments to {output_file}")
    return len(all_segments)

def prepare_choice_templates(output_file: str) -> None:
    """
    Prepare templates for choice generation.
    
    Args:
        output_file: Path to save the choice templates
    """
    # Templates for story choices in an interactive fiction context
    templates = [
        # Movement and exploration
        "Go to the {location}",
        "Investigate the {object}",
        "Look behind the {object}",
        "Search {location} for clues",
        "Enter the {location}",
        "Leave the {location}",
        "Hide behind the {object}",
        "Explore the {location}",
        "Climb the {object}",
        "Descend into the {location}",
        
        # Character interaction
        "Talk to {character}",
        "Ask {character} about {topic}",
        "Show {object} to {character}",
        "Follow {character}",
        "Confront {character} about {topic}",
        "Offer {object} to {character}",
        "Help {character} with their problem",
        "Observe {character} from a distance",
        "Introduce yourself to {character}",
        "Persuade {character} to help you",
        
        # Object interaction
        "Pick up the {object}",
        "Use {object} on {target}",
        "Examine the {object} closely",
        "Break the {object}",
        "Combine {object} with {object2}",
        "Open the {object}",
        "Read the {object}",
        "Move the {object}",
        "Hide the {object}",
        "Repair the {object}",
        
        # General actions
        "Wait and observe",
        "Think about the situation",
        "Remember what happened earlier",
        "Make a plan",
        "Listen carefully",
        "Look around the area",
        "Try to find a way out",
        "Call for help",
        "Rest for a moment",
        "Draw your weapon",
        
        # Genre-specific templates
        # Fantasy
        "Cast a spell",
        "Consult your map",
        "Speak the magic words",
        "Drink the potion",
        
        # Mystery
        "Analyze the evidence",
        "Question the witness",
        "Follow the trail",
        "Check for fingerprints",
        
        # Science Fiction
        "Activate the device",
        "Check your oxygen level",
        "Send a transmission",
        "Reprogram the computer",
        
        # Horror
        "Run away",
        "Hold your breath",
        "Turn on the flashlight",
        "Barricade the door"
    ]
    
    # Save templates to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for template in templates:
            f.write(template + "\n")
            
    logger.info(f"Saved {len(templates)} choice templates to {output_file}")

def main(args):
    """Main data preparation function."""
    logger.info(f"Preparing data with output path: {args.output_dir}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    raw_dir = os.path.join(args.output_dir, 'raw')
    processed_dir = os.path.join(args.output_dir, 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Download literary works
    downloaded_files = download_literary_works(raw_dir, args.num_books)
    logger.info(f"Downloaded {len(downloaded_files)} books")
    
    if not downloaded_files:
        logger.error("No files were downloaded. Exiting.")
        return
    
    # 2. Process files and extract training segments
    training_data_path = os.path.join(processed_dir, 'training_segments.txt')
    num_segments = prepare_training_data(
        downloaded_files, 
        training_data_path,
        segment_min_length=args.min_length,
        segment_max_length=args.max_length
    )
    
    # 3. Prepare choice templates
    templates_path = os.path.join(processed_dir, 'choice_templates.txt')
    prepare_choice_templates(templates_path)
    
    logger.info("Data preparation complete!")
    logger.info(f"Total segments extracted: {num_segments}")
    logger.info(f"Files saved to: {processed_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and process data for story generation')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save outputs')
    parser.add_argument('--num_books', type=int, default=10,
                        help='Number of books to download')
    parser.add_argument('--min_length', type=int, default=100,
                        help='Minimum character length for segments')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Maximum character length for segments')
    
    args = parser.parse_args()
    main(args)