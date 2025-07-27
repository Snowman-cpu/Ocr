# -*- coding: utf-8 -*-
"""
Historical Spanish Document OCR Pipeline

A comprehensive system for performing OCR on early modern Spanish documents,
with support for pre-processed images and .docx transcriptions.
"""

# ================= PART 1: SETUP AND IMPORTS =================

import os
import glob
import re
import random
import warnings
import shutil
import math
import string
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable

# Install required packages
import subprocess
try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image, ImageDraw
    from tqdm.notebook import tqdm
    import docx  # For handling .docx files
    
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from datasets import Dataset as HFDataset
    
    from IPython.display import display, HTML, clear_output
    import pandas as pd
except ImportError:
    print("Installing required packages...")
    subprocess.run(["pip", "install", "torch", "torchvision", "transformers", "datasets",
                    "opencv-python", "matplotlib", "tqdm", "pillow", "pandas", "python-docx"])
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image, ImageDraw
    from tqdm.notebook import tqdm
    import docx
    
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from datasets import Dataset as HFDataset
    
    from IPython.display import display, HTML, clear_output
    import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define base paths
base_path = '/content'
output_base_path = os.path.join(base_path, 'ocr_data')
transcriptions_path = os.path.join(base_path, 'transcriptions')
results_path = os.path.join(output_base_path, 'results')

# Create necessary directories
os.makedirs(os.path.join(output_base_path, "processed_images"), exist_ok=True)
os.makedirs(os.path.join(output_base_path, "binary_images"), exist_ok=True)
os.makedirs(transcriptions_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

# Check for available GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Environment setup complete!")

# ================= PART 2: LOADING PROCESSED IMAGES =================

def load_processed_images(processed_images_dir: str, binary_images_dir: str, max_pages: int = 6) -> Dict[str, List[str]]:
    """
    Load pre-processed images directly from the specified directories.
    
    Args:
        processed_images_dir: Directory containing processed images, organized by document ID
        binary_images_dir: Directory containing binary images, organized by document ID
        max_pages: Maximum number of pages to load per document
        
    Returns:
        Dictionary mapping document IDs to lists of processed image paths
    """
    # Dictionary to store document ID to image paths mapping
    document_images: Dict[str, List[str]] = {}
    
    # Check if directories exist
    if not os.path.exists(processed_images_dir):
        print(f"Error: Processed images directory {processed_images_dir} not found")
        return document_images
        
    # Get all document directories in the processed images folder
    document_dirs = [d for d in os.listdir(processed_images_dir) 
                     if os.path.isdir(os.path.join(processed_images_dir, d))]
    
    print(f"Found {len(document_dirs)} document directories")
    
    # Process each document directory
    for doc_id in tqdm(document_dirs, desc="Loading document images"):
        doc_dir = os.path.join(processed_images_dir, doc_id)
        
        # Get all image files in this document directory
        image_files = [f for f in os.listdir(doc_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sort image files to ensure correct page order
        image_files.sort()
        
        # Limit to max_pages
        image_files = image_files[:max_pages]
        
        # Create full paths to image files
        image_paths = [os.path.join(doc_dir, f) for f in image_files]
        
        # Store the image paths
        document_images[doc_id] = image_paths
        
        print(f"Loaded {len(image_paths)} processed images for {doc_id}")
        
    # Check if we loaded any images
    total_loaded = sum(len(paths) for paths in document_images.values())
    if total_loaded == 0:
        print("WARNING: No processed images were loaded")
    else:
        print(f"Successfully loaded {total_loaded} processed images from {len(document_images)} documents")
        
    return document_images

# ================= PART 3: DOCX TRANSCRIPTION HANDLING =================

def load_docx_transcriptions(transcriptions_path: str, image_paths: List[str]) -> Dict[str, str]:
    """
    Load transcriptions from .docx files for the provided images.
    
    Args:
        transcriptions_path: Path containing the transcription files (.docx)
        image_paths: List of image paths to find transcriptions for
        
    Returns:
        Dictionary mapping image paths to transcriptions
    """
    transcriptions: Dict[str, str] = {}
    
    # Get all transcription files (.docx)
    transcription_files = glob.glob(os.path.join(transcriptions_path, "*.docx"))
    
    # Check if we have any transcription files
    if not transcription_files:
        print("No transcription files (.docx) found.")
        return transcriptions
    
    print(f"Found {len(transcription_files)} .docx transcription files")
        
    # Helper function to extract text from a .docx file
    def extract_text_from_docx(docx_path: str) -> str:
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():  # Only include non-empty paragraphs
                full_text.append(para.text)
        return '\n'.join(full_text)
    
    # For each image path
    for image_path in image_paths:
        # Extract the document ID from the image path
        img_dir = os.path.dirname(image_path)
        doc_id = os.path.basename(img_dir)
        
        # Find the corresponding transcription file
        transcription_file = os.path.join(transcriptions_path, f"{doc_id}.docx")
        
        if os.path.exists(transcription_file):
            # Load the transcription from the .docx file
            try:
                transcription = extract_text_from_docx(transcription_file)
                # Assign the transcription to the image
                transcriptions[image_path] = transcription
            except Exception as e:
                print(f"Error loading transcription from {transcription_file}: {e}")
        else:
            # No transcription found for this document
            if 'dummy' not in doc_id.lower():  # Skip warning for dummy images
                print(f"No transcription found for {doc_id}")
                
    print(f"Loaded {len(transcriptions)} transcriptions from .docx files")
    return transcriptions

def create_dummy_transcriptions(document_images: Dict[str, List[str]], transcriptions_path: str) -> None:
    """
    Create dummy transcriptions for testing purposes if real transcriptions aren't available.
    
    Args:
        document_images: Dictionary mapping document IDs to processed image paths
        transcriptions_path: Path to save the transcriptions
    """
    os.makedirs(transcriptions_path, exist_ok=True)
    
    # Create dummy transcriptions for each document
    for doc_id, image_paths in document_images.items():
        # Create a transcription file for this document
        transcription_file = os.path.join(transcriptions_path, f"{doc_id}.txt")
        
        # Create dummy transcription content based on document type
        if "Buendia" in doc_id:
            transcription_content = """AL
INFINITAMENTE AMABLE
NIÑO JESUS.

A Vos, Dulcissimo Niño
JESUS, que no solo os
dignasteis de llamaros
Doctor de los Niños,
sino también de assis-
tir como Niño entre los Doctores,
se consagra humilde esta pequeña
Instrucción de los Niños. Es assi,
que ella también se dirige a la ju-
ventud; pero a esta, como recuer-
do de lo que aprendió, a los Ni-
ños, como precisa explicacion de
lo que deben estudiar. Por este so-
lo titulo es muy vuestra; y por
ser para Niños, que confiais a la
educacion de vuestra Compañia,
lo es mucho mas. En Vos, (Divi-
no Exemplar de todas las virtu-
des) tienen abreviado el mas se-"""
            
        elif "Mendo" in doc_id:
            transcription_content = """AL IllUSTRISSIMO SEÑOR

DON ALONSO PEREZ

DE GUZMAN EL BUENO,

PATRIARCHA DE LAS INDIAS

Arzobispo de Tyro, Limosnero mayor del Rey

Nuestro Señor Don Felipe IV. El Grande Rey de

las Españas, del Consejo de su Magestad, y Iuez

Eclesiastico Ordinario de su Real Capilla, Casa,

y Corte.

SEGUNDA vez, (Illustrissimo Señor) Salen de

la estampa estos Documentos Politicos, y Morales

para formar vn Principe perfecto, y Ministros aju-

stados, por averse despachado en tiempo breve la Im-

presion primera. Helos añadido de nuevo, y exornado

con estampas de Emblemas, que con mas halago de los ojos pongan a

la vista las enseñanzas."""
            
        elif "Ezcaray" in doc_id:
            transcription_content = """SEÑOR ILUSTRISSIMO

Ocupado en el exercicio
de las Missiones en el
Obispado de Guadala-
xara, recibi una de V.S.I.
en que me da noticia de
como su Magestad (que Dios guarde)
se avia servido de honrarme con la
merced de su Predicador; y como no
se opone la predicacion de su Mages-
tad a la Apostolica, tuve por de mi obli
gacion admitir el favor, rindiendo a
V.S.I. el agradecimiento.
El Rey mi señor (que Dios guarde)
hizo la gracia; mas a V.S.I. se le debe:
que por mas frutos, que diera la tierra
de Promission, no los lograra Moyses,
si Josue, y Caleb no los sacassen. Dos
sacaron el fruto, y de ambos necessito,
para hallar un simil proporcionado a la
grandeza de V.S.I."""
            
        elif "Constituciones" in doc_id:
            transcription_content = """DON PHELIPPE POR LA
Gracia de Dios, Rey de Castilla, de
Leon, de Aragon, de las dos Sici-
lias, de Hierusalem, de Portugal, de
Navarra, de Granada, de Toledo,
de Valencia, de Galizia, de Mallorca,
de Sevilla, de Cerdeña, de Cordova,
de Corcega, de Murcia, de Jaen, de
los Algarves, de Algecira, de Gibraltar, de las Islas de
Canaria, de las Indias Orientales, y Occidentales, Islas
y tierra firme del mar Oceano, Archiduque de Austria,
Duque de Borgoña, de Bravante, y Milan, Conde de Abs
purg, de Flandes, y de Tirol, señor de Vizcaya, y de Mo-
lina, &c. Por quanto por parte de vos, el Reverendo in
Christo Padre, don Pedro Manso, Obispo de Calahorra,
y la Calzada, del nuestro Consejo: nos fue hecha relacion"""
            
        elif "Paredes" in doc_id:
            transcription_content = """CAXA ALTA, Y BAXA

Estas están en lugar de versalillas.

        B    C    D    E    F    G    H    A    B    C    D    E    F    G    H
  ----- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
  I     K    L    M    N    O    P    Q    I         L    M    N    O    P    Q

  R     S    T    V    X    Y    Z    Æ    R    S    T    V    X    Y    Z    

  1     2    3    4    5    6    7    8    9    a    e    i    o    u         

  à     è    ı    ò    ù    q    á    é    í    ó    ú                        ¶"""
            
        else:
            transcription_content = f"""Sample transcription for {doc_id}

This is a placeholder text that simulates a transcription of the document.
In a real scenario, this would contain the actual text from the document.

Este documento histórico español requiere un análisis detallado.
El texto contiene información valiosa sobre la cultura y sociedad de su época."""
            
        # Write the transcription to the file
        with open(transcription_file, 'w', encoding='utf-8') as f:
            f.write(transcription_content)
            
        print(f"Created dummy transcription for {doc_id}")

# ================= PART 4: DATASET AND DATA LOADER CREATION =================

def create_train_val_split(image_paths: List[str], transcriptions: Dict[str, str], val_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
    """
    Create stratified train and validation splits to ensure representation of all document types.
    
    Args:
        image_paths: List of image paths
        transcriptions: Dictionary mapping image paths to transcriptions
        val_ratio: Ratio of validation data
        
    Returns:
        Tuple of (train_image_paths, val_image_paths)
    """
    # Filter image paths to only include those with transcriptions
    valid_image_paths = [path for path in image_paths if path in transcriptions]
    
    # Group images by document to ensure validation set includes all document types
    doc_to_images: Dict[str, List[str]] = {}
    for path in valid_image_paths:
        doc_id = os.path.basename(os.path.dirname(path))
        if doc_id not in doc_to_images:
            doc_to_images[doc_id] = []
        doc_to_images[doc_id].append(path)
    
    # For each document, split images into train and validation
    train_paths: List[str] = []
    val_paths: List[str] = []
    
    for doc_id, paths in doc_to_images.items():
        # Shuffle the paths for this document
        random.shuffle(paths)
        
        # Calculate split index
        split_idx = int(len(paths) * (1 - val_ratio))
        
        # Add to train and validation sets
        train_paths.extend(paths[:split_idx])
        val_paths.extend(paths[split_idx:])
    
    print(f"Created splits - Train: {len(train_paths)}, Validation: {len(val_paths)}")
    return train_paths, val_paths

class OCRDataset(Dataset):
    """
    Dataset class for OCR training and evaluation.
    """
    def __init__(self, 
                 image_paths: List[str], 
                 transcriptions: Dict[str, str], 
                 transform: Optional[Any] = None, 
                 max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image paths
            transcriptions: Dictionary mapping image paths to transcriptions
            transform: Optional transform to be applied to the images
            max_length: Maximum sequence length for the transcriptions
        """
        self.image_paths = image_paths
        self.transcriptions = transcriptions
        self.transform = transform
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get image path for the given index
        image_path = self.image_paths[idx]
        
        # Load image and apply transform if available
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a blank image as fallback
            image = Image.new("RGB", (384, 384), color="white")
            if self.transform:
                image = self.transform(image)
            
        # Get transcription and truncate if necessary
        text = self.transcriptions.get(image_path, "")
        if len(text) > self.max_length:
            text = text[:self.max_length]
            
        return {"image": image, "text": text, "image_path": image_path}

def create_data_loaders(train_image_paths: List[str], 
                        val_image_paths: List[str], 
                        transcriptions: Dict[str, str], 
                        batch_size: int = 2) -> Tuple[DataLoader, DataLoader]:  # Reduced batch size
    """
    Create data loaders for training and validation.
    
    Args:
        train_image_paths: List of training image paths
        val_image_paths: List of validation image paths
        transcriptions: Dictionary mapping image paths to transcriptions
        batch_size: Batch size (default reduced to 2 for memory efficiency)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Resize for TrOCR
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = OCRDataset(train_image_paths, transcriptions, transform)
    val_dataset = OCRDataset(val_image_paths, transcriptions, transform)
    
    # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Reduced to 0 for memory efficiency
        pin_memory=False  # Disabled for memory efficiency
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Reduced to 0 for memory efficiency
        pin_memory=False  # Disabled for memory efficiency
    )
    
    return train_loader, val_loader

def save_example_images(document_images: Dict[str, List[str]], 
                        output_base_path: str, 
                        num_examples: int = 2) -> None:
    """
    Save example images from different documents for visualization.
    
    Args:
        document_images: Dictionary mapping document IDs to processed image paths
        output_base_path: Base folder for outputs
        num_examples: Number of examples to save per document
    """
    # Create example folder
    example_folder = os.path.join(output_base_path, "examples")
    os.makedirs(example_folder, exist_ok=True)
    
    # Collect examples from each document
    for doc_id, image_paths in document_images.items():
        # Skip if no images
        if not image_paths:
            continue
            
        # Take the first few images from each document
        for i, path in enumerate(image_paths[:num_examples]):
            # Load and display the image
            try:
                img = cv2.imread(path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.figure(figsize=(8, 10))
                plt.imshow(img_rgb)
                plt.title(f"{doc_id} - Example {i+1}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error displaying example image {path}: {e}")
                continue

# ================= PART 5: HISTORICAL SPANISH TEXT PROCESSING =================

def normalize_historical_spanish(text: str) -> str:
    """
    Normalize historical Spanish text.
    
    This handles common variations in early modern Spanish typography:
    - Long s (ſ) -> s
    - Ligatures like æ -> ae
    - U/V variations (often interchangeable in early texts)
    - I/J variations (often interchangeable in early texts)
    - Double consonants
    - Contractions and abbreviations
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Replace long s with regular s
    text = text.replace('ſ', 's')
    
    # Replace ligatures
    text = text.replace('æ', 'ae').replace('œ', 'oe')
    
    # Handle common abbreviations in historical Spanish
    # This is a simplified example - a complete list would be much longer
    abbreviations = {
        'q̃': 'que',
        'ẽ': 'en',
        'õ': 'on',
        'ñ': 'nn',  # In some early texts
        'ȷ': 'i',    # dotless i
        'ç': 'z',    # cedilla
    }
    
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
        
    return text

def create_lexicon_from_transcriptions(transcriptions: Dict[str, str], min_word_length: int = 2) -> Set[str]:
    """
    Create a lexicon from the transcriptions to help with post-processing.
    
    Args:
        transcriptions: Dictionary mapping image paths to transcriptions
        min_word_length: Minimum word length to include in the lexicon
        
    Returns:
        Set of unique words
    """
    lexicon: Set[str] = set()
    
    for text in transcriptions.values():
        # Normalize the text
        normalized_text = normalize_historical_spanish(text)
        
        # Split into words and add to lexicon
        words = normalized_text.split()
        for word in words:
            # Clean the word
            clean_word = word.strip('.,;:!?()[]{}"\'-—')
            
            # Only add words that meet the minimum length
            if len(clean_word) >= min_word_length:
                lexicon.add(clean_word.lower())
                
    return lexicon

def augment_lexicon_with_variations(lexicon: Set[str]) -> Set[str]:
    """
    Augment the lexicon with common historical variations.
    
    Args:
        lexicon: Set of unique words
        
    Returns:
        Augmented lexicon
    """
    augmented_lexicon = set(lexicon)
    
    # Common character substitutions in early modern Spanish
    substitutions = [
        ('v', 'u'),   # v/u variations
        ('u', 'v'),
        ('i', 'j'),   # i/j variations
        ('j', 'i'),
        ('y', 'i'),   # y/i variations
        ('i', 'y'),
        ('ç', 'z'),   # cedilla/z variations
        ('z', 'ç'),
        ('f', 'ff'),  # single/double consonant variations
        ('ff', 'f'),
        ('s', 'ss'),
        ('ss', 's'),
        ('n', 'ñ'),   # n/ñ variations
        ('ñ', 'n'),
    ]
    
    # Add variations to the lexicon
    for word in lexicon:
        for old, new in substitutions:
            if old in word:
                variation = word.replace(old, new)
                augmented_lexicon.add(variation)
                
    return augmented_lexicon

def add_common_spanish_words(lexicon: Set[str]) -> Set[str]:
    """
    Add common Spanish words to the lexicon.
    
    Args:
        lexicon: Existing lexicon
        
    Returns:
        Expanded lexicon
    """
    # Common Spanish articles, prepositions, conjunctions, etc.
    common_words = {
        # Articles
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        
        # Prepositions
        'a', 'ante', 'bajo', 'con', 'contra', 'de', 'desde', 'en',
        'entre', 'hacia', 'hasta', 'para', 'por', 'según', 'sin', 'sobre', 'tras',
        
        # Conjunctions
        'y', 'e', 'ni', 'o', 'u', 'pero', 'mas', 'sino', 'aunque', 'porque',
        'pues', 'que', 'si', 'cuando', 'como',
        
        # Pronouns
        'yo', 'tu', 'él', 'ella', 'nosotros', 'nosotras', 'vosotros',
        'vosotras', 'ellos', 'ellas', 'me', 'te', 'se', 'nos', 'os',
        'mi', 'mis', 'tu', 'tus', 'su', 'sus',
    }
    
    # Add common words to lexicon
    lexicon.update(common_words)
    
    # Add historical variants of common words
    historical_variants = {
        'auer': 'haber',
        'deuer': 'deber',
        'assi': 'así',
        'reyno': 'reino',
        'mui': 'muy',
        'fecho': 'hecho',
        'fixo': 'hijo',
        'dexar': 'dejar',
        'dixo': 'dijo',
    }
    
    for old, new in historical_variants.items():
        lexicon.add(old)
        lexicon.add(new)
    
    return lexicon

class SpanishHistoricalPostProcessor:
    """
    Post-processing class for OCR results on historical Spanish texts.
    """
    def __init__(self, lexicon: Optional[Set[str]] = None):
        """
        Initialize the post-processor.
        
        Args:
            lexicon: Lexicon of valid words
        """
        self.lexicon: Set[str] = lexicon or set()
        
        # Add common Spanish words if lexicon is empty or small
        if len(self.lexicon) < 100:
            self.lexicon = add_common_spanish_words(self.lexicon)
        
    def correct_word(self, word: str, max_edit_distance: int = 2) -> str:
        """
        Correct a word using the lexicon.
        
        Args:
            word: Word to correct
            max_edit_distance: Maximum edit distance for correction
            
        Returns:
            Corrected word
        """
        # If the word is already in the lexicon, return it
        if word.lower() in self.lexicon:
            return word
            
        # If word is empty or too short, return it as is
        if len(word) < 2:
            return word
            
        # Simple edit distance function
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
                
            if len(s2) == 0:
                return len(s1)
                
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
                
            return previous_row[-1]
            
        # Find the closest word in the lexicon
        candidates = []
        for lex_word in self.lexicon:
            # Skip words with significantly different lengths
            if abs(len(lex_word) - len(word)) > max_edit_distance:
                continue
                
            # Calculate edit distance
            distance = levenshtein_distance(word.lower(), lex_word)
            
            # Only consider words within the maximum edit distance
            if distance <= max_edit_distance:
                candidates.append((lex_word, distance))
                
        # Sort candidates by edit distance
        candidates.sort(key=lambda x: x[1])
        
        # Return the closest match if any, otherwise return the original word
        return candidates[0][0] if candidates else word
        
    def process_text(self, text: str) -> str:
        """
        Process a complete OCR text.
        
        Args:
            text: OCR text
            
        Returns:
            Processed text
        """
        # Normalize the text
        text = normalize_historical_spanish(text)
        
        # Split into words
        words = []
        for word in text.split():
            # Extract the word and its surrounding punctuation
            prefix = ""
            suffix = ""
            
            while word and not word[0].isalnum():
                prefix += word[0]
                word = word[1:]
                
            while word and not word[-1].isalnum():
                suffix = word[-1] + suffix
                word = word[:-1]
                
            # Correct the word if it's not empty
            if word:
                corrected_word = self.correct_word(word)
                words.append(prefix + corrected_word + suffix)
            else:
                words.append(prefix + suffix)
                
        # Join the words back into text
        return ' '.join(words)

# ================= PART 6: EVALUATION AND METRICS =================

def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation:
    - Convert to lowercase
    - Remove punctuation
    - Remove extra whitespace
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def character_error_rate(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (S + D + I) / N
    Where:
    S = number of substitutions
    D = number of deletions
    I = number of insertions
    N = number of characters in reference
    
    Args:
        reference: Ground truth text
        hypothesis: OCR output text
        normalize: Whether to normalize texts before comparison
        
    Returns:
        CER value (lower is better)
    """
    # Normalize texts if requested
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    # Compute Levenshtein distance
    distances = np.zeros((len(reference) + 1, len(hypothesis) + 1))
    
    # Initialize first row and column
    for i in range(len(reference) + 1):
        distances[i][0] = i
    for j in range(len(hypothesis) + 1):
        distances[0][j] = j
        
    # Fill distance matrix
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i-1] == hypothesis[j-1]:
                distances[i][j] = distances[i-1][j-1]
            else:
                substitution = distances[i-1][j-1] + 1
                insertion = distances[i][j-1] + 1
                deletion = distances[i-1][j] + 1
                distances[i][j] = min(substitution, insertion, deletion)
                
    # Levenshtein distance is the value in the bottom right corner of the matrix
    levenshtein = distances[len(reference)][len(hypothesis)]
    
    # CER is Levenshtein distance divided by reference length
    if len(reference) == 0:
        return 0.0  # Handle empty reference case
        
    return levenshtein / len(reference)

def word_error_rate(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (S + D + I) / N
    Where:
    S = number of substituted words
    D = number of deleted words
    I = number of inserted words
    N = number of words in reference
    
    Args:
        reference: Ground truth text
        hypothesis: OCR output text
        normalize: Whether to normalize texts before comparison
        
    Returns:
        WER value (lower is better)
    """
    # Normalize texts if requested
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    # Split into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Compute Levenshtein distance
    distances = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    # Initialize first row and column
    for i in range(len(ref_words) + 1):
        distances[i][0] = i
    for j in range(len(hyp_words) + 1):
        distances[0][j] = j
        
    # Fill distance matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                distances[i][j] = distances[i-1][j-1]
            else:
                substitution = distances[i-1][j-1] + 1
                insertion = distances[i][j-1] + 1
                deletion = distances[i-1][j] + 1
                distances[i][j] = min(substitution, insertion, deletion)
                
    # Levenshtein distance is the value in the bottom right corner of the matrix
    levenshtein = distances[len(ref_words)][len(hyp_words)]
    
    # WER is Levenshtein distance divided by reference length
    if len(ref_words) == 0:
        return 0.0  # Handle empty reference case
        
    return levenshtein / len(ref_words)

def calculate_historical_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate metrics for historical OCR evaluation.
    
    Args:
        reference: Ground truth text
        hypothesis: OCR output text
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Apply historical text normalization
    normalized_ref = normalize_historical_spanish(reference)
    normalized_hyp = normalize_historical_spanish(hypothesis)
    
    # Calculate standard metrics
    standard_cer = character_error_rate(reference, hypothesis)
    standard_wer = word_error_rate(reference, hypothesis)
    
    # Calculate metrics with historical normalization
    historical_cer = character_error_rate(normalized_ref, normalized_hyp)
    historical_wer = word_error_rate(normalized_ref, normalized_hyp)
    
    return {
        "standard_cer": standard_cer,
        "standard_wer": standard_wer,
        "historical_cer": historical_cer,
        "historical_wer": historical_wer,
    }

def evaluate_ocr_by_document(predictions: List[str], references: List[str], doc_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate OCR results grouped by document type.
    
    Args:
        predictions: List of OCR predictions
        references: List of ground truth texts
        doc_ids: List of document IDs corresponding to predictions and references
        
    Returns:
        Dictionary with evaluation results by document type
    """
    # Group results by document type
    doc_results: Dict[str, Dict[str, List[Any]]] = {}
    
    for pred, ref, doc_id in zip(predictions, references, doc_ids):
        # Extract document type from document ID
        doc_type = doc_id.split('_')[0] if '_' in doc_id else doc_id
        
        if doc_type not in doc_results:
            doc_results[doc_type] = {
                "predictions": [],
                "references": []
            }
            
        doc_results[doc_type]["predictions"].append(pred)
        doc_results[doc_type]["references"].append(ref)
    
    # Calculate metrics for each document type
    evaluation_results: Dict[str, Dict[str, float]] = {}
    
    for doc_type, results in doc_results.items():
        # Calculate average CER and WER for this document type
        cer_values = []
        wer_values = []
        historical_cer_values = []
        historical_wer_values = []
        
        for pred, ref in zip(results["predictions"], results["references"]):
            metrics = calculate_historical_metrics(ref, pred)
            
            cer_values.append(metrics["standard_cer"])
            wer_values.append(metrics["standard_wer"])
            historical_cer_values.append(metrics["historical_cer"])
            historical_wer_values.append(metrics["historical_wer"])
            
        # Calculate averages
        evaluation_results[doc_type] = {
            "cer": sum(cer_values) / len(cer_values) if cer_values else 0,
            "wer": sum(wer_values) / len(wer_values) if wer_values else 0,
            "historical_cer": sum(historical_cer_values) / len(historical_cer_values) if historical_cer_values else 0,
            "historical_wer": sum(historical_wer_values) / len(historical_wer_values) if historical_wer_values else 0,
            "sample_count": len(cer_values)
        }
    
    return evaluation_results

def visualize_ocr_results(evaluation_results: Dict[str, Dict[str, float]], output_path: Optional[str] = None) -> None:
    """
    Visualize OCR results by document type.
    
    Args:
        evaluation_results: Dictionary with evaluation results by document type
        output_path: Path to save the visualization (optional)
    """
    # Extract data for plotting
    doc_types = list(evaluation_results.keys())
    
    if not doc_types:
        print("No results to visualize")
        return
        
    cer_values = [evaluation_results[doc]["cer"] for doc in doc_types]
    wer_values = [evaluation_results[doc]["wer"] for doc in doc_types]
    hist_cer_values = [evaluation_results[doc].get("historical_cer", 0) for doc in doc_types]
    hist_wer_values = [evaluation_results[doc].get("historical_wer", 0) for doc in doc_types]
    
    # Create figure with two subplots (CER and WER)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar width
    width = 0.35
    
    # Set x positions
    x = np.arange(len(doc_types))
    
    # Plot CER
    ax1.bar(x - width/2, cer_values, width, label='Standard CER', color='skyblue')
    ax1.bar(x + width/2, hist_cer_values, width, label='Historical CER', color='darkblue')
    
    # Set labels and title
    ax1.set_xlabel('Document Type')
    ax1.set_ylabel('Character Error Rate')
    ax1.set_title('Character Error Rate by Document Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(doc_types, rotation=45, ha='right')
    ax1.legend()
    
    # Add value labels
    for i, v in enumerate(cer_values):
        ax1.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')
    for i, v in enumerate(hist_cer_values):
        ax1.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')
    
    # Plot WER
    ax2.bar(x - width/2, wer_values, width, label='Standard WER', color='lightcoral')
    ax2.bar(x + width/2, hist_wer_values, width, label='Historical WER', color='darkred')
    
    # Set labels and title
    ax2.set_xlabel('Document Type')
    ax2.set_ylabel('Word Error Rate')
    ax2.set_title('Word Error Rate by Document Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels(doc_types, rotation=45, ha='right')
    ax2.legend()
    
    # Add value labels
    for i, v in enumerate(wer_values):
        ax2.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')
    for i, v in enumerate(hist_wer_values):
        ax2.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    # Show the figure
    plt.show()

def display_ocr_examples(predictions: List[str], 
                         references: List[str], 
                         doc_ids: List[str], 
                         image_paths: List[str], 
                         n_examples: int = 2) -> None:
    """
    Display examples of OCR results with ground truth for comparison.
    
    Args:
        predictions: List of OCR predictions
        references: List of ground truth texts
        doc_ids: List of document IDs
        image_paths: List of paths to the original images
        n_examples: Number of examples to display per document type
    """
    # Group results by document type
    doc_examples: Dict[str, List[Dict[str, Any]]] = {}
    
    for pred, ref, doc_id, img_path in zip(predictions, references, doc_ids, image_paths):
        # Extract document type
        doc_type = doc_id.split('_')[0] if '_' in doc_id else doc_id
        
        if doc_type not in doc_examples:
            doc_examples[doc_type] = []
            
        # Calculate metrics for this example
        metrics = calculate_historical_metrics(ref, pred)
        
        # Add to examples
        doc_examples[doc_type].append({
            "prediction": pred,
            "reference": ref,
            "image_path": img_path,
            "doc_id": doc_id,
            "metrics": metrics
        })
    
    # Display examples for each document type
    for doc_type, examples in doc_examples.items():
        print(f"\n=== Examples for {doc_type} ===")
        
        # Sort examples by CER (show worst examples first)
        examples.sort(key=lambda x: x["metrics"]["standard_cer"], reverse=True)
        
        # Display up to n_examples
        for i, example in enumerate(examples[:n_examples]):
            print(f"\nExample {i+1} (CER: {example['metrics']['standard_cer']:.2f}, WER: {example['metrics']['standard_wer']:.2f}):")
            
            # Display the image
            try:
                img = cv2.imread(example["image_path"])
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img_rgb)
                    plt.title(f"{doc_type} - {example['doc_id']}")
                    plt.axis('off')
                    plt.show()
                else:
                    print(f"Warning: Could not load image {example['image_path']}")
            except Exception as e:
                print(f"Error displaying image {example['image_path']}: {e}")
            
            # Display ground truth and prediction
            print("\nGround Truth:")
            print("-" * 50)
            print(example["reference"][:500] + ("..." if len(example["reference"]) > 500 else ""))
            
            print("\nOCR Prediction:")
            print("-" * 50)
            print(example["prediction"][:500] + ("..." if len(example["prediction"]) > 500 else ""))
            
            # Display differences
            gt_words = normalize_text(example["reference"]).split()
            pred_words = normalize_text(example["prediction"]).split()
            
            d = difflib.Differ()
            diff = list(d.compare(gt_words[:50], pred_words[:50]))
            
            print("\nWord Differences (first 50 words):")
            print("-" * 50)
            print(' '.join(diff))
            
            print("\n" + "=" * 80 + "\n")

# ================= PART 7: TROCR MODEL FUNCTIONS =================

def prepare_dataset(batch: Dict[str, Any], processor: Any) -> Dict[str, torch.Tensor]:
    """
    Prepare batch data for TrOCR model training.
    
    Args:
        batch: Batch of data from the dataset
        processor: TrOCR processor
        
    Returns:
        Processed batch with pixel_values and labels
    """
    # Handle images with proper format conversion
    raw_images = batch["image"]
    processed_images = []
    
    for img in raw_images:
        # Convert from numpy array to PIL Image
        # Handle different possible formats
        if isinstance(img, np.ndarray):
            if img.ndim == 3:  # RGB image
                pil_img = Image.fromarray(img.astype('uint8'))
            elif img.ndim == 2:  # Grayscale image
                pil_img = Image.fromarray(np.repeat(img[:, :, np.newaxis], 3, axis=2).astype('uint8'))
            else:
                # Create a blank image as fallback
                pil_img = Image.new('RGB', (384, 384), color='white')
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            # Last resort - create a blank image
            pil_img = Image.new('RGB', (384, 384), color='white')
            
        processed_images.append(pil_img)
        
    # Process the images with the TrOCR processor
    try:
        pixel_values = processor(images=processed_images, return_tensors="pt").pixel_values
    except Exception as e:
        print(f"Error processing images: {e}")
        # Create dummy tensor as fallback
        pixel_values = torch.zeros((len(processed_images), 3, 384, 384))
    
    # Tokenize the texts
    try:
        labels = processor.tokenizer(batch["text"], padding="max_length", truncation=True).input_ids
    except Exception as e:
        print(f"Error tokenizing texts: {e}")
        # Create dummy labels as fallback
        labels = [[processor.tokenizer.pad_token_id] * 10] * len(processed_images)
        
    return {"pixel_values": pixel_values, "labels": labels}

def convert_dataloader_to_dataset(data_loader: DataLoader) -> HFDataset:
    """
    Convert PyTorch DataLoader to HuggingFace Dataset.
    
    Args:
        data_loader: PyTorch DataLoader
        
    Returns:
        HuggingFace Dataset
    """
    all_data = []
    
    for batch in data_loader:
        for i in range(len(batch["image"])):
            # Convert tensor to numpy safely
            if torch.is_tensor(batch["image"][i]):
                # Transpose from [C, H, W] to [H, W, C] if needed
                if batch["image"][i].dim() == 3 and batch["image"][i].shape[0] == 3:
                    img_np = batch["image"][i].permute(1, 2, 0).numpy()
                else:
                    img_np = batch["image"][i].numpy()
                    
                # Normalize to 0-255 range for PIL compatibility
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype('uint8')
            else:
                # Just in case it's already a numpy array
                img_np = batch["image"][i]
                
            item = {
                "image": img_np,
                "text": batch["text"][i],
                "image_path": batch["image_path"][i]
            }
            all_data.append(item)
            
    # Create the dataset
    return HFDataset.from_list(all_data)

def evaluate_trocr_model(model: Any, 
                         processor: Any, 
                         val_loader: DataLoader, 
                         post_processor: Optional[SpanishHistoricalPostProcessor] = None, 
                         device: str = "cpu") -> Dict[str, Any]:
    """
    Evaluate a TrOCR model on a validation set.
    
    Args:
        model: TrOCR model
        processor: TrOCR processor
        val_loader: Validation data loader
        post_processor: Optional post-processor for Spanish historical text
        device: Device to run the model on
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    cer_scores = []
    wer_scores = []
    
    # Group results by document type
    cer_by_doc = {}
    wer_by_doc = {}
    doc_samples = {}
    
    all_predictions = []
    all_references = []
    all_image_paths = []
    all_doc_ids = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating model"):
            # Get images and texts
            images = batch["image"].to(device)
            texts = batch["text"]
            image_paths = batch["image_path"]
            
            # Generate predictions
            try:
                generated_ids = model.generate(
                    processor(images=images, return_tensors="pt").pixel_values.to(device)
                )
                
                # Decode predictions
                preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            except Exception as e:
                print(f"Error generating predictions: {e}")
                # Create empty predictions as fallback
                preds = [""] * len(texts)
            
            # Apply post-processing if available
            if post_processor:
                preds = [post_processor.process_text(p) for p in preds]
                
            # Store predictions and references for later analysis
            all_predictions.extend(preds)
            all_references.extend(texts)
            all_image_paths.extend(image_paths)
            
            # Extract document IDs from image paths
            doc_ids = [os.path.basename(os.path.dirname(path)) for path in image_paths]
            all_doc_ids.extend(doc_ids)
            
            # Calculate metrics for each sample
            for i, (pred, ref, doc_id) in enumerate(zip(preds, texts, doc_ids)):
                cer = character_error_rate(ref, pred)
                wer = word_error_rate(ref, pred)
                
                cer_scores.append(cer)
                wer_scores.append(wer)
                
                # Add to document-specific results
                if doc_id not in cer_by_doc:
                    cer_by_doc[doc_id] = []
                    wer_by_doc[doc_id] = []
                    doc_samples[doc_id] = []
                    
                cer_by_doc[doc_id].append(cer)
                wer_by_doc[doc_id].append(wer)
                
                # Store a sample prediction for each document type
                if len(doc_samples[doc_id]) < 2:  # Keep max 2 samples per type
                    doc_samples[doc_id].append({
                        "reference": ref,
                        "prediction": pred,
                        "cer": cer,
                        "wer": wer,
                        "image_path": image_paths[i]
                    })
                    
    # Calculate average metrics
    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0
    
    # Calculate average metrics by document type
    avg_cer_by_doc = {doc: sum(scores) / len(scores) if scores else 0 
                      for doc, scores in cer_by_doc.items()}
    avg_wer_by_doc = {doc: sum(scores) / len(scores) if scores else 0 
                      for doc, scores in wer_by_doc.items()}
                      
    # Get results by document type
    doc_results = evaluate_ocr_by_document(all_predictions, all_references, all_doc_ids)
    
    # Prepare results
    results = {
        "avg_cer": avg_cer,
        "avg_wer": avg_wer,
        "cer_by_doc": avg_cer_by_doc,
        "wer_by_doc": avg_wer_by_doc,
        "samples": doc_samples,
        "doc_results": doc_results,
        "predictions": all_predictions,
        "references": all_references,
        "image_paths": all_image_paths,
        "doc_ids": all_doc_ids
    }
    
    # Display some sample predictions
    print(f"\nOverall metrics - CER: {avg_cer:.4f}, WER: {avg_wer:.4f}")
    print("\nMetrics by document type:")
    for doc_id, cer in avg_cer_by_doc.items():
        print(f"  {doc_id} - CER: {cer:.4f}, WER: {avg_wer_by_doc[doc_id]:.4f}")
        
    print("\nSample predictions:")
    for i in range(min(3, len(all_predictions))):
        print(f"\nDocument: {all_doc_ids[i]}")
        print(f"Reference: {all_references[i][:100]}...")
        print(f"Prediction: {all_predictions[i][:100]}...")
    
    return results

def fine_tune_trocr_model(train_loader: DataLoader, 
                          val_loader: DataLoader, 
                          output_dir: str, 
                          num_epochs: int = 3) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Fine-tune a TrOCR model on custom data with detailed progress tracking.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Directory to save the fine-tuned model
        num_epochs: Number of training epochs
        
    Returns:
        Tuple of (model, processor, results)
    """
    import pandas as pd
    from IPython.display import display, HTML, clear_output
    import time

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processor and model
    print("Loading TrOCR model and processor...")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative model...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Set token IDs
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    
    # Move model to device
    model.to(device)
    
    # Convert PyTorch DataLoaders to datasets for training
    print("Converting data for training...")
    train_dataset = convert_dataloader_to_dataset(train_loader)
    val_dataset = convert_dataloader_to_dataset(val_loader)
    
    print(f"Created training dataset with {len(train_dataset)} samples")
    print(f"Created validation dataset with {len(val_dataset)} samples")
    
    # Use a partial function to include the processor
    from functools import partial
    prepare_with_processor = partial(prepare_dataset, processor=processor)
    
    train_dataset = train_dataset.map(
        prepare_with_processor,
        batched=True,
        batch_size=4,
        remove_columns=["image", "image_path"]
    )
    
    val_dataset = val_dataset.map(
        prepare_with_processor,
        batched=True,
        batch_size=4,
        remove_columns=["image", "image_path"]
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Initialize metrics tracking
    metrics_history = []
    best_cer = float('inf')
    best_model_state = None
    
    # Create a styled metrics table
    def display_metrics_table(metrics_history: List[Dict[str, Any]]) -> None:
        if not metrics_history:
            return
            
        df = pd.DataFrame(metrics_history)
        
        # Apply formatting
        for col in df.columns:
            if col not in ['epoch', 'time']:
                df[col] = df[col].apply(lambda x: f"{x:.4f}")
        if 'time' in df.columns:
            df['time'] = df['time'].apply(lambda x: f"{x:.2f}s")
            
        # Display styled table
        display(HTML(f"""
        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            <h3>Training Progress</h3>
            {df.to_html(index=False, border=0)}
        </div>
        """))
    
    # Create sample prediction display
    def display_sample_predictions(model: Any, processor: Any, val_loader: DataLoader, device: str) -> None:
        # Get a batch from validation set
        batch = next(iter(val_loader))
        images = batch["image"][:2].to(device)  # Just use 2 images
        texts = batch["text"][:2]
        
        # Generate predictions
        with torch.no_grad():
            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Display results
        display(HTML(f"""
        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
            <h3>Sample Predictions</h3>
            <p><strong>Reference:</strong> {texts[0][:100]}...</p>
            <p><strong>Prediction:</strong> {preds[0][:100]}...</p>
        </div>
        """))
    
    # Main training loop
    print(f"\nFine-tuning for {num_epochs} epochs:")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        # Progress bar for training
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_progress:
            # Move images to device
            images = batch["image"].to(device)
            
            # Tokenize texts
            tokenized = processor.tokenizer(
                batch["text"], 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            labels = tokenized.input_ids.to(device)
            
            # Forward pass
            try:
                outputs = model(
                    pixel_values=processor(images=images, return_tensors="pt").pixel_values.to(device),
                    labels=labels
                )
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                train_steps += 1
                
                # Update progress bar description
                avg_loss = train_loss / train_steps
                train_progress.set_description(f"Epoch {epoch+1}/{num_epochs} [Train: loss={avg_loss:.4f}]")
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        # Evaluation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        all_preds = []
        all_refs = []
        
        # Progress bar for evaluation
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validate]")
        with torch.no_grad():
            for batch in val_progress:
                # Move images to device
                images = batch["image"].to(device)
                texts = batch["text"]
                
                # Tokenize texts
                tokenized = processor.tokenizer(
                    texts, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                )
                labels = tokenized.input_ids.to(device)
                
                try:
                    # Forward pass
                    pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    
                    # Generate predictions
                    generated_ids = model.generate(pixel_values)
                    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_steps += 1
                    all_preds.extend(preds)
                    all_refs.extend(texts)
                    
                    # Update progress bar description
                    avg_loss = val_loss / val_steps
                    val_progress.set_description(f"Epoch {epoch+1}/{num_epochs} [Validate: loss={avg_loss:.4f}]")
                except Exception as e:
                    print(f"Error in validation step: {e}")
                    continue
        
        # Calculate CER and WER
        cer_values = [character_error_rate(ref, pred) for ref, pred in zip(all_refs, all_preds)]
        wer_values = [word_error_rate(ref, pred) for ref, pred in zip(all_refs, all_preds)]
        
        avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0
        avg_wer = sum(wer_values) / len(wer_values) if wer_values else 0
        
        # Save metrics
        epoch_time = time.time() - epoch_start_time
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / train_steps if train_steps else 0,
            'val_loss': val_loss / val_steps if val_steps else 0,
            'cer': avg_cer,
            'wer': avg_wer,
            'time': epoch_time
        }
        metrics_history.append(epoch_metrics)
        
        # Check if this is the best model
        if avg_cer < best_cer:
            best_cer = avg_cer
            best_model_state = model.state_dict().copy()
            
            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f"checkpoint-best")
            os.makedirs(checkpoint_path, exist_ok=True)
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)
            print(f"✓ New best model saved (CER: {best_cer:.4f})")
        
        # Display updated metrics and sample predictions
        clear_output(wait=True)
        print(f"Fine-tuning TrOCR model: {epoch+1}/{num_epochs} epochs completed")
        display_metrics_table(metrics_history)
        display_sample_predictions(model, processor, val_loader, device)
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with CER: {best_cer:.4f}")
    
    # Save final model
    final_path = os.path.join(output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Model saved to {final_path}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    results = evaluate_trocr_model(model, processor, val_loader, device=device)
    
    # Save evaluation results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    
    with open(results_file, 'w') as f:
        json.dump({k: v for k, v in results.items() 
                  if k not in ['samples', 'predictions', 'references', 'image_paths', 'doc_ids']}, 
                 f, indent=2)
    
    # Create visualization
    viz_path = os.path.join(output_dir, "error_rates_by_doc.png")
    visualize_ocr_results(results["doc_results"], viz_path)
    
    return model, processor, results

def run_ocr_evaluation(val_loader: DataLoader, 
                       transcriptions: Dict[str, str], 
                       output_dir: str) -> Dict[str, Any]:
    """
    Run OCR evaluation using a pre-trained TrOCR model.
    
    Args:
        val_loader: Validation data loader
        transcriptions: Dictionary mapping image paths to transcriptions
        output_dir: Directory to save evaluation results
        
    Returns:
        Evaluation results
    """
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processor and model
    print("Loading TrOCR model and processor...")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative model...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Move model to device
    model.to(device)
    
    # Create lexicon for post-processing
    print("Creating lexicon for post-processing...")
    lexicon = create_lexicon_from_transcriptions(transcriptions)
    lexicon = augment_lexicon_with_variations(lexicon)
    lexicon = add_common_spanish_words(lexicon)
    print(f"Created lexicon with {len(lexicon)} words")
    
    # Create post-processor
    post_processor = SpanishHistoricalPostProcessor(lexicon)
    
    # Run evaluation
    print("\nEvaluating pre-trained TrOCR model...")
    results = evaluate_trocr_model(model, processor, val_loader, post_processor, device)
    
    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    
    with open(results_file, 'w') as f:
        json.dump({k: v for k, v in results.items() 
                  if k not in ['samples', 'predictions', 'references', 'image_paths', 'doc_ids']},
                 f, indent=2)
    
    # Create visualization
    viz_path = os.path.join(output_dir, "error_rates_by_doc.png")
    visualize_ocr_results(results["doc_results"], viz_path)
    
    # Display some examples with images
    display_ocr_examples(
        results["predictions"],
        results["references"],
        results["doc_ids"],
        results["image_paths"],
        n_examples=2
    )
    
    return results

# ================= PART 8: MAIN PIPELINE FUNCTION =================

def run_ocr_pipeline_with_processed_images(max_pages: int = 4,  # Reduced default max pages
                                           evaluate_model: bool = True, 
                                           fine_tune: bool = False, 
                                           num_epochs: int = 2) -> Dict[str, Any]:  # Reduced default epochs
    """
    Run OCR pipeline using pre-processed images.
    
    Args:
        max_pages: Maximum number of pages to process per document (default reduced to 4)
        evaluate_model: Whether to evaluate a pre-trained model
        fine_tune: Whether to fine-tune a model
        num_epochs: Number of epochs for fine-tuning (default reduced to 2)
        
    Returns:
        Dictionary with pipeline results
    """
    # Define paths for processed and binary images
    processed_images_dir = os.path.join(output_base_path, "processed_images")
    binary_images_dir = os.path.join(output_base_path, "binary_images")
    
    print("Step 1: Loading pre-processed images")
    document_images = load_processed_images(
        processed_images_dir=processed_images_dir,
        binary_images_dir=binary_images_dir,
        max_pages=max_pages
    )
    
    # Check if we got any processed images
    total_processed = sum(len(paths) for paths in document_images.values())
    
    if total_processed == 0:
        print("\nNo images were loaded. Please check the image directories.")
        return None
        
    print(f"\nLoaded {total_processed} processed images from {len(document_images)} documents")
    
    print("\nStep 2: Loading or creating transcriptions")
    # Check if there are existing transcription files (.docx)
    existing_transcriptions = glob.glob(os.path.join(transcriptions_path, "*.docx"))
    
    if not existing_transcriptions:
        print("No existing transcriptions (.docx) found. Creating dummy transcriptions for testing.")
        create_dummy_transcriptions(document_images, transcriptions_path)
    else:
        print(f"Found {len(existing_transcriptions)} existing .docx transcription files.")
    
    # Collect all processed image paths
    print("\nStep 3: Collecting processed image paths")
    all_processed_images = []
    for doc_id, image_paths in document_images.items():
        all_processed_images.extend(image_paths)
        
    print(f"Collected {len(all_processed_images)} processed images")
    
    # Load transcriptions from .docx files
    print("\nStep 4: Loading transcriptions from .docx files")
    transcriptions = load_docx_transcriptions(transcriptions_path, all_processed_images)
    
    # Create train-validation split
    print("\nStep 5: Creating train-validation split")
    train_image_paths, val_image_paths = create_train_val_split(
        all_processed_images,
        transcriptions,
        val_ratio=0.2
    )
    
    print(f"Train set: {len(train_image_paths)} images")
    print(f"Validation set: {len(val_image_paths)} images")
    
    # Create data loaders
    print("\nStep 6: Creating data loaders")
    train_loader, val_loader = create_data_loaders(
        train_image_paths,
        val_image_paths,
        transcriptions,
        batch_size=2
    )
    
    # Show examples
    print("\nStep 7: Showing example images")
    save_example_images(document_images, output_base_path, num_examples=2)
    
    # Prepare results container
    results = {
        "document_images": document_images,
        "transcriptions": transcriptions,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_image_paths": train_image_paths,
        "val_image_paths": val_image_paths
    }
    
    # Evaluate model if requested
    if evaluate_model:
        print("\nStep 8: Evaluating pre-trained OCR model")
        eval_results = run_ocr_evaluation(
            val_loader=val_loader,
            transcriptions=transcriptions,
            output_dir=results_path
        )
        results["evaluation_results"] = eval_results
        
    # Fine-tune model if requested
    if fine_tune:
        print("\nStep 9: Fine-tuning OCR model")
        model, processor, ft_results = fine_tune_trocr_model(
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=os.path.join(output_base_path, "fine_tuned_model"),
            num_epochs=num_epochs
        )
        results["fine_tuned_model"] = model
        results["fine_tuned_processor"] = processor
        results["fine_tuning_results"] = ft_results
        
    print("\nOCR pipeline completed successfully!")
    return results

# ================= EXECUTION =================

# Print welcome message
print("=" * 80)
print("Historical Spanish Document OCR Pipeline")
print("=" * 80)
print("\nThis pipeline performs OCR on pre-processed historical Spanish document images.")
print("It will use the processed images already available in the directory structure.")

# Execute the pipeline with default settings
if __name__ == "__main__":
    # Ask for user preferences
    max_pages = int(input("\nMaximum pages to process per document (default: 4): ") or "4")
    evaluate = input("Evaluate OCR model? (y/n, default: y): ").lower() != 'n'
    fine_tune = input("Fine-tune OCR model? This may take significant time. (y/n, default: n): ").lower() == 'y'

    if fine_tune:
        num_epochs = int(input("Number of fine-tuning epochs (recommended: 2-3): ") or "2")
    else:
        num_epochs = 2

    print("\nStarting OCR pipeline with the following settings:")
    print(f"- Maximum pages per document: {max_pages}")
    print(f"- Evaluate pre-trained model: {evaluate}")
    print(f"- Fine-tune model: {fine_tune}")
    if fine_tune:
        print(f"- Fine-tuning epochs: {num_epochs}")
    print("\nExecuting pipeline...\n")

    # Run the pipeline
    results = run_ocr_pipeline_with_processed_images(
        max_pages=max_pages,
        evaluate_model=evaluate,
        fine_tune=fine_tune,
        num_epochs=num_epochs
    )
