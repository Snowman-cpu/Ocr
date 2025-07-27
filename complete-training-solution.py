# Complete training script with proper progress bars

# Import libraries
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from IPython.display import display, HTML
import time
import warnings
warnings.filterwarnings("ignore")

# Define paths
base_path = '/content'
output_base_path = os.path.join(base_path, 'ocr_data')
processed_images_dir = os.path.join(output_base_path, "processed_images")
transcriptions_path = os.path.join(base_path, 'transcriptions')
fine_tuned_model_dir = os.path.join(output_base_path, "fine_tuned_model")
os.makedirs(fine_tuned_model_dir, exist_ok=True)

# Find processed images
print("Finding processed images...")
all_processed_images = []
for doc_dir in os.listdir(processed_images_dir):
    doc_path = os.path.join(processed_images_dir, doc_dir)
    if os.path.isdir(doc_path):
        for img_file in os.listdir(doc_path):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                img_path = os.path.join(doc_path, img_file)
                all_processed_images.append(img_path)
print(f"Found {len(all_processed_images)} processed images")

# Load transcriptions
print("Loading transcriptions...")
transcriptions = {}
for doc_dir in os.listdir(processed_images_dir):
    doc_path = os.path.join(processed_images_dir, doc_dir)
    if os.path.isdir(doc_path):
        transcription_file = os.path.join(transcriptions_path, f"{doc_dir}.txt")
        if os.path.exists(transcription_file):
            with open(transcription_file, 'r', encoding='utf-8') as f:
                transcription = f.read()
            
            # Add transcription for all images from this document
            for img_file in os.listdir(doc_path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img_path = os.path.join(doc_path, img_file)
                    transcriptions[img_path] = transcription
print(f"Loaded {len(transcriptions)} transcriptions")

# Create train-val split
import random
random.seed(42)
valid_images = [img for img in all_processed_images if img in transcriptions]
random.shuffle(valid_images)
split_idx = int(len(valid_images) * 0.8)
train_images = valid_images[:split_idx]
val_images = valid_images[split_idx:]
print(f"Created split - Train: {len(train_images)}, Validation: {len(val_images)}")

# Define OCR dataset
class OCRDataset(Dataset):
    def __init__(self, image_paths, transcriptions, transform=None):
        self.image_paths = image_paths
        self.transcriptions = transcriptions
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (384, 384), color='white')
            if self.transform:
                img = self.transform(img)
        
        text = self.transcriptions.get(img_path, "")
        if len(text) > 512:
            text = text[:512]
        
        return img, text, img_path

# Define transforms
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = OCRDataset(train_images, transcriptions, transform)
val_dataset = OCRDataset(val_images, transcriptions, transform)

# Load TrOCR model - only import here to avoid potential conflicts
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
print("Loading TrOCR model and processor...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Configure model
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define metrics functions
def normalize_text(text):
    import re
    import string
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def character_error_rate(reference, hypothesis):
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    
    if len(reference) == 0:
        return 0.0
    
    # Simple edit distance calculation
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / m

def word_error_rate(reference, hypothesis):
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 0.0
    
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / m

# Define custom training loop with clear progress bars
def fine_tune_ocr(model, processor, train_dataset, val_dataset, num_epochs=4, batch_size=2, learning_rate=5e-5):
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Track metrics
    all_metrics = []
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Display epoch banner
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Training", leave=True)
        
        for batch_idx, (images, texts, _) in enumerate(train_bar):
            # Move images to device
            images = images.to(device)
            
            # Tokenize texts
            tokenized = processor.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")
            labels = tokenized.input_ids.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += loss.item()
            avg_loss = train_loss / (batch_idx + 1)
            train_bar.set_description(f"Training (loss: {avg_loss:.4f})")
        
        # Evaluation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        print("\nEvaluating...")
        val_bar = tqdm(val_loader, desc="Evaluating", leave=True)
        
        with torch.no_grad():
            for batch_idx, (images, texts, _) in enumerate(val_bar):
                # Move images to device
                images = images.to(device)
                
                # Tokenize texts
                tokenized = processor.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")
                labels = tokenized.input_ids.to(device)
                
                # Forward pass
                outputs = model(pixel_values=images, labels=labels)
                val_loss += outputs.loss.item()
                
                # Generate predictions
                generated_ids = model.generate(images)
                
                # Decode predictions and labels
                pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                label_texts = texts
                
                all_preds.extend(pred_texts)
                all_labels.extend(label_texts)
                
                # Update progress bar
                avg_loss = val_loss / (batch_idx + 1)
                val_bar.set_description(f"Evaluating (loss: {avg_loss:.4f})")
        
        # Calculate metrics
        cer_scores = [character_error_rate(label, pred) for label, pred in zip(all_labels, all_preds)]
        wer_scores = [word_error_rate(label, pred) for label, pred in zip(all_labels, all_preds)]
        
        avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0
        avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0
        avg_val_loss = val_loss / len(val_loader) if val_loader else 0
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Store metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': avg_val_loss,
            'cer': avg_cer,
            'wer': avg_wer,
            'accuracy': 1 - avg_cer,
            'time': epoch_time
        }
        all_metrics.append(metrics)
        
        # Create metrics table
        df = pd.DataFrame(all_metrics)
        for col in df.columns:
            if col not in ['epoch', 'time']:
                df[col] = df[col].apply(lambda x: f"{x:.4f}")
        df['time'] = df['time'].apply(lambda x: f"{x:.2f}s")
        
        # Display metrics
        print("\nTraining metrics:")
        display(HTML(df.to_html(index=False)))
        
        # Print some example predictions
        print("\nSample predictions:")
        for i in range(min(3, len(all_preds))):
            print(f"  Label: {all_labels[i][:50]}...")
            print(f"  Pred:  {all_preds[i][:50]}...")
            print(f"  CER: {cer_scores[i]:.4f}, WER: {wer_scores[i]:.4f}")
            print("")
        
        print(f"{'='*80}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(fine_tuned_model_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
    
    # Save final model
    final_path = os.path.join(fine_tuned_model_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    print("\nTraining completed! Model saved to:", final_path)
    return model, processor, all_metrics

# Run the training with proper progress tracking
print("Starting fine-tuning with visible progress tracking...")
model, processor, metrics = fine_tune_ocr(
    model=model,
    processor=processor,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=4,
    batch_size=2
)