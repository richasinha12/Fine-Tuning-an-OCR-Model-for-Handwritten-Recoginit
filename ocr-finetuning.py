import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AdamW, get_scheduler
from datasets import load_dataset
from PIL import Image, ImageOps
import cv2
from tqdm.auto import tqdm
import evaluate
from sklearn.model_selection import train_test_split

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Mixed precision setup
scaler = GradScaler()

# Load model and processor
model_name = "microsoft/trocr-large-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Move model to GPU(s)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)
model.to(device)

# Dataset constants
MAX_LENGTH = 128  # Maximum sequence length
IMAGE_SIZE = (384, 384)  # Input image size for TrOCR

class HandwrittenTextDataset(Dataset):
    def __init__(self, images, texts, processor, max_length=MAX_LENGTH, image_size=IMAGE_SIZE):
        self.images = images
        self.texts = texts
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def preprocess_image(self, image_path):
        """Preprocess image with OpenCV for better results"""
        if isinstance(image_path, str):
            # Load image from path
            image = cv2.imread(image_path)
            if image is None:
                # Fallback to PIL if OpenCV fails
                return Image.open(image_path).convert("RGB").resize(self.image_size)
        else:
            # Handle PIL Image or other formats
            if hasattr(image_path, 'convert'):
                return image_path.convert("RGB").resize(self.image_size)
            # Convert numpy array to PIL
            image = image_path
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # Convert back to RGB (3 channels)
        rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb)
        
        return pil_image

    def __getitem__(self, idx):
        image = self.preprocess_image(self.images[idx])
        text = self.texts[idx]
        
        # Prepare inputs for the model
        pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Prepare labels
        labels = processor.tokenizer(text, 
                                   padding="max_length", 
                                   max_length=self.max_length, 
                                   truncation=True, 
                                   return_tensors="pt").input_ids.squeeze()
        
        return {"pixel_values": pixel_values, "labels": labels, "text": text}

def load_iam_dataset():
    """Load and prepare IAM dataset"""
    print("Loading IAM dataset...")
    # Using Hugging Face's datasets library
    dataset = load_dataset("iamdataset/IAM", "lines")
    train_data = dataset["train"]
    
    # Extract images and texts
    images = [sample["image"] for sample in train_data]
    texts = [sample["text"] for sample in train_data]
    
    print(f"Loaded {len(images)} samples from IAM dataset")
    return images, texts

def load_imgur5k_dataset():
    """
    Load and prepare Imgur5K dataset
    Note: This is a placeholder. In a real scenario, you'd need to download 
    and process the Imgur5K dataset accordingly
    """
    print("Loading Imgur5K dataset...")
    try:
        # Attempt to load from Kaggle datasets
        # In a real implementation, you would specify the exact path in Kaggle
        dataset = load_dataset("imgur5k", split="train")
        images = [sample["image"] for sample in dataset]
        texts = [sample["text"] for sample in dataset]
        print(f"Loaded {len(images)} samples from Imgur5K dataset")
    except Exception as e:
        print(f"Could not load Imgur5K dataset: {e}")
        print("Using a small synthetic dataset instead")
        # Create a small synthetic dataset as a fallback
        images = []
        texts = []
    
    return images, texts

def prepare_datasets():
    """Prepare training, validation and test datasets"""
    # Load datasets
    iam_images, iam_texts = load_iam_dataset()
    imgur_images, imgur_texts = load_imgur5k_dataset()
    
    # Combine datasets
    all_images = iam_images + imgur_images
    all_texts = iam_texts + imgur_texts
    
    # Split into train, validation, and test sets
    train_images, temp_images, train_texts, temp_texts = train_test_split(
        all_images, all_texts, test_size=0.2, random_state=42
    )
    
    val_images, test_images, val_texts, test_texts = train_test_split(
        temp_images, temp_texts, test_size=0.5, random_state=42
    )
    
    print(f"Train set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples")
    print(f"Test set: {len(test_images)} samples")
    
    # Create datasets
    train_dataset = HandwrittenTextDataset(train_images, train_texts, processor)
    val_dataset = HandwrittenTextDataset(val_images, val_texts, processor)
    test_dataset = HandwrittenTextDataset(test_images, test_texts, processor)
    
    return train_dataset, val_dataset, test_dataset

def collate_fn(batch):
    """Custom collate function to handle different sized images"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    texts = [item["text"] for item in batch]
    
    return {"pixel_values": pixel_values, "labels": labels, "texts": texts}

def compute_metrics(pred_texts, target_texts):
    """Compute CER and WER metrics"""
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    cer = cer_metric.compute(predictions=pred_texts, references=target_texts)
    wer = wer_metric.compute(predictions=pred_texts, references=target_texts)
    
    return {"cer": cer, "wer": wer}

def validate(model, val_loader, device):
    """Run validation and return metrics"""
    model.eval()
    pred_texts = []
    target_texts = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(device)
            texts = batch["texts"]
            
            # Forward pass
            outputs = model.generate(pixel_values)
            
            # Decode predictions
            pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
            pred_texts.extend(pred_str)
            target_texts.extend(texts)
    
    # Compute metrics
    metrics = compute_metrics(pred_texts, target_texts)
    print(f"Validation CER: {metrics['cer']:.4f}, WER: {metrics['wer']:.4f}")
    
    return metrics

def train():
    """Main training function"""
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets()
    
    # Create data loaders
    batch_size = 8 if torch.cuda.device_count() > 1 else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Training hyperparameters
    learning_rate = 5e-5
    num_epochs = 10
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_cer = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                # If using DataParallel, the loss is a tensor with the mean value
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg training loss: {avg_train_loss:.4f}")
        
        # Validation
        metrics = validate(model, val_loader, device)
        
        # Save best model
        if metrics['cer'] < best_cer:
            best_cer = metrics['cer']
            patience_counter = 0
            print(f"New best CER: {best_cer:.4f} - Saving model")
            
            # Save model
            if isinstance(model, DataParallel):
                model.module.save_pretrained("./best_trocr_model")
            else:
                model.save_pretrained("./best_trocr_model")
            processor.save_pretrained("./best_trocr_processor")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    
    # Load best model for final evaluation
    best_model = VisionEncoderDecoderModel.from_pretrained("./best_trocr_model")
    best_model.to(device)
    
    if torch.cuda.device_count() > 1:
        best_model = DataParallel(best_model)
    
    test_metrics = validate(best_model, test_loader, device)
    print(f"Test metrics: CER: {test_metrics['cer']:.4f}, WER: {test_metrics['wer']:.4f}")
    
    # Check if we met the target metrics
    target_met = test_metrics['cer'] <= 0.07 and test_metrics['wer'] <= 0.15
    print(f"Target metrics achieved: {target_met}")
    
    return test_metrics

if __name__ == "__main__":
    train()
