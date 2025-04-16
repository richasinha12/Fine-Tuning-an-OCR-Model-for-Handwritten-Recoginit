import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import evaluate
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns

def load_model(model_path="./best_trocr_model", processor_path="./best_trocr_processor"):
    """Load the fine-tuned model and processor"""
    processor = TrOCRProcessor.from_pretrained(processor_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, processor, device

def preprocess_image(image_path, image_size=(384, 384)):
    """Preprocess a single image for inference"""
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = np.array(image_path)
        
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, image_size)
    
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

def recognize_text(model, processor, image, device):
    """Recognize text in an image"""
    # Preprocess image
    if isinstance(image, str) or not hasattr(image, 'convert'):
        image = preprocess_image(image)
    
    # Prepare input
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Generate prediction
    generated_ids = model.generate(pixel_values)
    
    # Decode prediction
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return predicted_text

def evaluate_model(model, processor, test_dataset, device):
    """Evaluate model performance on a test dataset"""
    model.eval()
    predictions = []
    references = []
    
    # Process each sample
    for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
        sample = test_dataset[idx]
        image = sample["pixel_values"]  # This should be a tensor
        
        # Convert tensor back to PIL image for preprocessing
        image = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        
        # Recognize text
        predicted_text = recognize_text(model, processor, image, device)
        predictions.append(predicted_text)
        references.append(sample["text"])
    
    # Calculate metrics
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    cer = cer_metric.compute(predictions=predictions, references=references)
    wer = wer_metric.compute(predictions=predictions, references=references)
    
    # Sample-wise errors for analysis
    sample_metrics = []
    for pred, ref in zip(predictions, references):
        sample_cer = cer_metric.compute(predictions=[pred], references=[ref])
        sample_wer = wer_metric.compute(predictions=[pred], references=[ref])
        sample_metrics.append({
            'prediction': pred,
            'reference': ref,
            'cer': sample_cer,
            'wer': sample_wer
        })
    
    return {
        'overall_cer': cer,
        'overall_wer': wer,
        'sample_metrics': sample_metrics
    }

def visualize_results(results, num_samples=5):
    """Visualize evaluation results"""
    # Sort samples by CER
    sample_metrics = sorted(results['sample_metrics'], key=lambda x: x['cer'])
    
    # Plot CER distribution
    plt.figure(figsize=(10, 6))
    cers = [m['cer'] for m in results['sample_metrics']]
    plt.hist(cers, bins=20, alpha=0.7)
    plt.axvline(results['overall_cer'], color='red', linestyle='dashed', linewidth=2)
    plt.title(f'CER Distribution (Average: {results["overall_cer"]:.4f})')
    plt.xlabel('Character Error Rate')
    plt.ylabel('Number of Samples')
    plt.savefig('cer_distribution.png')
    plt.close()
    
    # Plot WER distribution
    plt.figure(figsize=(10, 6))
    wers = [m['wer'] for m in results['sample_metrics']]
    plt.hist(wers, bins=20, alpha=0.7)
    plt.axvline(results['overall_wer'], color='red', linestyle='dashed', linewidth=2)
    plt.title(f'WER Distribution (Average: {results["overall_wer"]:.4f})')
    plt.xlabel('Word Error Rate')
    plt.ylabel('Number of Samples')
    plt.savefig('wer_distribution.png')
    plt.close()
    
    # Show best and worst samples
    best_samples = sample_metrics[:num_samples]
    worst_samples = sample_metrics[-num_samples:]
    
    # Create a DataFrame for best samples
    best_df = pd.DataFrame(best_samples)
    best_df = best_df[['reference', 'prediction', 'cer', 'wer']]
    best_df.to_csv('best_samples.csv', index=False)
    
    # Create a DataFrame for worst samples
    worst_df = pd.DataFrame(worst_samples)
    worst_df = worst_df[['reference', 'prediction', 'cer', 'wer']]
    worst_df.to_csv('worst_samples.csv', index=False)
    
    # Return overall metrics as a DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['CER', 'WER'],
        'Value': [results['overall_cer'], results['overall_wer']],
        'Target': [0.07, 0.15],
        'Achieved': [results['overall_cer'] <= 0.07, results['overall_wer'] <= 0.15]
    })
    
    return metrics_df

def demo_inference(model, processor, device, image_path):
    """Run inference on a single image and visualize results"""
    # Load and preprocess image
    image = preprocess_image(image_path)
    
    # Recognize text
    predicted_text = recognize_text(model, processor, image, device)
    
    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(Image.open(image_path))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Preprocessed Image")
    plt.imshow(image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_image_comparison.png')
    plt.close()
    
    return {
        'original_image': image_path,
        'preprocessed_image': 'demo_image_comparison.png',
        'predicted_text': predicted_text
    }

if __name__ == "__main__":
    # Load model
    model, processor, device = load_model()
    
    # You would need to load your test dataset here
    # For demonstration, we'll just print a message
    print("Model loaded successfully. Ready for inference!")
    print("This script can be imported to evaluate the model on your test dataset.")
