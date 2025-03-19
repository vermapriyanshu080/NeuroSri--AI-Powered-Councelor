import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.ml.deep_emotion_model import EEGEmotionNet

def train_model(train_data, train_labels, epochs=10, batch_size=32):
    """Train the emotion model"""
    model = EEGEmotionNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Convert data to tensors
    X = torch.FloatTensor(train_data)
    y = torch.LongTensor(train_labels)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def main():
    # Create dummy data for initial training
    n_samples = 1000
    n_features = 110  # Match model input size
    
    # Generate random data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # Binary classification
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    model_path = project_root / "models" / "emotion_model.pth"
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 