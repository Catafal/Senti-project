import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Part 1: Data Preprocessing
def load_and_preprocess_data(dataset_path, test_path=None):
    """
    Load and preprocess the dataset
    
    Args:
        dataset_path: Path to the main dataset CSV
        test_path: Path to the test dataset to submit (optional)
    
    Returns:
        X: Features for training data
        y: Labels for training data
        X_submit: Features for submission data (if test_path provided)
        scaler: Fitted StandardScaler object
    """
    
    df = pd.read_csv(dataset_path)
    
    
    X = df.drop('emotion', axis=1)
    y = df['emotion']
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    
    if test_path:
        df_submit = pd.read_csv(test_path)
        X_submit = scaler.transform(df_submit)
        X_submit = pd.DataFrame(X_submit, columns=df_submit.columns)
    else:
        X_submit = None
    
    return X_scaled, y, X_submit, scaler



# Part 2: Data Splitting
def create_balanced_split(X, y, val_size=0.15, test_size=0.15, random_state=27):
    """
    Create balanced train/validation/test splits
    
    Args:
        X: Feature matrix
        y: Target labels
        val_size: Size of validation set (proportion of training set)
        test_size: Size of test set (proportion of total dataset)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, X_test: Split feature matrices
        y_train, y_val, y_test: Split target labels
    """
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size/(1-test_size),  
        stratify=y_temp,
        random_state=random_state
    )
    
    print("\nData split summary:")
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    


    print("\nClass distribution:")
    print("\nTraining set:")
    print(y_train.value_counts(normalize=True))
    print("\nValidation set:")
    print(y_val.value_counts(normalize=True))
    print("\nTest set:")
    print(y_test.value_counts(normalize=True))
    
    return X_train, X_val, X_test, y_train, y_val, y_test



# Part 3: Model Training
class EmotionDataset(Dataset):
    """Custom Dataset for emotion classification task"""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X.values)
        if y is not None:
            # Convert labels to numerical format
            self.y = torch.LongTensor(pd.Categorical(y).codes)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super().__init__()
        
        # Build a list of layers dynamically
        layers = []
        
        # Input layer
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15):
    """Train the model and return training history"""
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print('-' * 60)
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Create data loaders
def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    train_dataset = EmotionDataset(X_train, y_train)
    val_dataset = EmotionDataset(X_val, y_val)
    test_dataset = EmotionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

# Model initialization
def initialize_model(input_size, num_classes, device):
    model = EmotionClassifier(
        input_size=input_size,
        hidden_sizes=[512, 256, 128],  # You can adjust these sizes
        num_classes=num_classes,
        dropout_rate=0.3
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    return model, criterion, optimizer



# Part 4: Hyperparameter Tuning
def perform_hyperparameter_tuning(X_train, X_val, y_train, y_val, device, num_epochs=15):
    """
    Test different model architectures and return the best configuration
    """
    # Configurations to test
    configurations = [
        # (hidden_layers, dropout_rate)
        ([64, 32], 0.2),              # Small network, low dropout
        ([128, 64], 0.2),             # Medium network, low dropout
        ([256, 128, 64], 0.2),        # Large network, low dropout
        ([64, 32], 0.4),              # Small network, high dropout
        ([128, 64], 0.4),             # Medium network, high dropout
        ([256, 128, 64], 0.4),        # Large network, high dropout
    ]
    
    results = []
    
    for hidden_sizes, dropout_rate in configurations:
        print(f"\nTesting configuration:")
        print(f"Hidden layers: {hidden_sizes}")
        print(f"Dropout rate: {dropout_rate}")
        print("-" * 50)
        
        # Create data loaders
        train_loader, val_loader, _ = create_data_loaders(
            X_train, X_val, X_val,  
            y_train, y_val, y_val, 
            batch_size=48
        )
        
       
        model = EmotionClassifier(
            input_size=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            num_classes=len(y_train.unique()),
            dropout_rate=dropout_rate
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        
        _, _, val_accuracies = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs
        )
        
        
        best_val_accuracy = max(val_accuracies)
        results.append({
            'hidden_sizes': hidden_sizes,
            'dropout_rate': dropout_rate,
            'best_val_accuracy': best_val_accuracy
        })
        
        print(f"\nBest validation accuracy: {best_val_accuracy:.2f}%")
    
    # Find the best configuration
    best_config = max(results, key=lambda x: x['best_val_accuracy'])
    
    print("\n" + "="*50)
    print("Best configuration found:")
    print(f"Hidden layers: {best_config['hidden_sizes']}")
    print(f"Dropout rate: {best_config['dropout_rate']}")
    print(f"Validation accuracy: {best_config['best_val_accuracy']:.2f}%")
    
    return best_config



# Main function
def main_train():
    """Main function for training and hyperparameter tuning"""
    X, y, X_submit, scaler = load_and_preprocess_data(
        dataset_path='dataset.csv',
        test_path='test_to_submit.csv'
    )
    X_train, X_val, X_test, y_train, y_val, y_test = create_balanced_split(X, y)
    
    print("\nData split summary:")
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    print("\nClass distribution:")
    print("\nTraining set:")
    print(y_train.value_counts(normalize=True))
    print("\nValidation set:")
    print(y_val.value_counts(normalize=True))
    print("\nTest set:")
    print(y_test.value_counts(normalize=True))
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    print(f"\nUsing device: {device}")
    
    # Your hyperparameter tuning code
    best_config = perform_hyperparameter_tuning(X_train, X_val, y_train, y_val, device)
    
    print("\nTraining final model with best configuration for test set evaluation...")
    final_model = EmotionClassifier(
        input_size=X_train.shape[1],
        hidden_sizes=best_config['hidden_sizes'],
        num_classes=len(y_train.unique()),
        dropout_rate=best_config['dropout_rate']
    ).to(device)
    
    
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        batch_size=32
    )
    
    # Train the final model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    
    train_losses, val_losses, val_accuracies = train_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=15
    )
    
    
    print("\nEvaluating final model on test set...")
    test_accuracy = evaluate_model(final_model, test_loader, device)
    print(f"\nFinal Test Set Accuracy: {test_accuracy:.2f}%")
    
    # Save the trained model and scaler
    print("\nSaving trained model and configuration...")
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'scaler_state_dict': scaler,
        'best_config': best_config,
        'test_accuracy': test_accuracy  
    }, 'trained_emotion_model.pth')
    print("Model saved successfully!")
    
    return test_accuracy  

# Prediction 
def generate_predictions(model, X_submit, device):
    """
    Generate predictions for submission data
    """
    # Create dataset and dataloader for submission data
    submit_dataset = EmotionDataset(X_submit)
    submit_loader = DataLoader(submit_dataset, batch_size=32, shuffle=False)
    
    # Set model to evaluation mode
    model.eval()
    
    # Store predictions
    all_predictions = []
    
    # Generate predictions
    with torch.no_grad():
        for inputs in submit_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    return all_predictions

def convert_numeric_to_emotion(predictions):
    """
    Convert numeric predictions back to emotion labels
    """
    # Define the mapping based on your original label encoding
    emotion_mapping = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'neutral',
        5: 'sad',
        6: 'surprise'
    }
    
    return [emotion_mapping[pred] for pred in predictions]

def main_predict():
    """Main function for generating predictions"""
    print("\nLoading trained model and configuration...")
    
    checkpoint = torch.load('trained_emotion_model.pth')
    scaler = checkpoint['scaler_state_dict']
    best_config = checkpoint['best_config']
    
    
    device = "cpu"
    # device = "mps" if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")
    
    print("\nInitializing model with best configuration:")
    print(f"Hidden layers: {best_config['hidden_sizes']}")
    print(f"Dropout rate: {best_config['dropout_rate']}")
    
    # Initialize model with saved configuration
    model = EmotionClassifier(
        input_size=20,  
        hidden_sizes=best_config['hidden_sizes'],
        num_classes=7,  
        dropout_rate=best_config['dropout_rate']
    ).to(device)
    
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nLoading and preprocessing submission data...")
    
    X_submit = pd.read_csv('test_to_submit.csv')
    print(f"Submission data shape: {X_submit.shape}")
    
    # Preprocess using loaded scaler
    X_submit_scaled = pd.DataFrame(
        scaler.transform(X_submit), 
        columns=X_submit.columns
    )
    
    print("\nGenerating predictions...")
    
    predictions_numeric = generate_predictions(model, X_submit_scaled, device)
    predictions_emotions = convert_numeric_to_emotion(predictions_numeric)
    
    
    with open('outputs', 'w') as f:
        for emotion in predictions_emotions:
            f.write(f"{emotion}\n")
    print("\nPredictions saved to 'outputs' file")
    
    print("\nPredictions distribution:")
    distribution = pd.Series(predictions_emotions).value_counts(normalize=True)
    print(distribution)
    
    
    print("\nVerifying output file format...")
    with open('outputs', 'r') as f:
        lines = f.readlines()
        print(f"Number of predictions: {len(lines)}")
        print("Sample of first 5 predictions:")
        for i, line in enumerate(lines[:5]):
            print(f"Line {i+1}: {line.strip()}")



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        print("Starting prediction process...")
        main_predict()
    else:
        print("Starting training process...")
        main_train()