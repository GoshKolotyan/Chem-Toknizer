import torch 
from torch.nn import MSELoss
from tokenizer import ChemTokenizer
from dataloader import ChemDataLoader
from transformer import ChemTransformer

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=100, scheduler=None):
    """
    Train the ChemTransformer model without early stopping.
    
    Args:
        model: ChemTransformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on (cuda/cpu)
        num_epochs: Number of epochs to train for
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        Trained model and training history
    """
    model.to(device)
    
    # For tracking best model
    best_val_loss = float('inf')
    best_model = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                # Calculate MAE
                mae = torch.abs(outputs - y).mean().item()
                val_mae += mae
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}')
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Still track the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            print(f"New best model found with validation loss: {best_val_loss:.4f}")
    
    # Load best model before returning
    model.load_state_dict(best_model)
    
    return model, history

# Example usage
if __name__ == "__main__":
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = ChemTokenizer()
    
    # Create data loaders
    train_loader = ChemDataLoader(
        csv_path="../Cleaned Data/train_data.csv",
        tokenizer=tokenizer,
        batch_size=16,
        shuffle=True
    )
    
    val_loader = ChemDataLoader(
        csv_path="../Cleaned Data/val_data.csv",
        tokenizer=tokenizer,
        batch_size=16,
        shuffle=False
    )
    
    # Initialize model
    model = ChemTransformer(
        input_dim=18,          # Match your feature vector length
        embedding_dim=32,      # Lightweight embedding
        num_heads=4,           # 4 attention heads
        num_layers=2,          # 2 transformer layers
        dim_feedforward=64,    # Small feedforward dimension
        dropout=0.1
    )
    
    # Loss function and optimizer
    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train model without early stopping
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        # scheduler=scheduler  # Commented out but available if needed
    )
    
    # Save model
    torch.save(model.state_dict(), "chem_transformer_model.pth")
    
    print("Training completed and model saved!")