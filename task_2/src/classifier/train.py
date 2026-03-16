import argparse

def prepare_data(data_dir, batch_size): 
    # Implement data loading and preprocessing here
    pass    

def build_model(num_classes):
    # Implement model architecture here
    pass

def train(model, dataloader, criterion, optimizer, device):
    # Implement training loop here
    pass

def main():
    parser = argparse.ArgumentParser(description="Train a classifier on the Animals-10 dataset")
    parser.add_argument('--data_dir', type=str, default='data/Animals-10', help='Path to the dataset directory')
    parser.add_argument('--model_dir', type=str, default='models/', help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    
    args = parser.parse_args()
    # 2. Викликаємо функції по черзі
    # train_loader, val_loader = prepare_data(args.data_dir, args.batch_size)
    # model = build_model(num_classes=10)
    # optimizer = ...
    # criterion = ...
    
    # 3. Тренуємо
    # train(model, train_loader, args.epochs, optimizer, criterion)
    
    # 4. Зберігаємо результат
    # torch.save(model.state_dict(), args.output_path)
    print(f"Модель успішно збережена у {args.output_path}")
    

if __name__ == "__main__":
    main()