from loader import Loader
from trainer import Trainer

def train_model(model_type='cnn', epochs=30, batch_size=16, learning_rate=0.001):
    loader = Loader("./dataset")
    loader.load_csv_data()
    
    X, y = loader.load_images_and_labels(target_size=(224, 224))
    
    loader.split_data(X, y)
    
    class_weights = loader.calculate_class_weights()
    
    num_classes = len(loader.class_names)
    
    if model_type in ['resnet50', 'vgg16']:
        learning_rate = learning_rate * 0.1  # lower learning rate for pretrained models
    
    trainer = Trainer(
        loader, 
        model_type=model_type,
        num_classes=num_classes, 
        learning_rate=learning_rate, 
        batch_size=batch_size,
        pretrained=True if model_type != 'cnn' else False
    )
    
    trainer.setup_data_loaders(class_weights=class_weights)
    
    print(f"\nStarting training with {num_classes} classes:")
    for i, class_name in enumerate(loader.class_names):
        print(f"  {i}: {class_name}")
    
    trainer.train(epochs=epochs, early_stopping_patience=8)
    
    y_true, y_pred = trainer.evaluate()
    
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(y_true, y_pred)
    
    print(f"\nModel trained successfully")
    
    return trainer

def main():
    model_type = 'vgg16'  # cnn, vgg16, resnet50
    
    train_model(model_type=model_type)

if __name__ == "__main__":
    main()
