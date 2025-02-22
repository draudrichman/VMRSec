# main.py
from train import main as train_main
from eval import main as eval_main

if __name__ == '__main__':
    # Train the model
    train_main()
    
    # Evaluate the model
    eval_main()