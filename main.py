# main.py
from tools.train import main as train_main
from tools.eval import main as eval_main

if __name__ == '__main__':
    # Train the model
    train_main()
    
    # Evaluate the model
    eval_main()