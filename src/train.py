import json
from pathlib import Path
import argparse

import numpy as np
import tensorflow as tf

from src.data_loader import get_prices, prepare_dataset
from src.model import build_lstm_model

def load_config(path: str|Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config not found: {path}')
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)
    
def main(config_path: str) -> None:
    configs = load_config(config_path)
    
    # Data configs
    ticker = configs['data']['ticker']
    start_date = configs['data']['start_date']
    end_date = configs['data']['end_date']

    # Model configs
    train_split = configs['model']['train_split']
    lookback = configs['model']['lookback']
    units1 = configs['model']['units1']
    units2 = configs['model']['units2']
    dropout = configs['model']['dropout']
    learning_rate = configs['model']['learning_rate']

    # Training configs
    epochs = configs['training']['epochs']
    batch_size = configs['training']['batch_size']

    print(f'Configs successfully loaded from {config_path}!')

    # Set seed for reproducibility 
    np.random.seed(42)
    tf.random.set_seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.json', help='Path to JSON config.')
    args = parser.parse_args()
    main(args.config)