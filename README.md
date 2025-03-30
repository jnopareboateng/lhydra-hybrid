# Lhydra-Flask: Hybrid Music Recommender System

A sophisticated music recommendation system that combines collaborative filtering with content-based features using a two-tower neural network architecture. The system processes user preferences, audio features, and temporal patterns to provide personalized music recommendations.

## Project Structure

```
Lhydra-Flask/
│
├── data/                       # Data-related modules
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset implementation
│   ├── preprocessor.py         # Data preprocessing and feature engineering
│   └── preprocessed/           # Directory for preprocessed data
│
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── hybrid_model.py         # Main hybrid recommender model
│   └── checkpoints/            # Model checkpoints
│
├── training/                   # Training-related code
│   ├── __init__.py
│   ├── trainer.py              # Model training logic
│   ├── metrics.py              # Evaluation metrics
│   └── configs/                # Training configurations
│       └── training_config.yaml # Default training configuration
│
├── inference/                  # Inference-related code
│   └── __init__.py
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── logger.py               # Comprehensive logging utilities
│
├── scripts/                    # Executable scripts
│   ├── train_model.py          # Model training script
│   ├── evaluate_model.py       # Model evaluation script
│   └── generate_recommendations.py # Generate recommendations
│
├── notebooks/                  # Jupyter notebooks
│   └── spotify_sample_data.csv # Sample data
│
├── logs/                       # Log files
├── evaluation/                 # Evaluation results
├── recommendations/            # Generated recommendations
│
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies
```

## Features

This implementation includes:

- **Hybrid recommendation approach** combining:
  - Collaborative filtering
  - Content-based features
  - Temporal patterns
- **Two-tower neural network architecture**:
  - User tower with demographics and audio preferences
  - Item tower with music features and metadata
- **Comprehensive feature engineering**:
  - Audio feature processing
  - Demographic feature integration
  - Temporal data processing
- **Advanced evaluation metrics**:
  - Standard classification metrics (accuracy, precision, recall, F1)
  - Ranking metrics (NDCG, MAP, Hit Rate)
  - Cohort analysis
- **Detailed logging and monitoring**:
  - Comprehensive logging system
  - Function-level logging
  - Execution time tracking
- **Production-ready design**:
  - Modular implementation
  - Configuration-driven architecture
  - Robust error handling

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/lhydra-hybrid.git
cd lhydra-hybrid
```

2. Create a conda environment:

```bash
conda create -n lhydra python=3.10
conda activate lhydra
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

Process the data using the preprocessor:

```bash
python scripts/preprocess_data.py --input notebooks/spotify_sample_data.csv --output data/preprocessed --config training/configs/training_config.yaml
```

### Model Training

Train the hybrid recommender model:

```bash
python scripts/train_model.py --input data/preprocessed/train_data.csv --config training/configs/training_config.yaml
```

To resume training from a checkpoint:

```bash
python scripts/train_model.py --input data/preprocessed/train_data.csv --config training/configs/training_config.yaml --resume models/checkpoints/best_model.pt
```

### Model Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate_model.py --model models/checkpoints/best_model.pt --data data/preprocessed/test_data.csv --config training/configs/training_config.yaml
```

### Generating Recommendations

Generate recommendations for specific users:

```bash
python scripts/generate_recommendations.py --model models/checkpoints/best_model.pt --data data/preprocessed/test_data.csv --preprocessor data/preprocessed/preprocessor.joblib --user-ids 1,2,3 --top-n 10
```

For all users:

```bash
python scripts/generate_recommendations.py --model models/checkpoints/best_model.pt --data data/preprocessed/test_data.csv --preprocessor data/preprocessed/preprocessor.joblib --user-ids all --top-n 10
```

## Model Architecture

The system uses a two-tower neural network architecture:

1. **User Tower**:

   - User embedding layer
   - Demographics processing
   - Audio profile processing (user's average audio preferences)
   - Multiple dense layers with batch normalization

2. **Item Tower**:

   - Track embedding layer
   - Artist embedding layer
   - Genre processing
   - Audio feature processing
   - Temporal feature integration
   - Multiple dense layers with batch normalization

3. **Prediction Layer**:
   - Concatenation of user and item representations
   - Dense layers with dropout
   - Sigmoid activation for final prediction

## Feature Engineering

The system performs extensive feature engineering:

1. **User Features**:

   - Demographics (age, gender, region, country)
   - Listen history patterns
   - Audio preference profile

2. **Item Features**:

   - Audio characteristics (danceability, energy, tempo, etc.)
   - Genre information
   - Artist embeddings
   - Release year and recency
   - Duration features

3. **Engineered Features**:
   - Energy-valence quadrants for mood categorization
   - Preference-track differences (comparing user's average preferences with track features)
   - Age-genre interactions
   - Geographic patterns

## Training Process

The training process includes:

- Early stopping based on validation loss
- Learning rate scheduling
- Model checkpointing
- Comprehensive metrics logging
- Cross-validation options

## Evaluation

The system offers comprehensive evaluation capabilities:

- **Classification Metrics**: Accuracy, precision, recall, F1 score
- **Ranking Metrics**: NDCG@k, MAP@k, Hit Rate@k
- **Analysis Tools**: Error analysis, cohort analysis, feature importance
- **Visualization**: ROC curves, precision-recall curves, confusion matrices

## Contributors

- Your Name

## License

[License Information]

## Contact

[Contact Information]
