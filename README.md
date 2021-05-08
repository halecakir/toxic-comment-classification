# Toxic Comment Classification

Yet another toxic comment classification

## Installation

#### Prerequisites
    - Python 3.7 or higher
    - GNU Make
    - CUDA 10.2 or higher
    
#### Cloning
Clone the repo to your local machine:
```sh
git clone https://github.com/halecakir/toxic-comment-classification
```

#### Installating
Build the python virtual environment:

```sh
make venv/bin/activate
```

#### Fetching Data
Fetch wordvec data from multiple sources (glove, google-news, fasttext):

```sh
make fetch_all
```

#### Training
Train the model with the jigsaw data:
```sh
make train ARGS=WORD_VECTOR  # WORD_VECTOR ∈ {"google.bin", "fasttext.bin", "glove.txt"})
```

#### Testing
Test the model:
```sh
make test
```

#### Cleaning
Remove all model artifacts:
```sh
make clean
```

#### Todos
- Try Attention mechanism
- Try tranformers-based mechanismss
- Try incorporation of hybrid (word level + character level) word vectors for words that have no pretrained vectors
- Try Gradient clipping for exploding gradient	
- Add hyperparamerer optimization
- Add sanity tests
- Documentation!
