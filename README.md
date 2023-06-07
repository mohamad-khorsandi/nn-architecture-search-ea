# evolutionary-network-arc-search
find a good neural network architecture by an evolutionary algorithm

#### the project goal is to find a neural network architecture that fits cifar10 data the best
- evolutionary algorithm should search among
  - different feature extractors
  - different layers count
  - different neuron count for each layer
  - different activation function for each layer such as ReLU, sigmoid, linear
- used pytorch to extract feature 
- used tensorflow to evaluate each model

### Installation
1. Create a virtual environment for the project:
```
python -m venv venv
```
2. Activate the virtual environment:
  - On Windows:
```
venv\Scripts\activate
```

  - On macOS and Linux:
```
source venv/bin/activate
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. run:
```
python main.py
```
