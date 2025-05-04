# LeNet5-handwritten-digits
LeNet5 implementation for detecting handwritten digits.

## LeNet5 - 1998 Implementation
The code for this model implementation is in LeNet1.ipynb.
LeNet1.ipynb includes:

1. The 1998 LeNet5 model architecture -- RBF prototypes calculated using digits data in './digits updated' ('Digits' dataset from Kaggle: https://www.kaggle.com/datasets/karnikakapoor/digits)
3. Loading the MNIST data to data loaders for train and test data
4. Training amd testing on MNIST data in './data'
5. Model evaluation

### Running the Model

Run the full LeNet1.ipynb notebook to train and test the model on MNIST data.

### Testing the Model

The LeNet1.ipynb model tests on MNIST data.
To test on another dataset, update the test1.py file to load in another dataset. The test1.py file uses the same trained model from LeNet1.ipynb.
