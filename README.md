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

## Problem 2 - Modified LeNet5 Architecture

This implementation addresses CS461 Homework 4 - Problem 2. The task involves modifying LeNet5 to handle distorted or stylistically different MNIST digits. It includes:

1. Architectural updates to the original LeNet5:
   - ReLU activation (instead of Tanh)
   - Max Pooling (instead of Average Pooling)
   - Batch Normalization after convolution layers
   - Dropout in fully connected layers

2. Data preprocessing:
   - Padding (to resize $28 \times 28$ to $32 \times 32$)
   - Manual normalization to pixel range [0, 1] (as required by `mnist.py`)

3. Data augmentation (during training only):
   - Random rotation (±15°)
   - Random translation (10%)
   - Random scaling (90%–110%)

4. Use of TA-provided `mnist.py` for both training and testing

---

## File Descriptions

| File | Description |
|------|-------------|
| `train2_.py` | Trains `LeNet2` on the TA's MNIST dataset with augmentation |
| `test2_.py`  | Evaluates `LeNet2` on the TA's test set using `mnist.py` |
| `model2.py`  | Defines the modified LeNet2 architecture |
| `mnist.py`   | Provided by TA — custom dataset loader |
| `LeNet2.pth` | Saved model weights (output of `train2_.py`) |

---

## Running the Model

### Train

To train the modified model:

```bash
python train2_.py
