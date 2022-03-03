### Parameter Initialization

While training a network, we need to set the weights to values that minimize the convergence between model's output and the training data. And the initial value of the weights plays a significant role here.

The initialization step can be critical to the model’s ultimate performance, and it requires the right method.
- **Case 0:** Initializing all the weights with `zeros` leads the neurons to learn the same features during training. In fact, any constant initialization scheme will perform very poorly.
- **Case 1:** A `too-large` initialization leads to exploding gradients.
- **Case 2:** A `too-small` initialization leads to vanishing gradients.

To prevent the gradients of the network’s activations from vanishing or exploding, we will stick to the following rules of thumb:

- The `mean` of the activations should be `zero`.
- The `variance` of the activations should stay the `same across every layer`.

**I will try to train the model with `ADAM`, `SGD`, and `RMSProp` optimizers with `normal` and `xavier` ininitialization.**

### About the data

Using the PyTorch dataset API to load a dataset with exactly the same properties as the MNIST handwritten digits dataset. However, instead of handwritten digits, this dataset contains images of 10 different **common clothing items**, hence the name **Fashion-MNIST** . Performance on MNIST saturates quickly with simple network architectures and optimization methods. This dataset is more difficult than MNIST and is useful to demonstrate the relative improvements of different optimization methods. 

Some of the characteristics are mentioned below.

- 28x28 images
- 10 classes
- Single color channel (B&W)
- Centered objects
- 50000 training set members
- 10000 test set members
