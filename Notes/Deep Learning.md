# Deep Learning

## Chapter 1.  Introduction to Deep Learning

### 1.1 Program Structure:

- **Neural Networks:**   Learn how to build and train a simple neural network from scratch using python

【Build your First Neural Network -- predict bike rental 】 

- **Convolutional Networks:**  Detect and identify objects in images. 

【Object Recognition -- Dog Breed Classifier】 

- **Recurrent Neural Networks ( RNNs ):**  Particularly well suited to data that forms sequences like text, music, and time series data.

【implement [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) model】【Generate TV scripts】 

- **Generative Adversarial Networks ( GANs ) :**   One of the newest and most exciting deep learning architectures, showing incredible capacity for understanding real-world data and can be used for generating images. 

【 [CycleGAN](https://github.com/junyanz/CycleGAN) project】【Generate Novel Human Faces】 

- **Deep Reinforcement Learning**:  Use deep neural networks to design agents that can learn to take actions in simulated environment and then apply it to complex control tasks like video games and robotics.

【Teaching Quadcopter how to Fly】 



### 1.2  Applying Deep Learning

**Related DL Project Examples**:  Style Transfer；Deep Traffic (Reinforcement Learning)；Flappy Bird 

**Books to read：**

- [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) by Andrew Trask. This provides a very gentle introduction to Deep Learning and covers the intuition more than the theory. （discount code: traskud17）
- [Neural Networks And Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen. This book is more rigorous than Grokking Deep Learning and includes a lot of fun, interactive visualizations to play with. 
- [The Deep Learning Textbook](http://www.deeplearningbook.org/) from Ian Goodfellow, Yoshua Bengio, and Aaron Courville. This online book contains a lot of material and is the most rigorous of the three books suggested. 



### 1.3  Anaconda

#### **Concepts**

- **Virtualenv:**  virtual environment that allows you to separate libraries required by different projects to avoid conflict between diff library version or python version. 

- **Anaconda:** a distribution of libraries and software specifically built for data science. 

- **Conda:** a package and environment manager.  


#### **General Commands**

- **Create a new project:** type in `conda create -n project_name python=3` in terminal. 

- **Activate project: ** `source activate project_name`

- **See package installed:**  `conda list`

- **If not sure exact package name:** `conda search *possible_name*`

- **Install numpy, pandas** **and matplotlib**: `conda install numpy pandas matplotlib`  or `conda install pandas matplotlib`  ( numpy is a dependency of pandas )

  --> Conda makes sure to also install any packages that are required by the package you're installing. 

- **install Jupyter Notebook:**  `conda install jupyter notebook`

- **Deactivete project**: `source deactivate`

- **Saving environment:**  `conda env export > environment_name.yaml`

- **Loading environment from an environment file (.yaml)** :`conda env create -f environment.yaml` (project name will be "environment") 

- **Listing all environments:**  `conda env list`

- **Remove environment:**  `conda env remove -n env_name`


**e.g.** Create an environment named data installed with Python 3.6, numpy, and pandas: 

--> “conda create -n data python=3.6 numpy pandas"



#### **Practice Advices** 

- Create two env on python2 and python3, then install general use packages for each version of python to avoid conflict.   —> "conda create -n py2 python=2" & "conda create -n py3 python=3”

- To share your code on Github, remember to include a “pip_requirements.txt” file using “pip freeze” (<https://pip.pypa.io/en/stable/reference/pip_freeze/>) 





### **1.4  Jupyter Notebook**

**Jupyter** comes from the combination of Julia, Python, and R.
When saving the notebook, it's written to the server as a JSON file with a .ipynb file extension.

![Jupyter-Notebook-Architechture](images/jupyter-notebook-structure.png)

#### **Benefits**

* A great part of this architecture is that code in any language can be sent between the notebook and kernel since they are separate. (e.g. code written in R will be sent to the R kernel [list of available kernels](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels))
* Another benefit is that the server can be run anywhere and accessed via the internet. (set up a server on a remote machine or cloud instance)



#### **Commands**

* **Install:**  `pip install jupyter notebook`  or `conda install jupyter notebook`
* **Launch notebook server:**  Enter `jupyter notebook` in terminal (As long as the server is still running, you can always come back to it by going to http://localhost:8888 in your browser. If you start another server, try 8889)
* **Manage notebook environment:** "conda install nb_conda”
* **Shutdown:** click “Shutdown” in notebook or press “control + C” twice in terminal.



#### **Markdown Cells**

- **Headers:**  using “#”(hash/pound/octothorpe sign)  # Header 1;   ## Header 2;   ### Header 3 
- **Links:**  enclosing text in square brackets and the URL in parentheses.  [Udacity's home page] or ([udacity.com](http://udacity.com)) 
- **Emphasis:**  * or _ for Italics;  ** or __ for Bold. ( _gelato_ or *gelato* ==> *gelato ;*  **aardvark** or __aardvark__ ==> aardvark ) 
- **Math Expression:**  To start math mode, wrap the **LaTeX** in dollar signs $y = mx + b$ for inline math. For a math block, use double dollar signs to enclose. ( About LaTeX: [please read this primer](http://data-blog.udacity.com/posts/2016/10/latex-primer/) ) 
- **Code:**  'inline code’ or create a ''' code block '''. 

  A cheatsheet for Markdown:  <https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet> 



#### **Magic Keywords** 

* **Magic Keywords:** preceded with one or two percent signs (%, %%) for line and cell magics, respectively. 

- **%matplotlib:**  a plotting library for Python and NumPy. 
- **%timeit:**  time how long it takes for a function ( %timeit ) or the whole cell ( %%timeit ) to run.

![%timeit](images/timeit-function.png)

![%%timeit](images/timeit-cell.png)

---> **List of all available magic commands:**  [Magic Command List](http://ipython.readthedocs.io/en/stable/interactive/magics.html) 



#### **Embedding Visualizations**

- **%matplotlib:** Set up matplotlib for interactive use in the notebook. By default figures will render in a new window. 
- **%matplotlib inline:** Render figures directly in the notebook.  ( To render higher resolution images —> Use "%config InlineBackend.figure_format = ‘retina’" after "%matplotlib inline” ) 

![matplotlib-retina-display](images/magic-matplotlib.png)



#### **Debugging**

- With Python kernel, turn on interactive debugger using magic command `%pdb` . To turn it off, enter `q` .
- Read more about `pdb` in [this documentation](https://docs.python.org/3/library/pdb.html)

![Notebook-Debugging](images/magic-pdb.png)



#### Converting Format

- Notebooks are big [JSON](http://www.json.org/) files with the extension `.ipynb` .
- **Convert to an HTML file:** `jupyter nbconvert --to html notebook.ipynb` in terminal.
- Learn more about converting: [nbconvert documentation](https://nbconvert.readthedocs.io/en/latest/usage.html)



#### Slideshow with Notebook

- In the menu bar, click `View > Cell Toolbar > Slideshow` to bring up the slide cell menu on each cell.

- **Slides**: full slides move through left to right. 

- **Sub-slides**: show up in the slideshow by pressing up or down. 

- **Fragments**: hidden at first, then appear with a button press. 

- **Skip:** skip cells in the slideshow. 

- **Notes**: leaves the cell as speaker notes.

- **Running Slideshow:**  `jupyter nbconvert notebook.ipynb --to slides` 

  ​					(need to serve it with an HTTP server to actually see the presentation)

-  **To convert and open it in browser :** `jupyter nbconvert notebook.ipynb --to slides --post serve`

  [An example of a slideshow introducing pandas](http://nbviewer.jupyter.org/format/slides/github/jorisvandenbossche/2015-PyDataParis/blob/master/pandas_introduction.ipynb#/). 



### 1.5 Matrix Math and NumPy Refresher

#### Data Dimension

- **Scalars**:  have 0 dimension
- **Vectors**:  row vectors & column vectors; have 1 dimensions **(length)**
- **Matrices**:  2 dimensional grid of values. ( n rows × m columns == n by m matrix; referred by indices )
- **Tensors**:  can refer to any n-dimensional collection of values. (e.g. scalar --> zero-dimensional tensor)

![tensor](images/tensor.png)



#### NumPy Introduction

- **NumPy**:  a library that provides fast alternatives to math operations in Python, designed to work efficiently with groups of numbers. (like matrices)  --> [documentation](https://docs.scipy.org/doc/numpy/reference/).

- **Import** **Command**:  `import numpy as np`

- **Data Type:** "nparray" object can store any number of dimensions and support fast math operations, thus it can represent any of the data types covered before: scalars, vectors, matrices, or tensors. 

- **Shape**:  Returns a tuple of array dimensions. ( If we define`s = np.array(5)`  then execute `s.shape`, it will return `( )` as 5 is a scalar and has 0 dimension. )

- **Reshape**:  Changing the shape of data without changing its value. --> [documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html) 

  --> e.g. `v = np.array([1,2,3,4])` ; use  `x = v.reshape(4,1)` or `x = v[:, None]` to add dimension;  (x will be a 4*1 matrix version of vector v )



- **Scalar**: Numpy allows specify size and sign of your data, so instead of "int" in Python, it has types like  `uint8`, `int8`, `uint16`, `int16`, and so on. ( **Every item in an array must have the same type**. )

- **Vector**:  Pass a Python list to the array function`v = np.array([1,2,3])`to create a vector and access elements by their indices. ( `v.shape` == `(3,)` ;  `v[1] == 2`)

- **Matrices**: pass a list of list to NumPy's array function. `m = np.array([[1,2,3], [4,5,6], [7,8,9]])`   ( `m.shape == (3,3)` ;  `m[1][2] == 6`)

- **Tensors**: To create a 3x3x2x1 tensor,` t = np.array([[[[1],[2]],[[3],[4]],[[5],[6]]],[[[7],[8]],\    [[9],[10]],[[11],[12]]],[[[13],[14]],[[15],[16]],[[17],[17]]]])`  (`t.shape == (3,3,2,1) `)


#### Matrix Operations
- **Element-wise**:  simply apply calculation on each pair of corresponding elements in the matrices. 

- **Add **(Element-wise): `values = [1,2,3,4,5] ` , `values = np.array(values) + 5`  or `values += 5` if values is already a nbarray (the new values == ndarray [6,7,8,9,10] )

- **Multiply**  (Element-wise)  : `np.multiply(array, 5)`; `array *= 5` or `np.multiply(m, n) `;  `m * n`

- **Set Zero**:  `m *= 0` before you want to set all elements of matrix m to zero.

- **Matrix Product**：`np.matmul(A, B)` ( Order Matters!  A · B ≠ B · A ) 

  --> `dot` function in NumPy can do the same when dealing with 2-dimensional matrices. [matmul](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul) ,[dot.](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html) 

  !Matrix-Product-Example](images/matrix-multi.png)

  ##### Important Reminders About Matrix Multiplication/Product 

  - The number of columns in the left matrix must equal the number of rows in the right matrix. 

  - The answer matrix always has the same number of rows as the left matrix and the same number of columns as the right matrix. 

  - Order matters. Multiplying A•B is not the same as multiplying B•A. 

  - Data in the left matrix should be arranged as rows., while data in the right matrix should be arranged as columns. 

- **Matrix Transpose**: a matrix with the same value as the original, but its rows and columns switched.

  Use `m.T` or `m.transpose()` to transpose m.

  --> **Warning**: Be careful when modifying objects because NumPy does this without actually moving any data in memory - it simply changes the way it indexes the original matrix.

  --> Only use transpose in a matrix multiplication if data in both original matrices is arranged as rows.

![Matrix-Transpose](images/matrix-transpose.png)





## Chapter 2.  Neural Networks

### 2.1  Introduction to Neural Networks

#### Classification Problems

- **2-dimensions**:  separate with a line (Linear Boundary)

![Classification-Example](images/classify-1.png)



- **Higher Dimensions**  (If it's 3 dimensional, separate with a plane)
![n-dimensional](images/classify-2.png)



#### Perceptron

- Why it's called Neural Network: perceptrons look like the neurals in the brain.

![Perceptron](images/perceptron.png) 

- AND perceptron, OR perceptron, NOT perceptron, XOR perceptron ( XOR returns true only when one of the inputs is true and the other is false; NAND refers to NOT + AND ). 

![XOR](images/xor.png)

#### Perceptron Trick

- **Learning Rate**:  To come closer to the misclassified point, we subtract each parameter with the coordinates of the point (4, 5) and bias (1) times learning rate 0.1 so the line won't move so drastically.

![Learning Rate](images/perceptron-trick.png)



- **Psedo-code & Python for Perceptron Algorithm**:

![Pseudocode](images/perceptron-algorithm.png) 


```python
import numpy as np
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0
    
def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.

def perceptronStep(X, y, W, b, learn_rate = 0.01)
    for i in range(len(X)):
        y_pred = prediction(X[i], W, b)
        if y[i] - y_pred == 1:
            W[0] += X[i][0] * learn_rate
            W[1] += X[i][1] * learn_rate
            b += learn_rate
        if y[i] - y_pred == -1:
            W[0] -= X[i][0] * learn_rate
            W[1] -= X[i][1] * learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
```



#### Error Function

- **Error function**: tells the model how far it is from the ideal solution, then the model will continuously take steps to minimize the error.
-  Minimizing error func leads to the best possible solution. (sometimes a local minima is good enough)
- To use **Gradient Descent **, error function should be continuous and differentiable so as to be sensitive to any tiny changes since there's a small learning rate.
- **Log-loss** Error Function: Assign a large penalty to the misclassified points and a small penalty to the correctly classified points. Error = sum of penalties --> move around to decrease error.
- (Penalty --> distance from the boundary for the misclassified points)   



#### Sigmoid vs Softmax

- **Exponential function**: exp( )  will return a positive number for any input, so it's helpful when there's any  negative scores.
- **Sigmoid function**: turns any number into something between 0 and 1.
- Softmax value for N = 2 is the same as the sigmoid function.


```python
import numpy as np
def softMax(L):
	softMax = []
    expL = np.exp(L)
    sumL = np.sum(expL)
    for n in L:
        softMax.append(n/sumL)
    return softMax
```



#### One-Hot Encoding

- **One-Hot**: a group of bits among which the legal combinations of values are only  those with a single high (1) bit and all the others low (0). 

- When the input dataset has non-numerical data, use One-Hot encoding tech to turn them into 0 and 1.

![One-Hot](images/one-hot.png)



#### Maximum Likelihood

- It's a method to choose the best model, pick the one that gives existing labels the highest probability.

- **Goal**: Maximize the probability that all points are correctly classified. (if independent: p1* p2 * ... * pn)

  --> However, calculating products can be slow when there's large amount of data. --> Sum will be better.  

  --> **log(ab) = log(a) + log(b)** : log function can turn products into sums. 



#### Cross Entropy (usually used in classification problems)

- It measures how likely is it that some events happen based on given probabilities. If it's very likely to happen, then cross entropy is small. (take negative of the logarithms and sum them up)
- Good model -- Low Cross Entropy & High Probability;  Bad Model -- High Cross Entropy & Low Possibility
- As an example:

![Example](images/cross-entropy.png)

- Calculating Cross Extropy:

```python
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy1(Y, P):
    ce = 0
    for i in range(len(Y)):
        ce += Y[i] * np.log(P[i]) + (1 - Y[i]) * np.log(1 - P[i])
    return -ce

def cross_entropy2(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
```



- **Multi-Class Cross Entropy**:

![Multi-Class](images/multiclass-cross-entropy.png)



#### Logistic Regression

- Take your data --> Pick a random model --> Calculate error --> Minimize error and obtain a better model

- **Calculate Error Function**: Our Goal is to Minimize the Error Function. (using **Gradient Descent**)

![Error-Function](images/error-func.png)

![Error-Function-Formula](images/error-func-formula.png)

  

  

#### Gradient Descent:  

- Closer/Further the label is to the prediction, Smaller/Larger the gradient. Thus, we tell which direction to move toward by calculating gradient.
- A small gradient means we'll change our coordinates by a little bit, and a large gradient means we'll change our coordinates by a lot.
- **Gradient Descent Step**: subtracting a multiple of the gradient of the error function at every point, then updates the weights and bias.

![Psedocode](images/gradient-descent-psedocode.png)

```python
# Implement the following functions

import numpy as np

# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Output (prediction) formula
def output_formula(features, weights, bias):
    pred = np.matmul(features, weights) + bias
    return sigmoid(pred)

# Error (log-loss) formula
def error_formula(y, output):
    return -y * np.log(output) - (1 - y) * np.log(1 - output)

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    error = y - output
    weights += learnrate * error * x
    bias += learnrate * error
    return weights, bias
```



#### Perceptron vs Gradient Descent

- In perceptron Algorithm, only misclassified points will change weights to get the line closer to it and the correctly classified points will do nothing. 

- While in GD, every point will change the weights, the correct points will ask the line to go further away.

- In Gradient Descent, y_hat can be any number between 0 and 1, but in Perceptron it can only be 0 or 1.

![Comparation](images/perceptron-vs-GD.png)



#### Neural Network Architecture

- **Non-Linear Models**: for the data which is not separable with a line, create a non-linear probability function to separate different regions. (combine two linear models --> one non-linear model)

- **Combine Models**:

![Combine](images/combine-regions.png)

- **Example**: in the graph below, weights on the left represents the equation of previous linear model, weights on the right shows the linear combination of the two models.

![NN-Architecture](images/nn-simple.png)

  

- **Neural Network Architecture**: NN takes the input vector and then apply a sequense of linear models and sigmoid functions, then combine them into a highly non-linear map. 

  NN consists of layers. ( Input --> Hidden --> Output ) 

  In general, if there's n nodes in the input layer, we'll consider the model in n-dimensional space.

  - **Input Layers**: contains the inputs.

  - **Hidden Layers**: a set of linear models created from the input layer.

  - **Output Layers**: where the linear model get combined to obtain a non-linear model. (If there's more than 1 outputs --> **Multiclass Classification Model**)

![NN-Architecture](images/nn-arch.png)

- **Deep Neural Network**: NN that contains more layers. We can do combination multiple times to obtain highly complex models with lots of hidden layers. (deep stack of hidden layers)

  - Linear models combine to create non-linear models, and then these non-linear models combines to create even more non-linear models.
  - NN will split the n-dimensional space with a highly non-linear boundary.

![Deep-NN](images/deep-nn-arch.png)

- **Multiclass Classification Model**: add more nodes in the output layer and each one will provide the probability that the image is A, B, C .... Then apply SoftMax function to the scores provided to obtain well-defined probabilities.

![Example](images/multi-class-classification.png)

  

#### Feedforward

- Train NN: figure out what parameter should be placed on the edges in order to model the data well.

- **Feedforward** is the process neural networks use to turn the input into an output (y_hat). 

![feedforward](images/feedforward.png)

- **Error Function**: gives a measure of how badly a point gets misclassified. ( It will be measuring how far the point is from the line if it's misclassified and error will be very small if it's correctly classified. )



#### Backpropagation

- Error for units is proportional to the error in the output layer times the weight between the units.
- We can flip the network over, use the error as input and keep propagating errors through the layers.

![backpropagation](images/backpropagation-1.png)

- In a nutshell, backpropagation will consist of:
  - Doing a feedforward operation.

  - Comparing the output of the model with the desired output.

  - Calculating the error.

  - Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights. (increase weights of model and reduce weights of those )

  - Use this to update the weights, and get a better model both in the hidden layer and output layer.

  - Continue until we get a model that is good enough.

![backpropagation](images/backpropagation.png)

- **Calculating the Gradient**:

![Calculate](images/gradient-calculate.png)

![Calculate](images/multiclass-gradient-calculate.png)

![Calculate](images/backpropagation-calculate.png)



### 2.2  Implementing Gradient Descent

#### Sum of Squared Error (SSE)

- The square ensures the error is always positive and larger errors (outliers) are penalized more than smaller errors. Also, it makes the math nice, always a plus.
- **variable *j***: represents the output units of the network. So the inside sum is saying for each output unit, find the difference between the true value y and the predicted value y_hat, then square the difference and sum up all those squares. 
-  **variable *μ***: the other sum over μ is a sum over all the data points. For each data point, calculate the inner sum of the squared differences for each output unit. Then sum up those squared differences of each data points. That gives us the overall error for all the output predictions for all data points.

- Since the output of a neural network (prediction) depends on the weights, so it can also be written as the second equation, and the 1/2 here is added to help clean the math later:

![SSE-formula2](images/sse-math.png)



#### Mean Squared Error (MSE, use in regression problems)

- When the dataset is large, summing up all the weight steps can lead to really large updates that make the gradient descent diverge.
- To compensate for this, we'll need to use a quite small learning rate or just divide by the number of records (m) in our dataset to take the average.

![MSE-formula](images/MSE.png)



#### Gradient Descent

- Our goal is to find weights *Wij* that minimize the squared error *E*. --> Gradient Descent. 

- Since the steps taken should be in the direction that minimizes error the most. We can find this direction by calculating *gradient* of the squared error. (**Gradient **is another term for rate of change or slope.)

- **Calculating Gradient**: The gradient is a derivative generalized to functions with more than one variable. We can use calculus to find the gradient at any point in error function, which depends on the input weights. ( A derivative of a function *f(x)* returns the slope of *f(x)* at point x. )

- **Local Minima**: If the weights are initialized with the wrong values, gradient descent could lead the weights into a local minimum where the error is low, but not the lowest. (To avoid this, try [momentum](http://sebastianruder.com/optimizing-gradient-descent/index.html#momentum).)

- **Activation function f(h)** :

    - A func that takes input signal and generates an output signal, but takes the threshold into account.
    -  `h=∑wixi`, If we use sigmoid as `f(h)`, gradient'll be `f'(h) = f(h)*(1-f(h))`

- **Data Clean-up**: one-hot encoding, standardize data (scale the values such that they have mean of zero and a standard deviation of 1) 

  - Standardize will be necessary because the sigmoid function squashes really small and really large inputs. Gradient of really small and large inputs is zero, which means the gradient descent step will go to zero too.

- **Initialize weights**: initialize the weights from a normal distribution centered at 0, n is the number of input units. 

    - This scale keeps the input to the sigmoid low for increasing numbers of input units. 
    - It's also important to initialize them randomly so that they all have different starting values and diverge, breaking symmetry. 

    ```python
    weights = np.random.normal(scale=1/n_features**.5, size=n_features)
    ```

- **Update Weights**:

  - Set the weight step to zero: **Δwi=0**
  - For each record of data: Make a forward pass through the network; calculating the output y_hat; Calculate error term for the output unit; Update the weight step **Δwi=Δwi+δxi**.
  - Update the weights wi=wi+ηΔwi/m. η is the learning rate and m is the number of records. Here we're averaging the weight steps to help reduce any large variations in the training data.
  - Repeat for e epochs.



#### Math for Gradient Descent

![math](images/gradient-math0.png)

![math](images/gradient-math1.png)

![chain-rule](images/chain-rule.png)

![math](images/gradient-math2.png)

![math](images/gradient-math3.png)

![math](images/gradient-math4.png)

![math](images/gradient-math5.png)



#### Code for Gradient Descent

- Basic Code:

```python
import numpy as np
from data_prep import features, targets, features_test, targets_test

# Defining the sigmoid function for activations
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
    	# Loop through all records, x is the input, y is the target

        # Activation of the output unit
        #   Notice we multiply the inputs and the weights here 
        #   rather than storing h as a separate variable 
        output = sigmoid(np.dot(x, weights))

        # The error, the target minus the network output
        error = y - output

        # The error term
        #   Notice we calulate f'(h) here instead of defining a separate
        #   sigmoid_prime function. This just makes it faster because we
        #   can re-use the result of the sigmoid function stored in
        #   the output variable
        error_term = error * output * (1 - output)

        # The gradient descent step, the error times the gradient times the inputs
        del_w += error_term * x

    # Update the weights here. The learning rate times the 
    # change in weights, divided by the number of records to average
    weights += learnrate * del_w / n_records
	
    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss
     
    
# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
```

- **Multilayer Perceptrons**:

```python
import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


# TODO: Make a forward pass through the network

hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)
```



#### Implementing Backpropagation:

General algorithm for updating the weights with backpropagation:

![backpropagation-algorithm](images/implement-backpropagation-1.png)

![error-term-formula](images/implement-backpropagation.png)

```python
import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        output = sigmoid(np.dot(hidden_output,
                                weights_hidden_output))

        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        error = y - output

        # TODO: Calculate error term for the output unit
        output_error_term = error * output * (1 - output)

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)

        # TODO: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        # TODO: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:, None]

    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

```



#### Making a column vector

- By default NumPy arrays work like row vectors, if we need row vector, use `array.T` for its transpose. - 
- However, for a 1D array, the transpose will still return a row vector. Instead, use `array[:,None]`  or `np.array(features, ndmin=2).T` to create a column vector.



### 2.3 Training Neural Networks

#### Overfitting and Underfitting

- **Overfitting**: 
  - overcomplicated model which is too specific, fail to generalize. 
  - Does well in training set but tend to memorize training data instead of learning its characteristics. 
  - Error due to variance.
- **Underfitting**:
  - model too simple to fit the data.
  - Does not do well in the training set. 
  - Error due to bias.
- **Model Selection**:
  - Choose the simple model that does the job instead of the complex one that does a little better.
  - If there's no "good" model, pick an overly complicated model and apply certain technique to prevent overfitting on it.
- **Tech to prevent Overfitting**: Early stopping, Regularization, Dropout



#### Early Stopping

- Training error is always decreasing as we train the model, it keeps fitting the training data better.
- Testing error is large when underfitting, then it decrease when the model generalize well until it get to the minium point. ( **the Goldilocks spot** ) Once pass this spot, the model overfits and stop generalizing.
- **Model Complexity Graph**: 
  - Do GD until testing error stops decreasing and starts to increase. <u>Stop training at that moment</u>.
  - helps determine the # of epochs we should use. (# of epochs reflects model complexity)

![Graph](images/model-complexity-graph.png)



#### Regularization ( L1, L2 )

- Error is smaller if the prediction is closer to the actual label. The bad model provides better prediction:

![activation-function](images/activation-func-0.png)

- **Regularization**: punish high coefficients by adding a term to the previous error function, which is big when there's large weights. (constant lambda --> how much to penalize the coefficients)

![regularization](images/regularization.png)

- **L1** **Regularization**: 
  - Small weights tend to go to 0. Reduce weights and end up with a small set. 
  - Usually used to <u>select important features</u> and turn the rest into zeros.
- **L2** **Regularization**: 
  - Try to maintain all weights homogeneously small --> smaller sum of squares --> smaller error func.
  - <u>Normally gives a better results</u> for training models. (Used the most)



#### Dropout

- Sometimes part of the network have very large weights and dominates all training. 
- **Dropout**: As we go through the epochs, randomly turn off some of the nodes (both the feedforward and backpropagation will not use it) and the other nodes will take more part in training. 
- **Dropout Algorithm**: we'll need to provide a parameter of the probability that each node gets dropped in a particular epoch. 



#### Problems of GD 

**1. Got stuck in a local minima**.

- **Solutions**:

  - **Random Restart**: start from several random places and do GD from all of them. It can increase the probability of getting to the global minimum. (or at least a pretty good local minimum)
  - **Momentum**:  The idea is to walk a bit fast with momentum to pass through the local minima to go for a lower minimum. ( Momentum is a constant beta between 0 and 1, those recent steps will matter more and help the model get over the hump. )

  ![momentum](images/momentum.png)

**2.  Vanishing Gradient**:

- **Problem of sigmoid function**: The curve of the sigmoid function gets flat on the sides, so the derivative will be almost zero. This gets worse in multi-layer perceptrons, the product of a bunch of small derivatives will be tiny. 

--> GD will make very tiny changes on weights at each steps and may never get to minimum spot. 

- **Solutions**: 

  - **Change Activation functions**: 
    - **Hyperbolic Tangent**:  It has a larger range than sigmoid, range(-1, 1), thus larger derivative.
    - **Rectified Linear Unit (ReLU)**:  It's widely used since it can imporve the training significantly without sacrifice much accuracy.`relu(x) = x if x ≥ 0 else 0`



#### Stochastic Gradient Descent

- When the dataset is big, GD will require huge matrix computations. 
- Though Stochastic GD will be less accurate, it can save lots of time and memory space.

- **Steps to Implement**:

1. Split the data into several batches 
2. Run the first batch through the network 
3. Calculate error and gradient 
4. Backpropagate to update weights, get better weights and boundary region.
5. Repeat until it finish all batches.

20:14:41	insert into activities (aid, name, date, acid) values (1, 'das', 2018-10-20, 121)	Error Code: 1292. Incorrect date value: '1988' for column 'Date' at row 1	0.021 sec

#### Learning Rate Decay

- BIg learning rate --> take huge steps --> fast at the beginning but may miss the minimum.

- Small learning rate --> better chance of reaching local minimum but slow.

- **Rule of thumb**: In general, if the model doesn't work, decrease the learning rate.

- **The best learning rate**: if steep, take long steps; if plain, take small steps. 

  --> learning rate should decrease as the model is getting closer to a solution. 



### 2.6 Sentiment Analysis (Project)

- What NN really does is to search for direct or indirect correlation between two datasets. ( learn to take one and predict the other one)
- Cutting off noise, making the correlation of dataset more obvious for the NN can greatly improve prediction accuracy and training speed, save lots of computation time &power.



### 2.7 Keras

- Keras requires the input shape to be specified in the first layer, but  it will automatically infer the shape of all other layers. (we only have to explicitly set the input dimensions for the first layer) 

- [Keras Document](https://keras.io/)

- **Dense Layer**: fully connected, the nodes are connected to every node in the previous layer. 

- Adding layers: `model.add(Dense(128, activation="softmax", input_dim=100)))`

- Models need to be compiled before it can be run: `model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])` 

- see the resulting model architecture: `model.summary()`

- scoring the model: `score = model.evaluate(features, targets)`

- Loading data: 

  - **num_words**: Top most frequent words to consider.  This is useful if you don't want to consider very obscure words such as  "Ultracrepidarian."
  - **skip_top**: Top words to ignore. This is useful if  you don't want to consider the most common words. For example, the word  "the" would add no information to the review, so we can skip it by  setting `skip_top` to 2 or higher.

  ````python
  from keras.datasets import imdb
  (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                       num_words=None,
                                                       skip_top=0,
                                                       maxlen=None,
                                                       seed=113,
                                                       start_char=1,
                                                       oov_char=2,
                                                       index_from=3)
  
  ````

- Training 

#### Keras Optimizers: 

- [document](https://keras.io/optimizers/), [related blog post](http://ruder.io/optimizing-gradient-descent/index.html#rmsprop)
- **SGD**: Stochastic Gradient Descent. It uses the following parameters:

  - Learning rate.
  - Momentum (This takes the weighted average of the previous steps, in  order to get a bit of momentum and go over bumps, as a way to not get  stuck in local minima).
  - Nesterov Momentum (This slows down the gradient when i
- **Adam**:  Adaptive Moment Estimation.

  - Adam uses a more complicated exponential decay that consists of not just considering the average (first moment),  but also the variance (second moment) of the previous steps.
- **RMSProp**:  RMS stands for Root Mean Squared Error.
  -  decreases learning rate by dividing it by an exponentially decaying average of squared gradients. 



### 2.8 TensorFlow

- constant tensor: its value never changes.`tf.constant("Hello World!")`
- Tensorflow Session: is an environment for running a graph.  The session is in charge of allocating the operations to GPU(s) and/or CPU(s), including remote  machines.



#### Basic functions

- `tf.Variable()`: usually used to define trainable variables, like weights and bias. must be initialized when define.

- `tf.placeholder()`: term parameter, can be initialized later by `session.run(feed_dict)`.

- [`tf.global_variables_initializer()`](https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer) function to initialize the state of all the Variable tensors.

  ```python
  sess.run(tf.global_variables_initializer())
  ```

- [`tf.truncated_normal()`](https://www.tensorflow.org/api_docs/python/tf/truncated_normal) function to generate random numbers from a normal distribution whose magnitude is no more than 2 standard deviations from the mean. 

  - choosing weights from a normal distribution prevents any weight overwhelming other weights.
  - get_weights: `tf.Variable(tf.truncated_normal((n_features, n_labels)))`

- [`tf.zeros()`](https://www.tensorflow.org/api_docs/python/tf/zeros) function returns a tensor with all zeros.

  - get bias: `tf.Variable(tf.zeros(n_labels))`

- [`tf.matmul()`](https://www.tensorflow.org/api_docs/python/tf/matmul) function for matrix multiplication. `tf.add()` for adding matrix.



#### TensorFlow Softmax

- The softmax function: 
  - squashes it's inputs (called **logits** or **logit scores**) to be between 0 and 1 and also normalizes the outputs such that they all sum to 1. 
  - This means the output of softmax function is equivalent to a categorical probability distribution.
  - It's the perfect function to use as the output activation for a network predicting  multiple classes.
- [`tf.nn.softmax()`](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) implements the softmax function. It takes in logits and returns softmax activations.

```python
import tensorflow as tf

def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # TODO: Calculate the softmax of the logits
    softmax = tf.nn.softmax(logits)  
    
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output

```



#### TensorFlow Cross Entropy

- [`tf.reduce_sum()`](https://www.tensorflow.org/api_docs/python/tf/reduce_sum) function takes an array of numbers and sums them together.
- [`tf.log()`](https://www.tensorflow.org/api_docs/python/tf/log) takes the natural log of a number.

```python
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
cross_entropy = - tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))

with tf.Session() as sess:
    print(sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data}))
```



#### Mini-batching

- training on subsets of the dataset instead of all the data at one time.
- can't calculate the loss simultaneously across all samples.
- quite useful combined with SGD. 
  - Randomly shuffle the data at the start of each epoch, then create the mini-batches. 
  - For each  mini-batch, you train the network weights with gradient descent. (Since these batches are random, you're performing SGD with each batch)
- Size of batchs may vary: [`tf.placeholder()`](https://www.tensorflow.org/api_docs/python/tf/placeholder) function to receive the varying batch sizes.
  - The `None` dimension is a placeholder for the batch size. At runtime, TensorFlow will accept any batch size greater than 0.

```python
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    mini_batches = []
    for start in range(0, len(features), batch_size):
        end = start + batch_size
        batch = [features[start: end], labels[start: end]]
        mini_batches.append(batch)
    
    return mini_batches
```



#### Epochs

- An epoch is a single forward and backward pass of the whole dataset.   
- This is used to increase the accuracy of the model without requiring more data.  
- Lowering the learning rate would require more epochs, but could ultimately achieve better accuracy.



#### TensorFlow ReLU

- The ReLU function is 0 for negative inputs and x for all inputs x > 0.

```python
hidden_layer = tf.add(tf.matmul(features, hidden_weights), hidden_biases)
hidden_layer = tf.nn.relu(hidden_layer)

output = tf.add(tf.matmul(hidden_layer, output_weights), output_biases)
```



#### Save and Restore TensorFlow Models

- Saving Models:

```python
import tensorflow as tf

# The file path to save the data
save_file = './model.ckpt'

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize all the Variables
    sess.run(tf.global_variables_initializer())

    # Show the values of weights and bias
    print('Weights:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))

    # Save the model
    saver.save(sess, save_file)
```

- Reload Models: 

```python
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
```



#### TensorFlow Dropout

- Dropout is a regularization technique for reducing overfitting.  
- This technique temporarily drops units from the network, along with all of those units' incoming and outgoing connections.
- The  [`tf.nn.dropout()`](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) function takes in two parameters:
  - `hidden_layer`: the tensor to which you would like to apply dropout
  - `keep_prob`: the probability of keeping (i.e. *not* dropping) any given unit. (can be used to adjust the number of units to drop) 
  - You should only drop units while training the model (`keep_prob = 0.5`). 
  - During validation or testing, keep all of the units to maximize accuracy. `keep_prob = 1.0`

```python
import tensorflow as tf

hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model with Dropout
keep_prob = tf.placeholder(tf.float32)
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

# TODO: Print logits from a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits, feed_dict={keep_prob: 0.5}))
```



## Chapter 3. Convolutional Neural Network

### 3.1  AWI Instance

18.223.122.5:8888/?token=d75c263f78f56729cf5ef608441c906028784be20db691c9 

### 3.2 CNN

#### MNIST Dataset(MLP)

- 70,000 images of hand-written digits from 0 to 9, each with 28*28 = 784 pixels.
- To feed an image to MLP (Multilayer Perceptron), we must first convert the image to a vector, which will be treated as simple vector of nums with no special structure.
- **Load MNIST**:

```python
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

- **Preprocess**: 
  - feature (grayscale) from 0 to 255 -> 0 to 1 
  - One-hot encoding labels (0 to 9) -> each label becomes a vector with 10 entries.
- **Flatten layer**: takes the image matrix's input and convert it to a vector.
- **Dropout layer**: minimize overfitting, provided the probability that any node is dropped during training.

```python
from keras.models import Sequential
from keras.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_shape = X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()
```

- **Loss function**: Categorical Cross-Entropy Loss
  - if prediction and label agree (close to each other) , it returns lower loss. If ... disagree, higher loss.
  - good model ==> loss in accuracy decreases with training epochs.
- **Compile**:

```python
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

- **Train**:

```python
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)
hist = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)
```

- Load the weights that got the best validation accuracy:

```
model.load_weights('mnist.model.best.hdf5')
```

- Calculate the classification accuracy on the test set:

```python
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100 * score[1]
print('Test Accuracy: %.4f%%' % accuracy)
```



#### Model Validation in Keras

- Seperate data: 

  - **Train Set**: The model look only on train set to decide how to modify (fit) its weights;

  - **Validation Set**:  At every epoch,  The model check how well the model is doing and if it's overfitting by checking its accuracy on the validation set. ( It won't use any of the validation set for back propagation step! )

    --> <u>Always pick the model with the lowest validation loss (best validation accuracy).</u>

    --> If training loss << validation loss, this model might overfits.

  - **Test Set**: Check accuracy of the trained model.

 

#### MLP vs CNN

- Difference:
  - MLP only use fully connected layers. CNN can also use sparsely connected layers. (hidden_layer)
  - MLP only accept vector input and CNN can also accept matrices. ( CNN understand that the image pixels closer to each other are more heavily related than those far apart. )
- Similarity:
  - They're both composed of a stack of layers. (Input + Hidden + Output)
  - They share the same Loss functions and optimizers to minimize loss func.



#### Local Connectivity

- Break the image into several regions, each hidden node will be connected to only the pixels in one of these regions and report to the output layer. Output layer combines all these findings into a prediction.   
  - Each node only need to find pattern from a small local group of pixels.
  - Hidden nodes within the same collection will share a common group of weights. (Different region might share the same pattern)
- Locally connected layer uses far fewer parameters than a densely connected layer.
- It's less likely to overfit and understands how to tease out patterns contained in the image data.

![Locally-connected-layer](images/locally-connected-layer.png)



#### Convolutional Layer

- A stack of feature maps, one for each filter. Each filter responsible for finding a pattern in the image.

- **Filter** (grid of weights) has the same size as convolutional window and can be used to discover pattern.
  - Filters (weights, bias) in CNN are initially generated randomly, but CNN can learn filters on its own. 
  - To increase number of nodes --> increase num of filters
  - To increase size of the detected patterns --> increase size of the filters
- **Stride**: the amount by which the filter slide over the image.
- **Calculation**: `K` = `filters`, `F` = `kernel_size`, and `S` = `stride`
  - The **depth** of the convolutional layer will always equal the number of filters `K`. 
  - If `padding = 'same'`, then the spatial dimensions of the convolutional layer are the following:
    - **height** = ceil(float(`H_in`) / float(`S`))
    - **width** = ceil(float(`W_in`) / float(`S`))
  - If `padding = 'valid'`, then the spatial dimensions of the convolutional layer are the following:
    - **height** = ceil(float(`H_in` - `F` + 1) / float(`S`))
    - **width** = ceil(float(`W_in` - `F` + 1) / float(`S`))
- When the filter extends outside of the image:
  - Just ignore those nodes --> set Padding = 'VALID' 
  - Or padding the image edge with 0s to give the filter enough space --> set Padding = 'SAME'
- With one filter, we can only detect one single pattern. --> multiple filters

![Convolutional-Layer](images/convolutional-layer.png)

- The lighter and darker color in the filter represents the pattern it detects: 

![Convolutional-Layer](images/convolutional-layer-car.png)

- Grayscale image ==> 2-D array,  Color image ==> 3-D array (RGB image ==> depth = 3 )
- RGB image has 3 different channels, thus the filter also has 3 channels(a stack of three 2-D arrays)

![rgb-filter](images/RGB-filter.png)

- Think about each of the feature maps in a convolutional layer as an image channel and stack them to a 3-D array. Then use it as input to another convolutional layer to discover the pattern within pattern.. (we can do this again and again...)

![cool](images/multi-convolutional-layer.png)

Do **not** include the `input_shape` argument if the convolutional layer is *not* the first layer in your network.

```python
from keras.layers import Conv2D

# create convolutional layer
Conv2D(filters, kernel_size, strides, padding, activation='relu', input_shape)

# If it's in this shape: 64 filters, each with a size of 2x2, stride=1, padding='valid'
Conv2D(64, (2,2), activation='relu')
```



#### Pooling Layer

- Often appears after convolutional layer and take convolutional layers (a stack of feature maps) as input.
- Returns a stack of the same number of feature maps but with smaller width and height.
  - **Non-global** pooling layer: moderate reduction in size. (window and stride) ->Max Pooling Layer
  - **Global** ... : significant reduction in size. ->Global Average Pooling Layer(GAP layer)
- **Max Pooling Layer**:`pool_size` is number specifying the height and width of the pooling window. `strides` will default to `pool_size`.

```python
from keras.layers import MaxPooling2D
MaxPooling2D(pool_size, strides, padding)
```



#### Design CNN Architechture (Image Classification)

- **Goal**: increase depth & decrease width and height of the input array(image).

- **Layers**: Fully-connected, convolutional, pooling.
  - Convolutional: make the array deeper when passing through the network (increase depth)
  - Pooling: decrease spatial dimensions(width and height)
- Input: fix-sized image. (usually resize the images to square) 
  - RGB: 3 channels, (x, x, 3)
  - Gray-scale: 1 channel, (x, x, 1)
- Toy model: 

![toy-model](images/CNN-image-classification.png)

![toy-model](images/CNN-max-pooling.png)

- Once there's no more spatial information left to discover in the image, flatten the array to a vector and feed it to one or more fully-connected layer. (to determine what is in the image)



#### Image Augmentation in Keras

- Variance: Scale invariance, Rotation ~ & Translation ~.
- To achieve these variances --> Data Augmentation. (e.g. to achieve rotation variance, try rotate the image to different angle and add it into the training set)
- This tech helps boost the performance of CNN models and can help prevent overfitting.
- Change `model.fit() ` to `model.fit_generator()`



#### Transfer Learning

- Applying the existing trained model to solve different but related problems.
- If dataset small and similar to the previous dataset --> Remove the final layers specific to the training set while keeping the earlier ones, which are usually generalized. Then add our own layer(s) to this model and train only the final layers we added(freeze the weights of other layers).  
- If your dataset is large and very different from the previous dataset --> Remove the final specific layer. Randomly initialize the new fully layer and use the pre-trained weight for initializing other layers' weights. Then Re-train the entire network.
- Bottleneck feature: the output of the chopped model will be the bottleneck features, which is also the input of your added layers.



### 3.3 CNNs in Tensorflow

- TensorFlow uses a stride for each `input` dimension, `[batch, input_height, input_width, input_channels]`.

- We generally always set the stride for `batch` and `input_channels` (i.e. the first and fourth element in the `strides` array) to be `1`.  This ensures that the model uses all batches and input channels.  (*It's good practice to remove the batches or channels you want to skip from the data set rather than use a stride to skip them.*)  

- You'll focus on changing `input_height` and  `input_width` (while setting `batch` and `input_channels` to 1).  The `input_height` and `input_width` strides are for striding the filter over `input`. This example code uses a stride of 2 with 5x5 filter over `input`.  I've mentioned stride as one number because you usually have a square stride where `height = width`.  When someone says they are using a stride of 2, they usually mean `tf.nn.conv2d(x, W, strides=[1, 2, 2, 1])`.

- The [`tf.nn.bias_add()`](https://www.tensorflow.org/api_docs/python/tf/nn/bias_add) function adds a 1-d bias to the last dimension in a matrix.  (**Note: using tf.add() doesn't work when the tensors aren't the same shape.**)

- Set up the `strides`, `padding`, filter weight (`F_w`), and filter bias (`F_b`) such that the output shape is `(1, 2, 2, 3)`

  - out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
  - out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
  - depth doesn't change during pooling operation.


```python
"""
Setup the strides, padding and filter weight/bias such that
the output shape is (1, 2, 2, 3).
"""
import tensorflow as tf
import numpy as np

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)


def conv2d(input):
    # Filter (weights and bias)
    # The shape of the filter weight is (height, width, input_depth, output_depth)
    # The shape of the filter bias is (output_depth,)
    # NOTE: Remember to wrap them in `tf.Variable`, they are trainable parameters after all.
    F_W = tf.Variable(tf.truncated_normal((2,2,1,3)))
    F_b = tf.Variable(tf.zeros(3))
    # Set the stride for each dimension (batch_size, height, width, depth)
    # Or F_W=[3,3,1,3] and strides=[1,1,1,1]
    strides = [1, 2, 2, 1]
    # set the padding, either 'VALID' or 'SAME'.
    padding = 'VALID'
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
    # `tf.nn.conv2d` does not include the bias computation so we have to add it ourselves after.
    return tf.nn.conv2d(input, F_W, strides, padding) + F_b

out = conv2d(X)

```

- Max pooling layer:

```python
...
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
# apply max pooling
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```



### 3.4 Initialize Weights

- Use the same number for weights (e.g.0 or 1) -> hard for backprop to find gradients and change weight.
- Instead, we usually initialize weights using uniform distribution.
- **Uniform Distribution:** `tf.random_uniform()` . a probability distribution where getting number anywhere within a certain range is equally probable. (default: 0-1)
- **Normal Distribution**: `tf.random_normal()` mostly around 0. usually does better than random uniform.
- Truncated Normal: `tf.truncated_normal()` , cutting off the tails and only focus on what's around 0.
- **Rules**:
  - Have random numbers. (like uniform distribution)
  - Initialize it with negative numbers as well as positive numbers. 
  - General Rule: y = 1/√n (n is number of units)



### 3.5 Autoencoder

- Can be Applied to: image denoising & Dimensionality Reduction
- Having problems in: Compression & Generalizing to other datasets



#### Shape of Tensor

> The shape of a tensor is the number of elements in each dimension. TensorFlow automatically infers shapes during graph construction. These inferred shapes might have known or unknown rank. If the rank is known, the sizes of each dimension might be known or unknown.

| Rank | Shape            | dim  | Example                                              |
| ---- | ---------------- | ---- | ---------------------------------------------------- |
| 0    | [ ]              | 0-D  | 0-D tensor,  scalar                                  |
| 1    | [ D0 ]           | 1-D  | 1-D tensor can have a shape like [ 5 ]               |
| 2    | [ D0, D1 ]       | 2-D  | 2-D 的 tensor can have a shape like [ 3, 4 ]         |
| 3    | [ D0, D1, D2 ]   | 3-D  | 3-D 的 tensor can have a shape like [ 1, 3, 4 ]      |
| n    | [ D0, D1, … Dn ] | n-D  | n-D 的 tensor can have a shape like [ D0, D1, … Dn ] |

- Shape can be shown in Python list/ tuples, or tf.TensorShape.

- **Get Tensor Shape**:
    - `tf.Tensor.get_shape()`
    - `tf.Tensor.shape`
    - For example, to make a vector of zeros the same size as the number of columns in a given matrix:

  ```python
  zeros = tf.zeros(tf.shape(my_matrix)[1])
  ```

  

- **Modify tensor shape**

  - The **number of elements** of a tensor is the product of the sizes of all its shapes. 
  - num of elems of a scalar is always 1.
  - `tf.reshape` can be used to modify shape of tensor.

```python
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
                                                 # a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             #4x3x5 tensor

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.
yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!
```

- **None**:  when initializing shape, `None` keyword is often included, which means there's unknown dimension(s) needs to be decided by the input later.  
  - **Fully-known shape**: has a known number of dimensions and a known size for each dimension. e.g. `TensorShape([16, 256])`
  - **Partially-known shape**: has a known number of dimensions, and an unknown size for one or more dimension. e.g. `TensorShape([None, 256])`
  - **Unknown shape**: has an unknown number of dimensions, and an unknown size in all dimensions. e.g. `TensorShape(None)`

```python
self.img = tf.placeholder(dtype= tf.float32, shape= (None, 256, 256, 3), name = 'input')
```



### Train CNN with transfer learning

- Obtain bottleneck features for train, valid and test sets.
- Create a model and add your own layers
- Compile and Train your model, save the best model/weights.
- Load the model with the best validation loss
- Test the model



## Chapter 4

### 4.1 Recurrent Neural Network (RNN)

### RNN Introduction

#### Temporal Dependency

- Recurrent: repeatedly ==> RNN: perform the same task for each element in the input sequence.
- RNN -- System with memory, 
- Able to capture temporal dependency (change over time), previous input(s) matter. (only current input matters in CNN)
- Application of RNN: Speech Recognition, Time Series Prediction (traffic patterns, movie selection, stock price movement and market condition), NLP (machine translation, question answering, chatbots), Gesture Recognition.

#### Mini batch training

- using GD, update weights once every N steps with the average value
- Advantage: Noise Reduction & Complexity Reduction. --> learning process converge faster and more accurately.
- Update weightss -- Chain rule.



#### RNN

- Train with a **sequence** of inputs since previous inputs matter.
- **Memory** is defined as the output of hidden layer neurons, which will serve as additional input to the network during next training step. (provide feedback)
- State Layer
- Backpropagation through time

#### Backpropagation Through Time (BPTT)

- chain rule, but can only achieve short term memory.
- can only backprop several steps like up to 8 or 10, if backprop too deep, the gradient will gradually vanish.
- Solve Vanishing Gradient problem --> LSTM (long short term memory)
- Solve Exploding Gradient problem --> Gradient Clipping. (set a threshold, if excceed, normalize the gradient)

#### LSTM

- Avoid vanishing problem & loss of info
- Long term memory + Short term memory (LST & STM)

![lstm](/Users/Xiaowen/Documents/GitHub/Deep-Learning-Nanodegree/Notes/images/LSTM-example.png)

- 4 gates: forget, remember, learn, use
  - **forget gate**: take in LTM, multiply by a forget factor(calculated by STM and event)  to decide what part to keep and what part to forget
  - **learn gate**: take in STM & event then ignore unimportant part and keep the important part of them by multiply a ignore factor(calculated by STM and event)  (recently learned info)
  - **Remember gate**: takes in output of forget gate and learn gate, add them. (outputs an updated LTM) --> new longterm
  - **use gate/output gate**: takes in output of forget gate and learn gate. make prediction --> new shortterm.

![LSTM-0](images/LSTM-0.png)

- Decide which info to forget, which to store, and when to use it. 
- LSTM has 3 sigmoid func. 
  - Sigmoid can be used as filter of what info goes into / retains within the cell or passes through to the output. (e.g = 0; = 1; between 0 and 1represent different states)

![lstm-cell](images/LSTM-cell.png)

- Similar architectures: 

  (1) Gated Recurrent Unit(GRU) combines forget and learn gate into one update gate and run it through to a combine gate and get one new memory instead of two in LSTM, 

  (2) Peehole Connections: also connect the LTM into calculation of forget factor, LTM has more access into the forgetting decision.



### 4.2 HyperParameters

#### Learning Rate

- Recommended starting point: 0.01 (usually between 0.1 and 0.000001)
- Possible problems :
  - If model is training but too slow --> increase the learning rate.
  - Training error oscillation could indicate the training process is stuck in a bottleneck in the error curve before reaching the optimal value. --> reduce learning rate / use an adaptive learning rate.

![learn-rate](images/learn-rate.png)

- Tune learning rate:
  -  Learning Rate Decay (decrease learn rate linearly every n eopchs)
  - Adaptive learning rate (e.g. AdamOptimizer, AdagradOptimizer)

#### Minibatch Size

- Recommended starting point: 32 (usually between 32 to 256)
  - large size --> computation boost, more memory will be required and might reduce accuracy.
  - small size --> slow & will bring more noice in error calculation, noise can help to prevent the training process from stopping at local minima.

#### Epochs

- Keep training the model as long as the validation error keeps decreasing.
- Early Stopping -- determine when to stop training the model. (suppoerted by TensorFlow)

#### Num of Hidden Unit Layers

- proved techs:
  -  set the first hidden layer larger than the inputs
  - 3 layer net outperform 2 layer net.
  - deep nn --> the deeper they are, the better they perform.
- The more the better, a little larger is not a problem. But too large might overfit.
- If overfit -- training acc >> testing acc --> reduce the num of units / dropout/ L2 Regularization
- If model don't train, add more hidden units until the validation error start to get worse.

#### Embedding Size

- While some tasks show reasonable performance with embedding sizes between 50-200, it's not unusual to see it go up 500 or even 1000.



### 4.3 Word2Vec

- onehot encoding ignores the relationship between words and causes giant computational waste since most of the results will be 0,
- Onehot --> Distributed representation (embedding layer ==hidden layer work as a lookup table, lower dimensional)
- Distributed:
  - Word2Vec (skip-gram): use the input words (keyword) to predict its surrounding words (context)
  - CBOW(Continuous Bag of Words) : predict keywords by its context.



