

# Deep Learning

## Chapter 1.  Introduction to Deep Learning

### 1.1 Program Structure:

**Neural Networks:**   Learn how to build and train a simple neural network from scratch using python.
【Build your First Neural Network ( predict bike rental ) 】 

**Convolutional Networks:**  Detect and identify objects in images. 
【Object Recognition — Dog Breed Classifier】 

**Recurrent Neural Networks ( RNNs ):**  Particularly well suited to data that forms sequences like text, music, and time series data.
【implement [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) model】【Generate TV scripts】 

**Generative Adversarial Networks ( GANs ) :**   One of the newest and most exciting deep learning architectures, showing incredible capacity for understanding real-world data and can be used for generating images. 
【 [CycleGAN](https://github.com/junyanz/CycleGAN) project】【Generate Novel Human Faces】 

**Deep Reinforcement Learning**:  Use deep neural networks to design agents that can learn to take actions in a simulated environment and then apply it to complex control tasks like video games and robotics. 
【Teaching a Quadcopter how to Fly】 



### 1.2  Applying Deep Learning

**Related DL Project Examples：**Style Transfer；Deep Traffic ( Reinforcement Learning)；Flappy Bird 

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
-  Minimizing error func leads to the best possible solution. (sometimes a local minimum is good enough)
- To use **Gradient Descent **, error function should be continuous and differentiable so as to be sensitive to any tiny changes since there's a small learning rate.
- **Log-loss** Error Function: Assign a large penalty to the misclassified points and a small penalty to the correctly classified points. Error = sum of penalties --> move around to decrease error.
- (Penalty --> distance from the boundary for the misclassified points)   



#### Sigmoid vs Softmax

- **Exponential function** (Exp) will return a positive number for any input, so it's helpful when there's any  negative scores.
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



#### Cross Entropy

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
- Calculate Error Function: 