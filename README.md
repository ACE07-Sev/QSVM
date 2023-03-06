# QSVM

## Abstract

In this repository, we are going to implement a Quantum Support Vector Machine Implementation using Qiskit and PennyLane embeddings for angle and amplitude encoding, and provide the best model found (1 qubit, depth of 2, accuracy of 97 percent) for the UCI Machine Learning Repository's Iris dataset.

## Introduction
### Dataset information
For this task, we will be using the UCI Machine Learning Repository's Iris dataset. This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) 

The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are **NOT** linearly separable from each other.

Inputted attribute(s) : sepal length in cm, sepal width in cm, petal length in cm, petal width in cm.

Predicted attribute: class of iris plant.

This is an exceedingly simple domain.

**Note** : This data differs from the data presented in Fishers article (identified by Steve Chadwick, spchadwick '@' espeedaz.net ). The 35th sample should be: [4.9,3.1,1.5,0.2],"Iris-setosa" where the error is in the fourth feature. The 38th sample: [4.9,3.6,1.4,0.1],"Iris-setosa" where the errors are in the second and third features.

Below, is a summary of the critical information regarding the dataset :
- There are 150 samples (instances) in the dataset.

- There are four features (attributes) in each sample.
  - sepal length in cm
  - sepal width in cm
  - petal length in cm
  - petal width in cm

- There are three labels (classes) in the dataset.

  - Iris Setosa
  - Iris Versicolour
  - Iris Virginica 

- The dataset is perfectly balanced (same  number of samples (50) in each class).

- features are not normalized, and their value ranges are different, i.e., [4.3,7.9] and [0.1,2.5] for sepal length and petal width, respectively.
![download](https://user-images.githubusercontent.com/73689800/223117466-4b8ed6bb-9a55-47cb-8ff0-ed0a4cafab10.png)

### Dataset Preprocessing

We will be applying two preprocessing steps :

1) Apply PCA to reduce the dimensionality from 4 to 2. (This will linearly decrease the width of our circuit in case of angle encoding)
2) Scale the features to be within a range of [0, 1]. 

We can see the distribution of the three classes below. As you can see, the class **Label 1** and **Label 2** are not linearly separable. 
![download](https://user-images.githubusercontent.com/73689800/223117562-923addc4-bba0-43ca-9f92-2c186a0a091d.png)

### Training a Classical SVM

For the sake of benchmarking, we will be comparing our QSVM implementation with its classical counterpart, and strive to maintain a 100 percent accuracy for the Quantum implementation as well. After applying the preprocessing, with two PCs, we have :
Classical SVC on the training dataset: 0.97
Classical SVC on the test dataset:     0.97

## QML Pipeline

To create and train any QML model, we will have three main steps :

1. Encoding the Classical data into Quantum states using Quantum Feature Maps. There are four classes of encodings, namely :
    - Amplitude Encoding (RFV)
    - Angle Encoding (ZZFeatureMap)
    - Basis Encoding
    - Arbitrary Encoding

2. Building a Quantum Ansatz/Variational form

3. Training the Quantum Ansatz classically using a classical optimizer, common algorithms are :
    - Stochastic Gradient Descent (SGD)
    - Simultaneous Perturbation Stochastic Approximation (SPSA)
 
![Alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSU0MAp9r7mbdeljWhhQn9UAQTIeH1tIJ6UJg&usqp=CAU "QML Pipeline")
                                  
                  Figure 1
### Building and Training a Quantum Support Vector Machine 

We can see the overall pipeline for QSVM in the figure below :

![Alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3A55nZMzuoRIWK0ydBb7O8VtHnSHChgrOpg&usqp=CAU "QML Pipeline")
                                  
                  Figure 2
                  
### 1. Data Encoding

Data Encoding the process of encoding classical data to quantum states using a quantum feature map.

### **Quantum Feature Maps**

A quantum feature map $\phi(\mathbf{x})$ is a map from the classical feature vector $\mathbf{x}$ to the quantum state $|\Phi(\mathbf{x})\rangle\langle\Phi(\mathbf{x})|$. This is done by applying the unitary operation $\mathcal{U}_{\Phi(\mathbf{x})}$ on the initial state $|0\rangle^{n}$ where _n_ is the number of qubits being used for encoding.

**ZZFeatureMap**

For this model, we will be using ZZFeatureMap, which is an angle encoding PQC (Parameterized Quantum Circuit), to encode the iris features into quantum states. Angle encoding is a 1-to-1 encoding, This means we can encode the two PCs using two qubits. 

`ZZFeatureMap` is conjectured to be hard to simulate classically and thus provides an incentive for using QML. This PQC can be implemented using short-depth circuits which makes it a great encoder for low dimensional datasets. 
 
 We will also show how to use Amplitude Encoding for this task as well, using  `RawFeatureVector`. This encoding is best utilized for high dimensional datasets, meaning we can encode the same information using one qubit. 
 
 **RawFeatureVector**
 
We will be using RFV, which is an amplitude encoding PQC, to encode the iris features into quantum states. Unlike angle encoding, amplitude encoding only requires Log2(N) qubits, where _N_ is the number of features in our dataset, which means we can encode the two PCs using one qubit.

The depth of the PQC is 5.
![download](https://user-images.githubusercontent.com/73689800/223118410-89c367dc-60ab-43b1-96b0-ef2446f5b24b.png)

The depth of the PQC is 3.
![download](https://user-images.githubusercontent.com/73689800/223118512-67d71400-1ff6-47b6-9f4f-8a8d6b612ba6.png)

### 2. Quantum Kernal Estimation

A quantum feature map, $\phi(\mathbf{x})$, naturally gives rise to a quantum kernel, $k(\mathbf{x}_i,\mathbf{x}_j)= \phi(\mathbf{x}_j)^\dagger\phi(\mathbf{x}_i)$, which can be seen as a measure of similarity: $k(\mathbf{x}_i,\mathbf{x}_j)$ is large when $\mathbf{x}_i$ and $\mathbf{x}_j$ are close. 
​
When considering finite data, we can represent the quantum kernel as a matrix: 
$K_{ij} = \left| \langle \phi^\dagger(\mathbf{x}_j)| \phi(\mathbf{x}_i) \rangle \right|^{2}$. We can calculate each element of this kernel matrix on a quantum computer by calculating the transition amplitude:
$$
\left| \langle \phi^\dagger(\mathbf{x}_j)| \phi(\mathbf{x}_i) \rangle \right|^{2} = 
\left| \langle 0^{\otimes n} | \mathbf{U_\phi^\dagger}(\mathbf{x}_j) \mathbf{U_\phi}(\mathbf{x_i}) | 0^{\otimes n} \rangle \right|^{2}
$$
assuming the feature map is a parameterized quantum circuit, which can be described as a unitary transformation $\mathbf{U_\phi}(\mathbf{x})$ on $n$ qubits. 
​
This provides us with an estimate of the quantum kernel matrix, which we can then use in a kernel machine learning algorithm, such as support vector classification.

As discussed in [*Havlicek et al*.  Nature 567, 209-212 (2019)](https://www.nature.com/articles/s41586-019-0980-2), quantum kernel machine algorithms only have the potential of quantum advantage over classical approaches if the corresponding quantum kernel is hard to estimate **classically**. 

With our training and testing datasets ready, we set up the `QuantumKernel` class with the [ZZFeatureMap](https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZZFeatureMap.html), and use the `BasicAer` statevector simulator `statevector_simulator` to estimate the training and testing kernel matrices.

Below, we can see what options we have for simulating the statevector, where we can use either CPU or GPU.
backend = StatevectorSimulator(precision='double')
backend.available_devices()

OUTPUT : ('CPU',)

Let's calculate the transition amplitude between the first and second training data samples, one of the entries in the training kernel matrix.
![download](https://user-images.githubusercontent.com/73689800/223118948-5ca2a4d6-cd5b-4811-9a6a-85aa61ed4c81.png)

The parameters in the gates are a little difficult to read, but notice how the circuit is symmetrical, with one half encoding one of the data samples, the other half encoding the other.

We then simulate the circuit. We will use the qasm_simulator since the circuit contains measurements, but increase the number of shots to reduce the effect of sampling noise.

This process is then repeated for each pair of training data samples to fill in the training kernel matrix, and between each training and testing data sample to fill in the testing kernel matrix. Note that each matrix is symmetric, so to reduce computation time, only half the entries are calculated explictly.

Here we compute and plot the training and testing kernel matrices:
![download](https://user-images.githubusercontent.com/73689800/223119062-1a14ee02-191c-4366-bd80-2f90b8aa5588.png)

### 3. Quantum Support Vector Classification

Introduced in [*Havlicek et al*.  Nature 567, 209-212 (2019)](https://www.nature.com/articles/s41586-019-0980-2), the quantum kernel support vector classification algorithm consists of two steps:

1. Build the train and test quantum kernel matrices.
    1. For each pair of datapoints in the training dataset $\mathbf{x_i},\mathbf{x_j}$, apply the feature map and measure the transition probability: $K_{ij} = \left| \langle 0 | \mathbf{U}^\dagger_{\Phi(\mathbf{x_j})} \mathbf{U}_{\Phi(\mathbf{x_i})} | 0 \rangle \right|^2$.
    2. For each training datapoint $\mathbf{x_i}$ and testing point $\mathbf{y_i}$, apply the feature map and measure the transition probability: $K_{ij} = \left| \langle 0 | \mathbf{U}^\dagger_{\Phi(\mathbf{y_i})} \mathbf{U}_{\Phi(\mathbf{x_i})} | 0 \rangle \right|^2$.
2. Use the train and test quantum kernel matrices in a classical support vector machine classification algorithm.

The `scikit-learn` `svc` algorithm allows us to define a [custom kernel](https://scikit-learn.org/stable/modules/svm.html#custom-kernels) in two ways: by providing the kernel as a callable function or by precomputing the kernel matrix. We can do either of these using the `QuantumKernel` class in Qiskit.

Quantum Kernel classification test score: 0.6842105263157895
linear kernel classification test score:  0.87
poly kernel classification test score:  0.97
rbf kernel classification test score:  0.97
sigmoid kernel classification test score:  0.42

As you can see, the quantum accuracy is quite subpar, let's try to improve that!

## Comparison of Approaches

We have seen a simple implementation of QSVM using Qiskit's ZZFeatureMap, and compared it to its classical counterpart, and yielded an accuracy of 86 percent. In this part, we are going to provide a rigorous treatment of how we can improve the model, and provide a comparison between all the quantum approaches presented. 

### **Comparison Variables**

We are going to try out all possible options for a few defined parameters, namely whether we are using PCA or not, what encoding protocol is used, what **package** is used, and some ML tricks and finally present a complete summary of the QSVM implementation.

For ease in implementation, we are going to convert our steps into a few functions, and create a bigger function which calls these smaller functions with different Comparison variables.

## Summary

In the table below, we can see the highest scoring combinations. We can see the best model in terms of efficiency and accuracy is achieved with scale status of **False** and PCA status of **True** with **two** PCs, using **angle-pennylane** embedding, with an accuracy of 97 percent. This model uses two qubits, and has a depth of two.

![download](https://user-images.githubusercontent.com/73689800/223119638-e921c197-ae01-4a83-954e-08cc1da2da64.png)
                         figure 3
                         
Another top combination is with scale status of **False** and PCA status of **True** with **two** PCs, using **amplitude-qiskit** embedding, with an accuracy of 97 percent. This model uses a single qubit, and has a depth of max two. (The figure below is simply an example, as we cannot draw RFV without values.)

![download](https://user-images.githubusercontent.com/73689800/223119697-86666943-9a83-42d5-a671-4ffdedd350d4.png)
                         figure 4
 
We can see the full table for the top scores below :

![download](https://user-images.githubusercontent.com/73689800/223119772-e4a01278-a5c1-4435-83be-203e0b2a48e6.png)

And for reference, the full table for the entirety of the experiment :

![download](https://user-images.githubusercontent.com/73689800/223119800-f95ca0a3-8511-429c-bb55-18732aca1f1b.png)

Let's establish some statistical knowledge from this dataset. We can observe :

1. By **increasing** the number of **reps**, the **accuracy decreases**.
2. By applying **PCA**, the **accuracy slightly drops** and **keeps dropping** as we **decrease** the number of **Principal Componennts**. (However, it provides a **trade-off** between **accuracy** and **depth**, meaning we can have a shallower PQC with a high enough accuracy)
3. We can also see how we can use a **single qubit**, with a **shallow depth** using **amplitude-qiskit** as well as **amplitude-pennylane**, which makes these two encodings the **most efficient**.
4. **Scaling** slightly **improves** the result in case of angle encoding, but not so for amplitude encoding.

Overall, the main points are to observe the tradeoff between the scale of the problem and its accuracy using PCA and encoder, and to achieve an even shallower representation, to exploit the amplitude encoding protocols. What we did in this model was to develop an understanding of the key components that build up QSVM, and how they compare to classical SVM in terms of cost and accuracy for the iris dataset.
