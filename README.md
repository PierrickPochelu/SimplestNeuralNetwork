<h1> Simple neural network with numpy </h1>

The goal of this project is purely academic, it helps student to understand computing, specially back propagation which is not always intuitive.  This example is a very simple multi layer perceptron that learns with SGD (Stochastic Gradient Descent). A lot of implementation on the internet presents errors, for example forgetting the bias in formula. 


<hr/>

Neural network model are made of 2 parts : neural network architecture (static part) and learning part (dynamic part)


For the neural architecture:
Universal Approximation Theorem formula is equivalent to the Multi Layer Perceptron computing. The theorem says MLP can approximate any non-linear function on a compact. In practice it works really well even on noisy data.
Source:
* https://en.wikipedia.org/wiki/Universal_approximation_theorem

For the learning part:
The neural network is update by gradient descent principle. After each batch of data, derivative of the loss is computed to update weights in right direction.
Source:
* Chain rule principle (to compute error's contribution to each weight) : https://en.wikipedia.org/wiki/Chain_rule
* Automatic Differentiation : https://en.wikipedia.org/wiki/Automatic_differentiation
* Gradient Descent : https://en.wikipedia.org/wiki/Gradient_descent
* Stochastic Gradient Descent : https://en.wikipedia.org/wiki/Stochastic_gradient_descent
* Back propagation : https://en.wikipedia.org/wiki/Backpropagation



