<h1> Simple neural network with numpy </h1>

 This example is a tiny multi layer perceptron that learns with SGD (Stochastic Gradient Descent).

<hr/>

Neural network model are made of 2 math models : neural network architecture (static part) and learning part (dynamic part)


<h2>For the neural architecture:</h2>
Universal Approximation Theorem formula is equivalent to the Multi Layer Perceptron computing. The theorem says MLP can approximate any non-linear function on a compact. In practice it works really well even on noisy data.
Source:
* https://en.wikipedia.org/wiki/Universal_approximation_theorem

<h2>For the learning part:</h2>
The neural network is update by gradient descent principle. After each batch of data, derivative of the loss is computed to update weights in right direction.
Source:
<ul>
<li> Chain rule principle (to compute error's contribution to each weight) : https://en.wikipedia.org/wiki/Chain_rule </li>
<li> Automatic Differentiation : https://en.wikipedia.org/wiki/Automatic_differentiation </li>
<li> Gradient Descent : https://en.wikipedia.org/wiki/Gradient_descent </li>
<li> Stochastic Gradient Descent : https://en.wikipedia.org/wiki/Stochastic_gradient_descent </li>
<li> Back propagation : https://en.wikipedia.org/wiki/Backpropagation </li>
</ul>


<h2> Other implementation </h2>
https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795 less maths and more intuitive explanation
