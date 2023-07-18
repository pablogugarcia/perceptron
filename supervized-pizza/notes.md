# The Equation

In the context of supervised learning, we encounter an equation of the form ŷ = x \* w, where **X** represents the input value and **w** denotes the "slope" or "weight" associated with it. However, we need to address a limitation in this equation: it always passes through the origin. To overcome this limitation, we introduce a modification by adding a term to the equation.

The revised equation becomes ŷ = x \* w + b, where **b** represents the "y-intercept." We call it the "bias." The bias is a constant value that determines the vertical shift of the line represented by the equation. By including the bias, we ensure that the line no longer passes exclusively through the origin, allowing for more flexibility in the relationship between the input value and the predicted output.

To clarify the overall process, there are two main steps involved: **training the model** and **making predictions based on the trained model**.

During the training phase, the model learns to adjust the values of the slope (**w**) and the bias (**b**) in order to minimize the discrepancy between the predicted output (**ŷ**) and the actual output. This adjustment is achieved through various techniques, such as **gradient descent**, which iteratively updates the model's parameters based on the observed errors.

Once the model has been trained, it can be utilized to predict output values given new input. By plugging in a specific value for **x** into the equation ŷ = x \* w + b, the model calculates the corresponding predicted output (**ŷ**). This prediction process allows us to estimate the relationship between the input and the output based on the learned parameters from the training phase.

# Summary

A supervised learning system learn from examples composed of _input variables_ and _labels_. In our case the input variables were the nr of reservations and the labels were the nrs of pizzas.
Supervised learning works by approximating the examples with a function, also called the _model_. In our model we use a line with _weight_ and _bias_, the idea of approximate the examples
with a line is called _linear regression_.

The first phase is the training phase. The system tweaks the _parameters_ of the model to approximate the examples.During this search, the system is guided by a _loss_ function that measures the
distance between the current model and the _ground truth_: the lower the loss, the better the model. Our program calculates the loss with a formula called _mean squared error_.The result
of training are a weight and a bias.

The second phase is _prediction phase_. This phase passes an unlabeled input through the parametrized model. The result is a forecast, such as: “Tonight, you can expect to sell 42 pizzas.”
