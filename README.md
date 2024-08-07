# LASSO

Here is a small package that can be used to implent LASSO regression. This is mostly for learning purpose, it may be better to use arlready optimised library such as scikit.

## How to use it

Firstly you need to create an instance of the LASSO class. Then you can use the train method of that class. I may in the future implement other method
to get the accuracy, the error and to make prediction.

The train method takes in the following parameters,

1. **y**: array, expected value
2. **X**: array, variables of the data
3. **iters**: int, number of iteration
4. **lamb**: float, determine how restrictive the model will be
5. **method**: string, either "Fast" or "Normal", choose which algorithm to use either proximal descent of fast proximal descent, if not precised it use fast proximal gradient, better for convergence speed, computation time is roughly the same for both method
6. **stepsize**: string, either "Backtracking" or "Constant", choose which method will be used to determine the stepsize, if not precised it use the constant stepsize, better for computation time, convergence speed is roughly the same for both stepsize method.

*For the dimension of the array, look at the example

## How it work

I won't go too much into the details here but here is quick expalanation of how this work.
Let's take a quick example, let's suppose that we have a dataset giving us the salary and the age of a group of people. If we plot this data on a 
2d plane we would obtain a graph that would more or less look like that.

![5ab87522-132b-4a76-b63d-544b8df63df1](https://github.com/Doivssel/LASSO-/assets/172904759/e436c7d7-011d-4537-a397-55e51d554563)

Indeed we could expect that the older someone is the more likely it is for that person to have a higher salary. In other word we
expect a linear relation between the age and the salary. We expect our data to be of the 
following form $y=\beta_0+\beta_1X+\epsilon$. Now how would could we explicit that
relation. What we could to is try to fit a line that best expalin this graph in other word find
$\hat{y}=\hat{\beta_0}+\hat{\beta_1}X$ (there the $\hat{.}$ signifie estimate of). From that we understand that we 
want to find the estimate of $\beta$ So To do that
we could use the ordinary least square method. The principle behind it is quite simple, consider the 
following graph.

![2312358f-b723-4856-b586-b185dcd2e548](https://github.com/Doivssel/LASSO-/assets/172904759/1d6ea96f-96a7-431d-b4c8-a4b75cb44796)

What the ordinary least square does is to minimise the sum of the red line. In other word
it minimise the squared difference of the expected and the estimated value  
```math
S(b)=\sum_{i=1}^{n}E_i^2=\lVert Y-Xb \rVert_2^2
```

Now the problem is that finding the best coefficient is quite easy in small dimensions
there is even an explicit solution. But for high dimension it get quite complicated. One
solution to this is to select a subset of the coefficient. Now the problem is how to
do that, one possible way would be to test all subset one by one and use a criterion 
like the AIC to determine the best possible set. A good idea theorically but in practive
we would have to try $2^N$ subsets for a set of size N. Another idea is to perform
a somewhat "automatic" selection by using LASSO regression.

## Penalised regression

LASSO regression belong to a whole class of regression method called penalised regression. The principle
behind LASSO regression is to penalise our model for having certain proprieties like
too many variables. In particular the LASSO can be seen as the following constrained problem.

```math
\min_{\beta\in\mathbb{R}^p}\{S(\beta)\}\ \text{under the constraint}\sum_{j=1}^{k}|\beta_j|\le t\ , t\in\mathbb{R}^+
```
As to why this select variable, there is good geometric explanation of that.
![ced78897-cb36-4ca2-b9a6-1d30f0e13b65](https://github.com/Doivssel/LASSO-/assets/172904759/7570cc5c-0160-4a42-be29-ca9ff4933618)
On this graph is drawn the contour line of the OLS function and the constraint region of the $L^1$ norm .
To find the solution obtained by the OLS, you need to look at the point were they first
intersect. From this you can get a good understanding on how the LASSO regression select
variable. Due to to the geometric nature of the $L^1$ norm, the intersection of the OLS function and the constraint region
is more likely to be a peak. And these peak represent the part where some of the unknown coefficients are set to zero.

Now this is a bit difficult to solve numerically and there is no explicit solution. What
we may do instead is to the KKT conditions to solve an unconstrained problem instead.
four condtions on the KKT, two gives conditions on $t$ and $\lambda$, another a relation 
between $t$ and $\lambda$. But the really important part to remember about the KKT is that one way 
of solving the LASSO is by searching for the zero of the Lagrangian form associated to our problem, 
as stated by the no feasible descent condition i.e
```math
0\in \underset{sub}{\nabla}L(\beta^*,\lambda^*)=-2X^T(y-X\beta^*)+ \lambda^*\begin{cases}\{+1\} & \text { if } \beta>0\\ \{-1\} & \text { if } \beta<0 \\ {[-1,+1]} & \text { if } \beta=0\end{cases}$
```
The operator $\underset{sub}{\nabla}$ is the subgradient because the gradient is not defined for the $L^1$ norm.

Now to find the zero of the Lagrangian is a bit bothersome. A simple gradient descent can not
work since the gradient is not well defined for the Lagrangian. But there exist a way to apply
gradient descent on function of the form f=g+h with g convex differentiable and h convex
but non differentiable. This invovle a bit of not too complex mathematics that I won't show
here but in the end we obtain the following algorithm, named proximal descent

1. Choose  an initial point $\beta^{(0)}\in\mathbb{R}^p$
2. Take a gradient step $$z=\beta^t-s^t\nabla g(\beta^t)$$
3. Solve $\text{prox}_{s^th} (\beta^t-s^t\nabla g(\beta^t))$ as to obtain the generalised gradient stepsize. This can be done by applying the soft tresholding operator to z as such we can express the next iteration as

```math
\beta^{(t+1)}=S_{s^t\lambda}(z)
```

It can also be shown that by using this alogorithm the error decrease similarly to the function $\frac{1}{t}$
. 
But this can be improven by using a different algorithm that use momentum. This only involve computing two more steps, 

1. Choose  an initial point $\beta^{(0)}=\theta^{(0)}\in\mathbb{R}^p$
2. Take a gradient step
```math
\beta^{t+1}=\theta^{t}-s^{t} \nabla f\left(\theta^{t}\right)
```
```math
\theta^{t+1}=\beta^{t+1}+\frac{t}{t+3}\left(\beta^{t+1}-\beta^{t}\right)
```
Now you only need to make z equal to to this new formula of $\beta^{t+1}$ in the proximal descent algorithm to improve the error. Indeed, it can be shown that using this alogrithm, the error descreased similarly to the function $\frac{1}{t^2}$.

## Parameters

Concerning the parameters, backtracking line search or Nesterov is used constant to obtain the learning rate (stepsize). And the
$\lambda$ can be obtained by using cross-validation even if in practice cross validation tend to choose
a $\lambda$ close to zero due to the bias of the LASSO. It may be more interesting to select a subset of variable by performing the LASSO for a range of 
$\lambda$, then estimate the coefficients in the subset by using OLS has it is without bias. Then calculate the MSE associated to each $\lambda$, and finaly selecting the
lowest one. 
