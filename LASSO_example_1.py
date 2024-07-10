from LASSO_regression import *
import matplotlib.pyplot as plt

M1=[1]*4+[0]*6
X=np.random.normal(0,1,(500,10))
eps=np.random.normal(0,1,500)
y_M1=np.dot(X,M1)+eps


reg=LASSO()

beta,evol,mse=reg.train(y_M1,X,50,.05)

print("Estimation of beta: ")
print(np.round(beta,2))

plt.figure()
plt.plot(evol)
plt.title("Evolution of estimated beta")
plt.xlabel("Iteration")
plt.ylabel("coefficient value")


plt.figure()
plt.plot(mse)
plt.title("Evolution of the MSE")
plt.xlabel("Iteration")
plt.ylabel("MSE")

plt.show()