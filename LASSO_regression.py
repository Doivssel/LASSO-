import numpy as np

class LASSO():

    def __init__(self):
        pass

    def MSE(self,y,Y_esti):
        """
        Parameters:
        y: list look at the example for the dim, expected output
        Y_esti, list look at the example for the dim, estimated output

        Goal:
        Calculate the mean square error of an estimation to measure its 
        performance

        Note: 
        Hardly useful maybe to delete
        """
        N=np.size(y)
        Squared=[(y[i]-Y_esti[i])**2 for i in range(N)]
        return(np.sum(Squared)/N)

    def cost_OLS(self,y,X,B):
        """
        Parameters:
        y: list look at the example for the dim, expected output
        X: matrix look at the example for the dim, input
        B: list, unknown parameters of our model
        
        Goal:
        Compute the cost of Ordinary Least Square method for a certain B
        """
        N=np.size(y)
        err=y-np.dot(X,B)
        return(np.dot(err.T,err)/(2*N))
    
    def grad_OLS(self,y,X,B):
        """
        Parameters:
        y: list look at the example for the dim, expected output
        X: matrix look at the example for the dim, input
        B: list, unknown parameters of our model
        
        Goal:
        Compute the gradient of the Ordinary least Square method
        """
        N=np.size(y)
        return(np.dot(X.T,np.dot(X,B)-y)/N)

    def cost_LASSO(self,y,X,B,lam):
        """
        Parameters:
        y: list look at the example for the dim, expected output
        X: matrix look at the example for the dim, input
        B: list, unknown parameters of our model
        lam: float, determine how restrictive our estimation is
        
        Goal:
        Compute the cost of the LASSO method for a certain B
        """
        return(self.cost_OLS(y,X,B)+lam*np.sum(np.abs(B)))

    def soft_tresh(self,x,tau):
        """
        Parameters:
        x: list
        tau: float
        
        Goal:
        Perform the soft tresholding operation, look at the README for more details
        """
        return(np.sign(x)*np.maximum(np.abs(x)-tau,np.zeros(len(x))))

    def armijo(self,y,X,Bk,a=0.5,s=0.8):
        """
        Parameters:
        y,X:already defined before
        Bk: list, current value of B estimated through gradient descent
        a,s: float, parameters of the backtracking algorithm

        Goal:
        Perform a backtracking line search to find a good stepsize for the gradient descent
        For more details, https://en.wikipedia.org/wiki/Backtracking_line_search 
        """
        step=0.01
        while(True):
            Bkp=Bk-step*self.grad_OLS(y,X,Bk)
            if(self.cost_OLS(y,X,Bkp)<=self.cost_OLS(y,X,Bk)+a*step*np.dot(self.grad_OLS(y,X,Bk).T,step*self.grad_OLS(y,X,Bk))):
                return(step)
            else:
                step=s*step

    def proximal_descent_A(self,y,X,beta,iters,lamb):
        """
        Parameters:
        y: list look at the example for the dim, expected output
        X: matrix look at the example for the dim, input
        beta: list, unknown parameters of our model
        iters: int, number of iteration
        lamb: float, determine how restrictive our model is

        Goal:
        Perform a proximal descent using backtracking to find a good stepsize
        """
        evol=[]
        mse=[]
        for i in range(iters):
            grad=self.grad_OLS(y,X,beta)
            s=self.armijo(y,X,beta)
            beta=self.soft_tresh(beta-s*grad,s*lamb)
            evol.append(beta)
            mse.append(self.MSE(y,np.dot(X,beta)))
        return(beta,evol,mse)

    def fast_proximal_descent_A(self,y,X,beta,iters,lamb):
        """
        Parameters:
        y: list look at the example for the dim, expected output
        X: matrix look at the example for the dim, input
        beta: list, unknown parameters of our model
        iters: int, number of iteration
        lamb: float, determine how restrictive our model is

        Goal:
        Perform a fast proximal descent using backtracking to find a good stepsize
        """
        evol=[]
        mse=[]
        theta=np.copy(beta)
        for i in range(iters):
            grad=self.grad_OLS(y,X,beta)
            s=self.armijo(y,X,beta)
            betam=beta
            beta=self.soft_tresh(theta-s*grad,s*lamb)
            theta=beta+(i/(i+3))*(beta-betam)
            evol.append(beta)
            mse.append(self.MSE(y,np.dot(X,beta)))
        return(beta,evol,mse)


    def constant(self,X):
        N=np.shape(X)[0]
        eig=np.linalg.eig(np.dot(X.T,X)/N)[0]
        return(1/(20*max(eig)))

    
    def proximal_descent_C(self,y,X,beta,iters,lamb):
        """
        Parameters:
        y: list look at the example for the dim, expected output
        X: matrix look at the example for the dim, input
        beta: list, unknown parameters of our model
        iters: int, number of iteration
        lamb: float, determine how restrictive our model is

        Goal:
        Perform a proximal descent using the constant strategy to find a good stepsize
        """
        evol=[]
        mse=[]
        s=self.constant(X)
        for i in range(iters):
            grad=self.grad_OLS(y,X,beta)
            beta=self.soft_tresh(beta-s*grad,s*lamb)
            evol.append(beta)
            mse.append(self.MSE(y,np.dot(X,beta)))
        return(beta,evol,mse)

    def fast_proximal_descent_C(self,y,X,beta,iters,lamb):
        """
        Parameters:
        y: list look at the example for the dim, expected output
        X: matrix look at the example for the dim, input
        beta: list, unknown parameters of our model
        iters: int, number of iteration
        lamb: float, determine how restrictive our model is

        Goal:
        Perform a fast proximal descent using the constant strategy to find a good stepsize
        """
        evol=[]
        mse=[]
        s=self.constant(X)
        theta=np.copy(beta)
        for i in range(iters):
            grad=self.grad_OLS(y,X,beta)
            betam=beta
            beta=self.soft_tresh(theta-s*grad,s*lamb)
            theta=beta+(i/(i+3))*(beta-betam)
            evol.append(beta)
            mse.append(self.MSE(y,np.dot(X,beta)))
        return(beta,evol,mse)
    
    def train(self,y,X,iters,lamb,method="Fast",stepsize="Constant"):
        """
        Parameters:
        y: list look at the example for the dim, expected output
        X: matrix look at the example for the dim, input
        beta: list, unknown parameters of our model
        iters: int, number of iteration
        lamb: float, determine how restrictive our model is
        method:  string either Fast for Fast proximal descent or Normal for proximal descent
        stepsize: string either Constant or Backtracking

        Goal:
        Train our model and return the estimated beta, the evolution of the estimation beta and the evolution of the MSE
        """
        beta=np.random.uniform(-1,1,np.shape(X)[1])
        if(method=="Fast" and stepsize=="Constant"):
            return(self.fast_proximal_descent_C(y,X,beta,iters,lamb))
        elif(method=="Fast" and stepsize=="Backtracking"):
            return(self.fast_proximal_descent_A(y,X,beta,iters,lamb))
        elif(method=="Normal" and stepsize=="Backtracking"):
            return(self.proximal_descent_A(y,X,beta,iters,lamb))
        elif(method=="Normal" and stepsize=="Constant"):
            return(self.proximal_descent_C(y,X,beta,iters,lamb))
        else:
            print("Unknown method/stepsize, possible choice are \n methods: Fast, Normal \n stepsizes: Constant, Backtracking")
            raise SystemExit(0)

