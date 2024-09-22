#Gradient Descent for linear regression
#yhat=ax=b
#loss=(y-yhat)**2/N
import numpy as np
#initialise some parameters
x=np.random.rand(10,1)
y=5*x+np.random.randn()
#Parameters
a=0.0
b=0.0
#Hyperparameter
learning_rate=0.1
#create gradient function
def descent(x,y,a,b,learning_rate):
    dlda=0.0
    dldb=0.0
    N=x.shape[0]
    #loss=(y-(ax+b))**2
    for xi,yi in zip(x,y):
        dlda+=-2*xi*(yi-(a*xi+b))
        dldb+=-2*(yi-(a*xi+b))
    a=a-learning_rate*(1/N)*dlda
    b=b-learning_rate*(1/N)*dldb
    return a,b

#iteratively make updates
for epoch in range(1000):
    a,b=descent(x,y,a,b,learning_rate)
    yhat=a*x+b
    loss=np.divide(np.sum((y-yhat)**2, axis=0),x.shape[0])
    print(f'{epoch} loss is {loss}, parameter a:{a}, parameter b:{b}')
print(x,y)