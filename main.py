import numpy as np
LOG_LEVEL=0
def log_debug(txt):
	global LOG_LEVEL
	if LOG_LEVEL==1:
		print(txt)

############################
# NEURAL NETWORK FUNCTIONS #
############################
		
def active(x):
	# return x # linear
	return np.where(x <= 0, 0, x)  # relu

def dactive(x):
	#return np.ones(x.shape) # derivative linear
	return np.where(x <= 0, 0, 1) # derivate relu. Warning x=0 is not derivable.
	
def loss(output,groundtrue):
	return (output-groundtrue)**2

def dloss(output,groundtrue):
	return 2*(output-groundtrue)

def addbias(x):
    return np.concatenate((np.ones(1,),x),axis=0)

def rmbias(x):
    return x[:1]

def dot(x1,x2):
    return np.dot(x1,x2)

def multiply(x1,x2):
    return np.multiply(x1,x2)

def transpose_vector(x):
    return np.array([x]).T

####################
# HYPER PARAMETERS #
####################
hidden_layer_size=5
batch_size=10
lr=1e-5

######################
# SYNTHETIC DATASET #
######################

nb_x=100
nb_features=2 # input size
nb_labels=1 # output size
X=np.random.uniform(-0.5,+0.5,(nb_x,nb_features))
Y=X[:,0]
input_layer_size=nb_features

###################
# INITIALISATION  #
###################
w1_shape=((input_layer_size + 1),hidden_layer_size)
w2_shape=((hidden_layer_size+1,nb_labels))
w1=np.random.normal(0.,+0.25,w1_shape)
w2=np.random.normal(0.,+0.25,w2_shape)

w1_grad=np.zeros(w1.shape)
w2_grad=np.zeros(w2.shape)



############
# TRAINING #
############

cumul_loss=0


for epoch in range(1000): # for each epoch
    for id_sample in range(nb_x): # for each data
    
        ###########
        # FORWARD #
        ###########
        a1=X[id_sample] # affectation called "input layer"

        # layer 1 (layer hidden)
        a1b=addbias(a1)
        a2=np.dot(a1b,w1)
        z1=active(a2)

        # layer 2 (output layer)
        a2b=addbias(z1)
        a3=np.dot(a2b,w2)
        z2=active(a3)


        #################
        # COMPUTE DELTA #
        #################
        
        # layer 2
        delta_loss=dloss(z2,Y[id_sample])
        w2_delta=delta_loss

        # layer 1
        delta_a2b=multiply(dot(w2,w2_delta),dactive(a3))
        w1_delta=rmbias(delta_a2b)

        #####################
        # COMPUTE GRADIENTS #
        #####################
        w1_grad=w1_grad+w1_delta*2*transpose_vector(a1b)
        w2_grad=w2_grad+w2_delta*2*transpose_vector(a2b)
        
    
        
        #################################
        # UPDATE WEIGHTS WITH GRADIENTS #
        #################################
        if (id_sample%batch_size==0 and id_sample!=0) or id_sample==nb_x: # call it for each batch
            w1=w1 - w1_grad*lr
            w2=w2 - w2_grad*lr
            # reset gradients to the next batch
            w1_grad=np.zeros(w1_grad.shape)
            w2_grad=np.zeros(w2_grad.shape)
        
        cumul_loss+= loss(z2,Y[id_sample]) # compute loss to debug purpose
    print(cumul_loss)
    cumul_loss=0

"""
********************************************
* FORMULAS DEMONSTRATION  WITH PSEUDO CODE *
********************************************

'A -> B' mean B is derivative of A (regarding minimazing of the cost function.)

1) COMPUTE ERROR CONTRIBUTION (DELTA) :

layer 2 :
loss(z2) -> loss(z2)
w1_delta=dloss(z2)

layer 1 :
loss(active(z1*w2)) -> dloss(active(z1*w2))*dactive(z1*w2)*w2  # chain rule principle
w2_delta=w1_delta*dactive(z1*w2)*w2

2) COMPUTE GRADIENTS :
w1_grads=2*a1b*w1_delta
w2_grads=2*a2b*w2_delta

Demonstration : w_delta=d(loss_layer)/d(W)=dloss_layer(W)
loss_layer(W)=(y-W*xb)**2 
= y**2-2*y*(W*xb)+(W*xb)**2                       | regarding formula : (a*b)**2=(a**2)*(b**2)
= y**2 - 2*y*W*xb + (W**2)*(xb)**2
d(loss_layer)/d(W)=  - 2*y*xb   + 2*W * (xb)**2   | regarding formula : derivative a*x**2 -> a*2*x
= 2 * (W*(xb)**2  - y*xb  )
= 2 * xb * (W*xb - y)
=2*xb*(W*xb-y)

3) EFFECTIVELY UPDATE WEIGHTS
regarding formula of gradient descent : w=w-lr*f'(w)
w1=w1 - w1_grad*lr
w2=w2 - w2_grad*lr

"""
