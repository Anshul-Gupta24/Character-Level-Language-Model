from __future__ import print_function
import numpy as np

f=open('input.txt','r')
inp=f.read()
vocab=list(set(inp))
inp_sz=len(vocab)

char_ind = {ch:i for i,ch in enumerate(vocab)}
ind_char = {i:ch for i,ch in enumerate(vocab)}


#initialize weight matrices
W_xh = np.random.randn(100,inp_sz)*0.01     #inp_sz X 100 matrix
W_hh = np.random.randn(100,100)*0.01        #100 X 100 matrix
W_hy = np.random.randn(inp_sz,100)*0.01     #100 X inp_sz matrix
b_h = np.zeros((100,1))
b_y = np.zeros((inp_sz,1))

grad_Wxh = np.zeros_like(W_xh); grad_Whh = np.zeros_like(W_hh); grad_Why = np.zeros_like(W_hy); grad_bxh = np.zeros_like(b_h); grad_bhy = np.zeros_like(b_y)

h = {}
h[0] = np.zeros((100,1))

while(1):
    #over multiple epochs

    #variable initializations
    y = {}
    x={}
    i=0
    c=0

    ya_ind={}

    for t in xrange(len(inp)-1):
        #print("i is " + str(i))
        #print("c is " + str(c))

        #get 1 hot encoding
        ind=char_ind[inp[t]]
        xenc=np.zeros((inp_sz,1))
        xenc[ind]=1
        x[i]=xenc


        h[i+1] = np.tanh(np.dot(W_xh,xenc) + np.dot(W_hh,h[i]) + b_h)
        
        y[i] = np.dot(W_hy,h[i+1]) + b_y
        y[i] = np.exp(y[i]) / np.sum(np.exp(y[i]))

        ya = inp[t+1]
        ya_ind[i] = char_ind[ya]
        

        #backpropagate every 25 chars

        dh_next=np.zeros_like(b_h)
        d_bhy=np.zeros_like(b_y)
        d_bxh=np.zeros_like(b_h)
        d_Wxh=np.zeros_like(W_xh)
        d_Whh=np.zeros_like(W_hh)
        d_Why=np.zeros_like(W_hy)

        if(i==24):
            for v in reversed(xrange(25)):
                dy = np.copy(y[v])
                dy[ya_ind[v]]-=1
                
                d_Why += np.dot(dy,h[v+1].T)

                d_bhy += dy
                
                dh = np.dot(W_hy.T,dy) + dh_next
                dhh = (1 - h[v+1]*h[v+1])*dh
                d_Whh += np.dot(dhh,h[v].T)
                d_Wxh += np.dot(dhh,x[v].T)
                d_bxh += dhh

                dh_next = np.dot(W_hh.T,dhh)

            #clip gradients to prevent exploding gradients problem
            for dparam in [d_Wxh, d_Whh, d_Why, d_bhy, d_bxh]:
                np.clip(dparam, -5, 5, out=dparam) 

            i=-1
            c+=1

	    h[0]=np.copy(h[25])


            #update weights with AdaGrad
            for weights,grad,old_grad in zip([W_xh, W_hh, W_hy, b_y, b_h],[d_Wxh, d_Whh, d_Why, d_bhy, d_bxh],[grad_Wxh, grad_Whh, grad_Why, grad_bhy, grad_bxh]):
                old_grad+=grad*grad
                weights += -(1e-1) * grad / np.sqrt(old_grad + 1e-8)



        #sample every 100 iterations
        if(c==100):
            

            #pick 1st character at random
            #ip = np.zeros((inp_sz,1))
            #ip_1 = np.random.randint(0,inp_sz)
            #ip[ip_1]=1
            ip = x[0]


            #print a sample of size n

            n=200
            h0 = np.copy(h[0])

            for b in range(n):
                h0 = np.tanh(np.dot(W_xh,ip) + np.dot(W_hh,h0) + b_h)
                op = np.dot(W_hy,h0) + b_y
                op = np.exp(op) / np.sum(np.exp(op))
                op_ind = int(np.random.choice(range(inp_sz), p=op.ravel()))   #sample character according to output probabilties
		
            	ip = np.zeros((inp_sz,1))
		ip[op_ind]=1
                ch=ind_char[op_ind]
                #h0=np.copy(h1)
                print(ch,end='')
            
            #print ("at iteration " + string(t) + ", loss is: " + string(loss))  #print loss
            print ('')
            print ('')
            print ('')
            print ('')
            print ('')
            
            c=0

        i+=1
