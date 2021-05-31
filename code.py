import numpy as np

n=4             #no of variables


cn=(np.array([list(3**(n-j) for j in range(1,n+1))])).T

cb=(np.array([np.zeros(n)])).T

B=np.eye(n)

b=np.zeros(n)
b[0]=1
for i in range(2,n+1):
    b[i-1]=9**(i-1)
b=np.array([b]).T

N=np.zeros((n,n))
N[0][0]=1
for i in range(2,n+1):
    N[i-1,0:i-1]=[2*(3**(i-j)) for j in range(1,i)]
    N[i-1,i-1]=1

    
xn=np.array([i for i in range(1,n+1)])
xb=np.array([i for i in range(n+1,n+n+1)])



#REVISED SIMPLEX
import time
import numpy as np
tic=time.perf_counter()

cn=(np.array([list(3**(n-j) for j in range(1,n+1))])).T

cb=(np.array([np.zeros(n)])).T

B=np.eye(n)

b=np.zeros(n)
b[0]=1
for i in range(2,n+1):
    b[i-1]=9**(i-1)
b=np.array([b]).T

N=np.zeros((n,n))
N[0][0]=1
for i in range(2,n+1):
    N[i-1,0:i-1]=[2*(3**(i-j)) for j in range(1,i)]
    N[i-1,i-1]=1

    
xn=np.array([i for i in range(1,n+1)])
xb=np.array([i for i in range(n+1,n+n+1)])
i=1

while True:
    B_inverse=np.linalg.inv(B)
    xb_prime=np.dot(B_inverse,b)
    z_prime=np.dot(cb.T,xb_prime)
    rc=cn.T-np.dot(np.dot(cb.T,B_inverse),N)
    print(f'Number {i} iteration')
    i+=1
    print('Reduced Cost',rc)
    if rc[rc>0].size==0:
        print('No more +ve rc')
        break
    
    else:
        entering_N=np.where(rc==[max(rc[rc>0])])[1][0]
        print('ent',xn[entering_N])
        aj=N[:,entering_N:entering_N+1]  #column of the entering variable
        dividing=np.dot(B_inverse,aj)
        #print(dividing)
        
        f=xb_prime/dividing   
        if min(f[f>0]).all()==np.NaN: 
            break
        else:
            leaving_B=np.where(f==[min(f[f>0])])[0][0]
            print('leav',xb[leaving_B])
            

            a,v=np.copy(cn[entering_N]),np.copy(cb[leaving_B])
            j,k=np.copy(xn[entering_N]),np.copy(xb[leaving_B])
            xn[entering_N],xb[leaving_B]=k,j

            cn[entering_N],cb[leaving_B]=v,a
            a=np.copy(B[:,leaving_B:leaving_B+1])
            v=np.copy(N[:,entering_N:entering_N+1])
            B[:,leaving_B:leaving_B+1],N[:,entering_N:entering_N+1]=v,a
            
print(f'Final Basic variables X: {xb[cb.T[0]>0]} Corresponding Values :{xb_prime[cb.T[0]>0]}, and optimal Z: {z_prime}')

toc=time.perf_counter()

print('Runtime ' ,toc-tic)
