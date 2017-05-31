'''
Created on May 29, 2017

@author: zwieback
'''
from scipy.interpolate import BSpline, CubicSpline, splrep
import numpy as np
import pylab as plt
import theano.tensor as tt
nsegments=12
segmentpoints=np.linspace(0,1,nsegments+1,endpoint=True)
x=np.linspace(0,1,1500)
y=np.sin(x*2*np.pi+0.3)

k=3
assert k==3
#knots=np.hstack(([0,0],segmentpoints,[1,1]))
dsegment=segmentpoints[1]-segmentpoints[0]
knotsend=np.linspace(dsegment,(k)*dsegment,k)
knots=np.hstack((-knotsend[::-1],segmentpoints,knotsend+1))

nknots=len(knots)

basis_functions=BSpline(knots,np.eye(nknots),k=k)
Bxraw=basis_functions(x)

Bx=Bxraw[:,1:nsegments+1]
Bx[:,0]=Bx[:,0]+Bxraw[:,nsegments+1]
Bx[:,1]=Bx[:,1]+Bxraw[:,nsegments+2]
Bx[:,nsegments-1]=Bx[:,nsegments-1]+Bxraw[:,0]

#check periodicity (value + derivative)
#parameterize in terms of central differences of coefficients
print(Bx[0,:]-Bx[-1,:])
Bxdiff=np.diff(Bx,axis=0)
print(Bxdiff[0,:]-Bxdiff[-1,:])
#
coeff=np.zeros(nsegments)
coeff[11]=1

M=(np.diag(np.ones(nsegments),k=0)-np.diag(np.ones(nsegments-1),k=-1))
M[0,nsegments-1]=-1

Minv=np.linalg.pinv(M)
U,s,V = np.linalg.svd(M,full_matrices=True)

coefftodiffcoeff=lambda coeff:np.dot(M,coeff)
diffcoefftocoeff=lambda diffcoeff: np.dot(Minv,diffcoeff)
evalbspline=lambda coeff:np.tensordot(Bx,coeff,axes=([1],[0]))
interpfunction=evalbspline(coeff)


print(coefftodiffcoeff(coeff))
#print(diffcoefftocoeff(coefftodiffcoeff(coeff)))


plt.plot(x,interpfunction)
plt.plot(x,evalbspline(diffcoefftocoeff(coefftodiffcoeff(coeff))))
#plt.plot(x,Bx2)

plt.show()

'''
nsegments=11
segmentpoints=np.linspace(0,1,nsegments+1,endpoint=True)
x=np.linspace(0,1,1500)
y=np.sin(x*2*np.pi+0.3)

k=3
assert k==3
#knots=np.hstack(([0,0],segmentpoints,[1,1]))
dsegment=segmentpoints[1]-segmentpoints[0]
knotsend=np.linspace(dsegment,(k)*dsegment,k)
knots=np.hstack((-knotsend[::-1],segmentpoints,knotsend+1))

nknots=len(knots)

basis_functions=BSpline(knots,np.eye(nknots),k=k)
Bx=basis_functions(x)

Bx2=Bx[:,1:nsegments+1]
Bx2[:,0]=Bx2[:,0]+Bx[:,nsegments+1]
Bx2[:,1]=Bx2[:,1]+Bx[:,nsegments+2]
Bx2[:,nsegments-1]=Bx2[:,nsegments-1]+Bx[:,0]

#check periodicity (value + derivative)
#parameterize in terms of central differences of coefficients
#print(Bx2[0,:]-Bx2[-1,:])
Bx2diff=np.diff(Bx2,axis=0)
#print(Bx2diff[0,:]-Bx2diff[-1,:])
#
coeff=np.zeros(nsegments)
coeff[0]=1
#coeff[4]=-0.5
#coeff[6]=0.5
interpfunction=np.tensordot(Bx2,coeff,axes=([1],[0]))

M=0.5*(np.diag(np.ones(nsegments-1),k=1)-np.diag(np.ones(nsegments-1),k=-1))
M[0,nsegments-1]=-0.5
M[nsegments-1,0]=+0.5
Minv=np.linalg.pinv(M)
U,s,V = np.linalg.svd(M,full_matrices=True)

#print(np.array_str(M,precision=2))
#print(V[-1,:])
#print(V[-2,:])


coefftodiffcoeff=lambda coeff:np.dot(M,coeff)
diffcoefftocoeff=lambda diffcoeff: np.dot(Minv,diffcoeff)
print(diffcoefftocoeff(coefftodiffcoeff(coeff)))


plt.plot(x,interpfunction)
plt.plot(x,np.tensordot(Bx2,diffcoefftocoeff(coefftodiffcoeff(coeff)),axes=([1],[0])))
#plt.plot(x,Bx2)



plt.show()
'''