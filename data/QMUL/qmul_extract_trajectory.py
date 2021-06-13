import scipy.io
import numpy as np
import os

dest_path = './Trajectory/'
directory = 'overall'
mat = scipy.io.loadmat('qmul.mat')
labled_tra = mat['labled_tra']
print (labled_tra.shape[1])
no_tracks = labled_tra.shape[1]
print (labled_tra[0][0][0].shape)
destfile = "test.txt"
np.savetxt(destfile, labled_tra[0][0][0])

x=np.array(labled_tra[0][0][0][0])
y=np.array(labled_tra[0][0][1][0])
z=np.array(labled_tra[0][0][2][0])
#print(x)
#print(y) 
#print(z)

np.savetxt('myfile.txt', np.transpose([x,y,z]), fmt='%-3.2f', delimiter=' ')
var2 = np.loadtxt('myfile.txt')
print(var2.shape)

for i in range(no_tracks):
    x=np.array(labled_tra[0][i][0][0])
    y=np.array(labled_tra[0][i][1][0])
    z=np.array(labled_tra[0][i][2][0])
    if not os.path.exists(dest_path+directory):
        os.makedirs(dest_path+directory)
    filename = str(i)+".txt"
    destfile = os.path.join(dest_path, directory, filename) 
    print(destfile)
    np.savetxt(destfile, np.transpose([x,y,z]), fmt='%-3.2f', delimiter=' ')
'''
array1=np.array([1.5397e-05,8.7383e+00,2.6633e+01,1.1309e+03,4.3194e+02,2.5086e+01])
array2=np.array([4.83,1.4,0.4,-7.2,-3.64,0.6])
array3 = ['Sun','Sirius','Arcuturus','Betelgeuse','Polaris','Vega']


with open('star.txt', 'w') as f:
    for a, b, name in zip(array1, array2, array3):
        f.write('{0:15}{1:15}{2:15}\n'.format(name, a, b))


#DataLin = mat['DataLin']
DataHu2 = mat['DataHu2']
#DataMorris = mat['DataMorris']
print len(DataHu2)
for i in range(1500):
    data = DataHu2[i,:]
    #print len(data[0][0])
    print data[0][1][0][0]
    directory = str(data[0][1][0][0])
    if not os.path.exists(dest_path+directory):
        os.makedirs(dest_path+directory)
    filename = str(i)+".txt"
    destfile = os.path.join(dest_path, directory, filename) 
    print destfile
    np.savetxt(destfile, data[0][0], fmt='%-3.8f')
    #var2 = np.loadtxt('test.txt')
    #print var2
'''
