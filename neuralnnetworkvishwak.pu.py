import numpy as np
def sig_der(value):
    return value * (1 - value)
def sig(value):
    return 1 / (1 + np.exp(-value))
train_ip = np.array([[0,1],
                     [1,0],
                     [1,1]])
train_op = np.array([[1,1,1]]).T                 
np.random.seed(1)                    
syn_weight = 2 * np.random.random((2,1)) - 1
#print syn_weight
for i in range(8000):
    ip = train_ip
    op = sig(np.dot(ip,syn_weight))
    error = train_op - op
    adj = error * sig_der(op)
    syn_weight += np.dot(ip.T , adj)
print '##################'    
print sig(syn_weight[0][0] * 0 + syn_weight[1][0] * 0)
print '##################'
print op

