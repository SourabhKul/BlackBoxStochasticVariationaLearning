import fa_model
import numpy as np

#Load data
Xtrain = np.load("../data/nips_train.npy")
Xtest  = np.load("../data/nips_test.npy")
Vocab = np.load("../data/nips_vocab.npy")

#Subtract the data set mean from train and test
#and re-scale the data
xmu    = np.mean(Xtrain,axis=0)
Xtrain = Xtrain - xmu
xmax   = np.max(Xtrain)
Xtrain = Xtrain/xmax
Xtest  = (Xtest - xmu)/xmax



#3.1

D = 3
K = 2
W = np.array([[1,1,0],[0,0,1]])
Psi = np.identity(D)
new_model = fa_model.fa_model(D, K, W, Psi)
infer_exact = new_model.infer(x = np.array([[1,1,1]]),W=W, Psi=Psi, method = "exact") 
infer_bbsvi = new_model.infer(x = np.array([[1,1,1]]),W=W, Psi=Psi, method = "bbsvi") 

print '3.1 Exact:', infer_exact, 'BBSVI: ', infer_bbsvi

#3.2

D = 3
K = 2
W = np.array([[1,1,0],[0,1,1]])
Psi = np.identity(D)
new_model = fa_model.fa_model(D, K, W, Psi)
infer_exact = new_model.infer(x = np.array([[1,1,1]]),W=W, Psi=Psi, method = "exact") 
infer_bbsvi = new_model.infer(x = np.array([[1,1,1]]),W=W, Psi=Psi, method = "bbsvi") 

print '3.2 Exact:', infer_exact, 'BBSVI: ', infer_bbsvi

#3.3

W = np.array([[1, 1, 0],[1, 1, 0]])
Psi = np.diag([1e-12, 1e-12, 1e-12])
x = np.array(([[1, 1, 1]]))
new_model = fa_model.fa_model(D, K, W, Psi)
infer_exact = new_model.infer(x ,W=W, Psi=Psi, method = "exact") 
infer_bbsvi = new_model.infer(x ,W=W, Psi=Psi, method = "bbsvi") 

print '3.3 Exact:', infer_exact, 'BBSVI: ', infer_bbsvi

#4.1 , 4.2, 4.3 

new_model = fa_model.fa_model(D=999, K = 10)

#new_model = fa_model.fa_model(D=999, K = 10, W = np.load('W_exact_nonabs.npy'), Psi = np.load('Psi_exact_nonabs.npy'))


new_model.fit(X=Xtrain, method = "exact")

np.save('W_exact_nonabs.npy', new_model.W)
np.save('Psi_exact_nonabs.npy', new_model.Psi)



avg_log_lik_train_exact = np.mean(new_model.marginal_likelihood( X = Xtrain ))
avg_log_lik_test_exact = np.mean(new_model.marginal_likelihood( X = Xtest ))

print 'avg_log_lik_train_exact: ', avg_log_lik_train_exact, '\n avg_log_lik_test_exact', avg_log_lik_test_exact

for i in range(new_model.W.shape[0]):
    top_xs = np.argsort(new_model.W[i],)[-10:]
    top_xs_rev = top_xs[::-1]
    for j in range(10):
        print  i, ',',Vocab[top_xs_rev[j]],' weight(', new_model.W[i,top_xs_rev[j]]  , '), '

#new_model = fa_model.fa_model(D=999, K = 10, W = np.load('W_bbsvi_nonabs.npy'), Psi = np.load('Psi_bbsvi_nonabs.npy'))

new_model.fit(X=Xtrain, method = "bbsvl")

np.save('W_bbsvi_nonabs.npy', new_model.W)
np.save('Psi_bbsvi_nonabs.npy', new_model.Psi)

avg_log_lik_train_bbsvi = np.mean(new_model.marginal_likelihood( X = Xtrain ))
avg_log_lik_test_bbsvi = np.mean(new_model.marginal_likelihood( X = Xtest ))

print 'avg_log_lik_train_bbsvi: ', avg_log_lik_train_bbsvi, '\n avg_log_lik_test_bbsvi', avg_log_lik_test_bbsvi


for i in range(new_model.W.shape[0]):
    top_xs = np.argsort(new_model.W[i],)[-10:]
    top_xs_rev = top_xs[::-1]
    for j in range(10):
        print  i, ',',Vocab[top_xs_rev[j]],',', new_model.W[i,top_xs_rev[j]]     
