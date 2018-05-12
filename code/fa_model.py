#Must use autograd versions of numpy/scipy functions
#for automatic differentiation support
import autograd.numpy as np
import autograd.numpy.linalg
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.scipy.stats import multivariate_normal



class fa_model:
   
    def __init__(self,D,K,W=None,Psi=None):
        '''
        function: __init__
        Description: Initialize the object
        Inputs:
            D - (int) number of data dimensions
            K - (int) number of latent factors
            W - (np.array) Factor loading matrix. Shape (KxD).
            Psi - (np.array) Output covariance matrix. Shape (DxD). Positive, diagonal.
        Outputs: None
        '''
        self.D = D
        self.K = K
        self.W = W
        self.Psi = Psi
        np.random.seed(5)
        #thanks for pre-initilizing the class! Saved some time.

    def joint_likelihood(self,Z,X,W=None,Psi=None):
        '''
        function: joint_likelihood
        Description: Compute the joint log likelihood for given data X and latent factors Z.
        Inputs:
            Z -   (np.array) Latent factor matrix. Shape (N,K).
            X -   (np.array) Data matrix. Shape (N,D)
            W -   (np.array) Factor loading matrix. Shape (K,D).
            Psi - (np.array) Output covariance matrix. Shape (D,D). Positive, diagonal.
        Outputs:  
            jl -  (np.array) Array of shape (N,) where jl[n] is the joint
                  log likelihood of data case X[n,:] and latent factor vector Z[n,:]
        '''
        #print 'JL'
        if(W is None): W = self.W
        if(Psi is None): Psi = self.Psi
        
        D,_ = Psi.shape
        N,K = Z.shape 
        
        #the joint has a mean of zero and a covariance matrix as specified in the notes
        

        al = np.concatenate (((np.dot(W.T,W)+ Psi),W.T), axis = 1)
        bl = np.concatenate ((W,np.identity(K)), axis = 1)
        cov_mat = np.concatenate ((al,bl), axis = 0)
        
        #computing the joint log probability of Xs and Zs
        
        #print X, Z
        X_Z = np.concatenate((X,Z), axis = 1) 
        #jl = (autograd.scipy.stats.multivariate_normal.logpdf(X_Z,np.zeros(D+K), cov_mat))

        inv_cov_mat = np.linalg.inv(cov_mat)
        sgn, logdet = np.linalg.slogdet(2*np.pi*cov_mat)
        norm_term = -0.5*(logdet)*sgn
        #norm_term = -0.5*np.abs(logdet)
        jl = (norm_term)+(-0.5*(np.diag(np.dot(np.dot((X_Z),inv_cov_mat),X_Z.T))))
        #print  jl , N, K
        return jl
    
    def marginal_likelihood(self, X, W=None, Psi=None):
        '''
        function: marginal_likelihood
        Description: Compute the marginal likelihood for given data X.
        Inputs:
            X -   (np.array) Data matrix. Shape (N,D)
            W -   (np.array) Factor loading matrix. Shape (K,D).
            Psi - (np.array) Output covariance matrix. Shape (D,D). Positive, diagonal.
        Outputs:  
            ml -  (np.array) Array of shape (N,) where ml[n] is the marginal
                  log likelihood of data case X[n,:].
        '''
        #print 'ML'
        
        N,D = X.shape 
        
        if(W is None): W = self.W
        if(Psi is None): Psi = self.Psi
        #ml = np.zeros(N)
        cov_mat = np.dot(W.T,W)+ Psi
        inv_cov_mat = np.linalg.inv(cov_mat)
        #norm_term = (1/(np.sqrt(np.power(2*np.pi,D)*np.linalg.det(cov_mat))))
        sgn, logdet = np.linalg.slogdet(2*np.pi*cov_mat)
        norm_term = -0.5*(logdet)*sgn
        second_term = (-0.5*(np.diag(np.dot(np.dot(X,inv_cov_mat),X.T))))
        ml1 = (norm_term)+second_term
        #ml2 = (autograd.scipy.stats.multivariate_normal.logpdf(X,np.zeros(D), cov_mat))
        #print norm_term, second_term
        
        
        return ml1
    
    def svi_obj(self, X, W, Psi, mu, std,  num_samples=1000):
        '''
        function: svi_obj
        Description: Compute the stochastic variational inference objective
        function for data X, model parameters W and Psi, and variational
        parameters mu and std using the re-parameterization trick.
        Uses num_samples samples in the stochastic approximation. 
        
        The variational approximation is based on a diagonal Gaussian with 
        mean mu and vector of standard deviations std. The covariance of the 
        variational distribution is thus np.diag(std**2).
        
        Inputs:
            X -   (np.array) Data matrix. Shape (1,D)
            W -   (np.array) Factor loading matrix. Shape (K,D).
            Psi - (np.array) Output covariance matrix. Shape (D,D). Positive, diagonal.
            mu -  (np.array) Variational mean parameters. Shape(1,K). 
            std - (np.array) Variational standard deviation parameters. Shape(1,K).
            num_samples - (int) Number of samples to use in approximation
        Outputs:  
            obj - (float) Value of the stochastic variational inference objective function.
        '''        
        #print 'svi_obj'
        K = self.K
        D = self.D
        #print std
        eps_t = np.random.multivariate_normal(np.zeros(K), np.identity(K), num_samples)
        #print 'here'
        Q_Z = 0.5*K*np.log(2*np.pi*np.exp(1)) + 0.5*(np.sum(np.log(std[0]**2)))
        X_D = np.repeat(X,num_samples,axis=0)
        Z_new = (mu + np.dot(eps_t,np.diag(std[0])))
        #print 'also here', 'Z',Z_new.shape,'X', X_D.shape, K, D,         
        P = np.mean(self.joint_likelihood(X = X_D, Z = Z_new,W=W, Psi=Psi))
        obj = (-Q_Z - P)
        #print obj
        return obj
    
    def svl_obj(self, X, mus, stds, W, Psi): 
        '''
        function: svl_obj
        Description: Compute the stochastic variational learning objective
        function for data X, model parameters W and Psi, and collection of 
        variational parameters mus and stds using the re-parameterization trick.
        Use one sample per data case in the stochastic approximation. 
        
        The variational approximation is based on a diagonal Gaussian with 
        mean mus[n,:] and vector of standard deviations stds[n,:] for each 
        data case n. The covariance of the variational distribution 
        for data case n is thus np.diag(stds[n,:]**2).
        
        Inputs:
            X -   (np.array) Data matrix. Shape (N,D)
            mus - (np.array) Variational mean parameters. Shape(N,K). 
            stds- (np.array) Variational standard deviation parameters. Shape(N,K).
            W -   (np.array) Factor loading matrix. Shape (K,D).
            Psi - (np.array) Output covariance matrix. Shape (D,D). Positive, diagonal.
        Outputs:  
            obj - (float) Value of the stochastic variational learning objective function.
                  Use an average over the data cases.
        '''         
        #print 'svl_obj'
        N,_ = X.shape
        K = self.K
        D = self.D
        eps_t = np.random.multivariate_normal(np.zeros(K), np.identity(K), N)
        #print 'here'
        Q_Z = 0.5*K*np.log(2*np.pi*np.exp(1)) + 0.5*(np.sum(np.log(stds**2),axis=1))
        X_D = X
        Z_new = (mus + np.multiply(eps_t,stds))
      
        obj = np.mean(Q_Z+ self.joint_likelihood(X = X_D, Z = Z_new,W=W, Psi=Psi))
     

        return obj
          
    
    def infer(self,x, W=None, Psi=None, method="exact"):
        '''
        function: infer
        Description: Run inference to obtain the posterior distribution for 
        a single data case x. method can either be  "exact" for  exact
        inference, or "bbsvi" for black-box stochastic variational inference.
        Output is a tuple consisting of the posterior mean and  the posterior
        covariance matrix.
            
        Inputs:
            x -    (np.array) Data matrix. Shape (1,D)
            W -    (np.array) Factor loading matrix. Shape (K,D).
            Psi -  (np.array) Output covariance matrix. Shape (D,D). Positive, diagonal.
            method-(string) Either  "exact" or "bbsvi" 
        Outputs:  
            mu    - (np.array) Value of the exact or approximate posterior mean. Shape (1,D)
            Sigma - (np.array) Value of the exact or approximate posterior 
                    covariance matrix. Shape (D,D)
        '''
        if(W is None): W = self.W
        if(Psi is None): Psi = self.Psi        
        K = self.K
        D = self.D
        if method == "exact":
            #print 'exact'
            inter = np.linalg.inv(np.dot(W.T,W) + Psi)
            mean_conditional = (np.dot(W,np.dot(inter,x.T))).T
            cov_conditional = np.identity(K) - np.dot(W,np.dot(inter,W.T))
            return mean_conditional, cov_conditional

        elif method  == "bbsvi":
            #print 'bbsvi'
            init_mean = np.random.randn(1,K)/100
            init_log_std = 1e-5*np.ones((1,K))
            init_var_params = np.concatenate((init_mean.flatten(), init_log_std.flatten()))
            gradient = self.svi_wrapper(x, init_var_params, W, Psi)
            variational_params = adam(gradient, init_var_params, num_iters = 1000)
            return variational_params[:K] , np.diag((np.exp(variational_params[K:])**2))
        else: print 'invalid method'
        pass
    
    def svi_wrapper(self, X, init_var_params,W = None, Psi = None):

        if(W is None): W = self.W
        if(Psi is None): Psi = self.Psi
        def grada(params, t):
            SVI = self.svi_obj(X=X,W=W, Psi=Psi, mu = params[:self.K].reshape(1,self.K), std = np.exp(params[self.K:]).reshape(1,self.K))
            print SVI
            return SVI
        gradient = grad(grada)
        return gradient

    def svl_wrapper(self, X, init_var_params,W = None, Psi = None):

        if(W is None): W = self.W
        if(Psi is None): Psi = self.Psi
        self.N,_ = X.shape
        def grada(params, t):
            #print 'here'
            W = params[self.D:self.D*(self.K+1)].reshape((self.K,self.D))
            Psi = np.diag(np.exp(params[:self.D].reshape(self.D)))
            mus = params[-2*self.N*self.K:-self.N*self.K].reshape((self.N,self.K))
            stds = np.exp(params[-(self.N*self.K):].reshape((self.N,self.K)))
            SVL = self.svl_obj(X=X,W=W, Psi=Psi, mus = mus, stds = stds)

            return -SVL
        gradient = grad(grada)
        return gradient


    def fit(self, X, method="exact"):
        '''
        function: fit
        Description: Fit the model to the data in X. method can either be  "exact" 
        for standard maximum likelihood learning using the exact marginal log likelihood,
        or "bbsvl" for black-box stochastic variational learning using diagonal
        Gaussian variational posteriors. The optimized W and Psi parameters should be stored
        in member variables W and Psi after learning. 
            
        Inputs:
            X -    (np.array) Data matrix. Shape (N,D)
        Outputs:   None
        '''        
        K = self.K
        D = self.D

        
        gamma = np.log(np.diag(np.cov(X.T)))
        #gamma = np.random.randn(D)*1e-5
        N,_ = X.shape
        W = np.random.randn(K,D)*1e-5
        
        if method == "exact":
            #gamma = np.log(np.diag(np.cov(X.T)))
            init_params = np.concatenate((gamma.flatten(),W.flatten()))
            fprime = (self.marginal_likelihood_wrapper(init_params, X))       
            learnt_params = adam(fprime, init_params)
            self.W = learnt_params[D:].reshape(K,D)
            self.Psi = np.diag(np.exp(learnt_params[:self.D]))
        elif method == "bbsvl":
            #gamma = np.log(np.diag(np.cov(X.T)))
            mus = np.random.randn(X.shape[0],X.shape[1])/100
            stds = np.random.randn(X.shape[0], X.shape[1])*1e-5          
            init_var_params = np.concatenate((gamma.flatten(),W.flatten(),mus.flatten(),stds.flatten()))
            fprime = (self.svl_wrapper(X, init_var_params))
            learnt_params = adam(fprime, init_var_params)
            self.W = learnt_params[D:(D*K+D)].reshape((K,D))
            self.Psi = np.diag(np.exp(learnt_params[:D].reshape(D)))
            self.mus = learnt_params[D*(K+1):D*(K+1)+N*K].reshape((N,K))
            self.stds = np.exp(learnt_params[-(N*K):].reshape((N,K)))
        else: print 'invalid method'
        pass  
    
    def marginal_likelihood_wrapper(self, init_params, X):
        def mlgrad(params, t):
            ml = self.marginal_likelihood( X, W = params[self.D:].reshape(self.K,self.D), Psi= np.diag(np.exp(params[:self.D])))
            return -(ml.sum())
        return grad(mlgrad)