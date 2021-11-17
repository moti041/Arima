import math
import numpy as np
import torch

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.arima.model as stats
import torch.distributions.normal as norm

class Arima():
    def __init__(self,len):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        self.len=len
        self.er = torch.randn(1,self.len+1).numpy()
        self.er_t = torch.from_numpy(self.er)[:,1:]
        self.er_t_m1 =  torch.from_numpy(self.er)[:,:-1]

    def generate(self):
        """
        In the generate function we  return samples which derived from ARIMA model.
        """
        er_mat = torch.cat((self.er_t_m1.T, self.er_t.T), 1)
        coeff = torch.tensor([ [float(i) for i in [self.phi,1]]])
        drf = self.drift * torch.ones((1,self.len) )
        dy = torch.mm(er_mat, coeff.T) +drf.T
        y = torch.cumsum(dy, dim=0)
        return y
    def extract_err_for_model(self, y ,er_0):
        """
        find the errors  which should occur in order to observe the
        last 6 samples (given the best model's  parameters).
                """
        err=[er_0]
        for i in range(0,len(y)-1):
            er = y[i+1] - y[i] -self.model_drift -self.err_coeff * err[i]
            err.append(er)
        return err
    def cdf(self, value):
        """
        Evaluate the cumulative distribution function at the value.
        """
        value=value  / (math.sqrt(2) * 1)
        return (1 + torch.erf(torch.from_numpy(np.array(value)) ).add(1e-5)).mul_(0.5)
    def pdf(self, value):
        """
        Evaluate the cumulative distribution function at the value.
        """
        value=-math.pow(value,2)/2
        return (1/np.sqrt(2*np.pi))*np.exp(value)
    def err_prob(self,err):
        """
               find the probability of errors (err) of the last 6 samples  which should occur in order to observe the
               last 6 samples (given the best model's  parameters).
                       """
        er_prob=1
        for i in range( len(err)):
            temp=self.pdf(err[i])
            er_prob *= temp
        return er_prob
if __name__ == '__main__':
    arima=Arima(20)
    arima.drift= 2
    arima.phi = 0.46
    # generate 20 samples with drift
    y=arima.generate( )
    # fit the model on 14 samples, trend='t' -- for drift case
    model = stats.ARIMA(y.numpy()[:-6], order=(0, 1, 1), trend="t")  # 
    model_fit = model.fit()
    model_fit.summary()
    # last 6 observations  probability based on the given model
    arima.model_drift=model_fit.params[0]
    arima.err_coeff = model_fit.params[1]
    state_prob=0
    rng=[-4,4]
    n=200
    delt=(rng[1]-rng[0])/n
    er_0_prob_normlize=0
    # since we have D.O.F on what will be the first noise value, we should sum all the options
    for er_0_vec in enumerate(np.linspace(rng[0],rng[1],n)):
        er_0=er_0_vec[1]
        er_0_prob_normlize = er_0_prob_normlize + arima.pdf(er_0)
        err=arima.extract_err_for_model( y.numpy()[-6:], er_0)
        er_prob=arima.err_prob(err)#*er_0_pr
        state_prob += er_prob
    # in order to cancel the dependency on 'n' we divide it by er_0_prob_normlize
    state_prob = state_prob/er_0_prob_normlize
    print(f' last 6 observations  probability density  is {state_prob}' )

