import math
import numpy as np
import torch
import matplotlib.pyplot as plt
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

    def plot_df(self, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
        plt.figure(figsize=(16, 5), dpi=dpi)
        plt.plot(x, y, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()
class Arima_modul(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, er_t,er_t_m1):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        dy=self.d + self.c*er_t_m1 +er_t
        return torch.cumsum(dy, dim=0)

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'dy = {self.d.item()} + {self.c.item()} er_t_m1 + er_t'
if __name__ == '__main__':
    len=20; drift= 2; phi = 0.4
    er = torch.randn(1, len + 1)
    er_t = er[:, 1:]
    er_t_m1 = er[:, :-1]
    dy = drift + phi* er_t_m1 + er_t
    y=torch.cumsum(dy, dim=0)
    model = Arima_modul()

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for t in range(200):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(er_t,er_t_m1)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Result: {model.string()}')


    len_series=200
    arima=Arima(len_series)
    arima.drift= 0
    arima.phi = 0.46
    # generate 20 samples with drift
    y=arima.generate( )
    arima.plot_df( x=[float(i) for i in range(len_series)], y=y, title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.')

    # fit the model on 14 samples, trend='t' -- for drift case
    model = stats.ARIMA(y.numpy()[:-6], order=(0, 1, 1), trend="t")  # 
    model_fit = model.fit()
    model_fit.summary()
    # last 6 observations  probability based on the given model
    arima.model_drift=model_fit.params[0]
    arima.err_coeff = model_fit.params[1]
    state_prob=0
    rng=[-4,4]
    n=800
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
    state_prob = state_prob/n#er_0_prob_normlize
    print(f' last 6 observations  probability density  is {state_prob}' )

