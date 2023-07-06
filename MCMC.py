import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
from scipy.integrate import odeint
import pandas as pd;
import concurrent.futures;
import time;

class MCMC:
    def __init__(self,):

        self.LL = pfilter_LL;

        # tracks all tested parameter sets--appends the initial guess to start
        self.param_set = [];

        # tracks the log likelihood of the param set
        self.LL_set = [];

        # tracks all param guesses--even if rejected
        self.all_guesses = [];

        self.iter = 0;

        self.scale = np.array([0.1]);

    def metropolis(self):
        num_accept = 0;
        for i in range(1, self.iter):
            # monitor progress
            print("Percent complete: {0}".format(round((i / self.iter) * 100, 2)));
            print("acceptance rate: {0}\n".format(round((num_accept / i) * 100, 2)));
            if num_accept / i > .234:
                self.scale *= 1.1;
            else:
                self.scale *= 0.9;
            paramtest_1 = abs(np.random.normal(loc=self.param_set[-1][0], scale= self.scale[0]));
            paramtest_2 = abs(np.random.normal(loc=self.param_set[-1][1], scale = self.scale[0]));


            paramtest = [paramtest_1,paramtest_2];

            #compute the log likelihood of the paramtest given the data
            LL_test = pfilter_LL(paramtest);
            #acceptance criteria
            accept = min(1, np.exp(LL_test - self.LL_set[-1]))
            #acceptance check
            if np.random.random() < accept:
                num_accept += 1;
                self.LL_set.append(LL_test);
                self.param_set.append(paramtest);
            else:
                self.LL_set.append(self.LL_set[-1]);
                self.param_set.append(self.param_set[-1]);

            self.all_guesses.append(self.param_set[-1]);

    def run(self,iter):
       self.iter = iter;
       #scale values for the variance of the prior distributions
       self.scale = np.array([0.1]);
       self.param_set.append([0,0]);
       self.LL_set.append(-50000);
       #Metropolis algorithm

       self.metropolis();
       print(self.param_set);



class model:
    def __init__(self,S_0,I_0,R_0,params):
        self.S = [];
        self.I = [];
        self.R = [];
        self.N = [];
        self.reports = [];
        self.t = [];

        self.S.append(S_0);
        self.I.append(I_0);
        self.R.append(R_0);
        self.N.append(S_0 + I_0 + R_0);
        self.reports.append(0);
        self.t.append(0);
        self.params = params;


    def simulate(self):

        #rates

        #accumulators
        tInf = 0;
        tRem = 0;

        if self.S[-1] != 0 and (0<self.params[0]<1):
            for i in range(0,self.S[-1]):
                inf = np.random.binomial(1,self.params[0]);
                tInf += inf;

        if self.I[-1] != 0  and (0<self.params[1]<1):
            for i in range(0,self.I[-1]):
                rem = np.random.binomial(1,self.params[1]);

                tRem += rem;

        self.reports.append(np.random.binomial(self.I[-1], 0.5));
        self.S.append(self.S[-1] - tInf);
        self.I.append(self.I[-1] + tInf - tRem);
        self.R.append(self.R[-1] + tRem);

class pfilter:

    def __init__(self,num_particles):
        self.real_measurements = pd.read_csv("measurements.csv")['reports'];
        self.real_measurements = self.real_measurements.values.tolist()


        self.weights = np.ones(num_particles, dtype=float );
        self.num_particles = num_particles;


        self.samples = np.zeros(num_particles, dtype=(int,3));


        self.create_samples();
        self.measurements = np.zeros(num_particles,dtype = int);


        self.log_likelihood = 0;
        self.reports = [];

        self.average_S = [];
        self.average_I = [];
        self.average_R = [];

    def create_samples(self):
        for i in range(self.num_particles):
            pop = 100;
            rand_S = random.randint(0,pop);
            rand_I = pop-rand_S;
            self.samples[i] = [rand_S,rand_I,0];

    def predict(self,params):
        for i in range(self.num_particles):
            SIRmodel = model(self.samples[i][0],self.samples[i][1],self.samples[i][2],params);
            if SIRmodel.I[-1] + SIRmodel.S[-1] > 1:
                SIRmodel.simulate();
                self.samples[i] = [SIRmodel.S[-1],SIRmodel.I[-1],SIRmodel.R[-1]];

                self.measurements[i] = np.random.binomial(SIRmodel.I[-1],0.5);



    def update(self,real_measurement):
        self.weights = np.ones(self.num_particles);
        for i in range(0,self.num_particles):
            nd = norm(loc=real_measurement, scale=1);
            self.weights[i] = nd.pdf(self.measurements[i]);

        self.weights += 1.e-300  # avoid round-off to zero
        self.log_likelihood += np.log(sum(self.weights) / len(self.weights));
        self.weights /= sum(self.weights)


    def resample(self):
        indexes = np.zeros(self.num_particles);
        for i in range(self.num_particles):
            indexes[i] = i;
        new_sample_indexes = np.random.choice(a=indexes, size=self.num_particles, replace=True, p=self.weights);
        sample_copy = np.copy(self.samples);
        for i in range(len(self.samples)):
            self.samples[i] = sample_copy[int(new_sample_indexes[i])];

    def average(self):
        average_S = 0;
        average_I = 0;
        average_R = 0;
        average_reports = 0;
        for sample in self.samples:
            average_S += sample[0];
            average_I += sample[1];
            average_R += sample[2];
        average_S = average_S/self.num_particles;
        average_I = average_I / self.num_particles;
        average_R = average_R / self.num_particles;
        average_reports = np.sum(self.measurements);
        average_reports = average_reports / self.num_particles;

        self.average_S.append(average_S);
        self.average_I.append(average_I);
        self.average_R.append(average_R);
        self.reports.append(average_reports);


    def run(self,tend,params):

        for i in range(0, tend):


            self.predict(params=params);

            self.update(self.real_measurements[i]);
            self.resample();
            self.average();


        return [self.log_likelihood,self.average_S,self.average_I,self.average_R,self.real_measurements,self.reports];

def pfilter_run_par(num_filters,num_particles,params):
    pfilters = [];
    process_outputs = [];

    for _ in range(num_filters):
        pfilters.append(pfilter(num_particles));


    with concurrent.futures.ProcessPoolExecutor() as executor:
        for filter in pfilters:
            process_outputs.append(executor.submit(filter.run, 100,params));

    return process_outputs;

def logmeanexp(log_likelihoods):
    likelihoods = [];
    for ll in log_likelihoods:
        likelihoods.append(np.exp(ll));
    mean = np.sum(likelihoods)/len(likelihoods);
    return np.log(mean);

def pfilter_LL(params):

    process_outputs = pfilter_run_par(5, 100,params);
    log_likelihoods = [];
    for output in process_outputs:
        result = output.result();
        log_likelihoods.append(result[0]);

        for i in range(1, 6):
            plt.plot(result[i]);

    print(log_likelihoods);
    print(params);

    return np.sum(log_likelihoods)/5;


def main():
    MCMC_model = MCMC();
    MCMC_model.run(10000);



if __name__ == "__main__":
    main()
