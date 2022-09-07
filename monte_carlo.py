import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 8)
import warnings
warnings.filterwarnings("ignore")


class monte_carlo():
    def __init__(self, range_=True, size=10000, distributions=['normal'], **data):
        
        try:
            self.size = size
            self.data = data
            self.range_ = range_
            
            if len(distributions) == 1:
                self.distributions = list(distributions) * len(self.data)
            else:
                self.distributions = list(distributions)

                if len(self.distributions) != len(self.data):
                    raise ValueError
                
        except ValueError:
            print("Arguments are not equal \nDistributions input - {0} , Data input - {1} ".format(len(self.distributions), len(self.data)))

    def generate(self):
        self.gen_data = {}

        for idx, dist in enumerate(self.distributions):

            item = list(self.data.items())[idx][0]    
            mean = np.mean(self.data[item])
            sigma = (mean - self.data[item][0])/3

            if dist == 'normal':
                if self.range_:
                    self.gen_data[item] = np.random.normal(mean, sigma, size=self.size)
                else:
                    self.gen_data[item] = np.random.normal(self.data[item][0], self.data[item][1], size=self.size)
                
            if dist == 'lognormal':
                if self.range_:
                    self.gen_data[item] = np.random.lognormal(mean, sigma, size=self.size)
                else:
                    self.gen_data[item] = np.random.lognormal(self.data[item][0], self.data[item][1], size=self.size)

            if dist == 'exponential':
                if self.range_:
                    self.gen_data[item] = np.random.exponential(1/mean, size=self.size)
                else:
                    self.gen_data[item] = np.random.exponential(self.data[item], size=self.size)

            if dist == 'poisson':
                if self.range_:
                    self.gen_data[item] = np.random.poisson(mean, size=self.size)
                else:
                    self.gen_data[item] = np.random.poisson(self.data[item], size=self.size)

            if dist == 'uniform':
                self.gen_data[item] = np.random.uniform(self.data[item][0], self.data[item][1], size=self.size)

        print('Generated: {0} of data for {1} elements'.format(self.size, len(self.gen_data)))
        return self.gen_data

    def plot_outputs(self, cum=False):

        for component in self.gen_data.keys():
            plt.figure();
            sns.histplot(self.gen_data[component], color='navy', cumulative=cum);
            plt.xlabel(component.upper());
            plt.ylabel('PROBABILITY');
            plt.grid();
            
            