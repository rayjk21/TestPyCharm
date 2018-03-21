
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt


# Compares x with Normal dist, under NULL hypothesis that distributions are IDENTICAL
np.random.seed(123)
x = stats.norm.rvs(size=1000)
stats.kstest(x, 'norm')
# >> KstestResult(statistic=0.031007551695833357, pvalue=0.2863763836872417)
# pValue is not <0.05 so cant reject Ho => Is normal
# plt.hist(x)


# Compare x with norm dist => k, pValue
x = np.linspace(-15, 15, 9)
stats.kstest(x, 'norm')
# >> KstestResult(statistic=0.4443560271592436, pvalue=0.038850142705171065)
# pValue is  <0.05 so reject Ho => Is NOT normal


stats.kstest(x,'norm', alternative = 'less')





np.random.seed(12345678)
rvs1 = stats.norm.rvs(loc=0,scale=1,size=500)
rvs2 = stats.norm.rvs(loc=0,scale=1,size=500)
t = stats.ttest_ind(rvs1,rvs2)
print(t)
# => (0.26833823296239279, 0.78849443369564776)
# Accept Ho: Same
plt.hist(rvs1)
plt.hist(rvs2)


# T-test with different variance
rvs1 = stats.norm.rvs(loc=0,scale=1,size=500)
rvs2 = stats.norm.rvs(loc=0,scale=40,size=500)
plt.hist(rvs1)
plt.hist(rvs2)
stats.ttest_ind(rvs1,rvs2, equal_var = False)
# Ttest_indResult(statistic=0.17855731962721777, pvalue=0.8583501513212636)
stats.ttest_ind(rvs1,rvs2)
# Ttest_indResult(statistic=0.17855731962721774, pvalue=0.8583215490057864)
# Not much difference with default








############# Plot distribution

def norm_cdf(mean=0.0, std=1.0):
    # 50 numbers between -3σ and 3σ
    x = np.linspace(-3*std, 3*std, 50)
    # CDF at these values
    y = stats.norm.cdf(x, loc=mean, scale=std)

    plt.plot(x,y, color="black")
    plt.xlabel("Variate")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF for Gaussian of mean = {0} & std. deviation = {1}".format(
               mean, std))
    plt.draw()

norm_cdf()






#############  Modelling Metrics  ###############
from sklearn.metrics import precision_recall_fscore_support as score

predicted = [1,2,3,4,5,1,2,1,1,4,5]
y_test = [1,2,3,4,5,1,2,1,1,4,1]

precision, recall, fscore, support = score(y_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
