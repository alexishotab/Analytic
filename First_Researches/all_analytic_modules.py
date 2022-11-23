import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns


def mediana(sample):
    return sample.median()


def moda(sample):
    return sample.mode()


def avarage(sample):
    """
    Mean of Data
    """
    return sample.mean()


def razmah(sample):
    return max(sample) - min(sample)


def D(sample):
    """
    dispersion
    """
    xsr = avarage(sample)
    cnt = 0
    for i in range(len(sample)):
        cnt += (xsr - sample[i]) ** 2
    return cnt / len(sample)


def sigma(sample):
    """
    standart deviation in general data
    """
    return D(sample) ** 0.5


def sd(sample):
    """
    standart deviation in nongeneral data
    """
    xsr = avarage(sample)
    cnt = 0
    for i in range(len(sample)):
        cnt += (xsr - sample[i]) ** 2
    d = cnt / (len(sample) - 1)
    return d ** 0.5


def se(sample):
    """
    standart deviation. We change of sd and sigma
    """
    if len(sample) > 30:
        return sd(sample) / len(sample) ** 0.5
    else:
        return sigma(sample) / len(sample) ** 0.5


def truerange():
    """
    confidence interval
    """
    print ('Введите стандартную ошибку (se):')
    srednee = float(input())
    print ('Введите вашe число:')
    x = float(input())
    print('Введите вашу последовательность в строку через пробел:')
    data = np.array(list(map(int, input().split())))
    xsr = sum(data) / len(data)
    print (('Доверительный интервал = [' , round(xsr - 1.96 * srednee, 4), ' ', round(xsr + 1.96 * srednee, 4), ']'))
    return (xsr - 1.96 * srednee, xsr + 1.96 * srednee)


def veroyatnost_of_one_value_general(sample):
    """
    probability of case in general data
    """
    print ('Покажет, какой процент людей обладает нужным параметром')
    mean = avarage(sample)
    std = se(sample)
    x = int(input( 'Введите значение' ))
    # sf - Survival function = (1 - cdf) - Cumulative distribution function
    return stats.norm(mean, std).sf(x) * 100


def t_veroyatnost_of_one_value():
    """
    probability of case in nongeneral data, len(data)<30
    """
    print ('Введите вашу последовательность в строку через пробел:')
    data = np.array(list(map(int, input().split())))
    print('Введите предполагаемое среднее значение генеральной совокупности:')
    u = float(input())
    xsr = sum(data) / len(data)
    df = len(data) - 1
    t = (xsr - u) / se(data)
    p = 2 * stats.t.sf(abs(t), df)
    return p


def t_test(sample1, sample2):
    """
    checking the equality of the means in two samples
    """
    sd1 = sd(sample1)
    sd2 = sd(sample2)
    SE = (sd1 ** 2 / len(sample1) + sd2 ** 2 / len(sample2)) ** 0.5
    xsr1 = avarage(sample1)
    xsr2 = avarage(sample2)
    T = (xsr1 - xsr2) / SE
    df = len(sample1) + len(sample2) - 2
    DF = \
        pd.DataFrame({'Выборка1': sample1,
                     'Выборка2': sample2}).agg(['mean'
            , 'std', 'count', 'sem']).transpose()
    DF.columns = ['Mx', 'SD', 'N', 'SE']
    K = stats.t.ppf(0.975, DF['Mx'] - 1)
    print (DF)
    p = stats.t.sf(T, df)
    print ('P-уровень значимости =',p)
    if p >= 0.05:
        print ('Мы не можем отклонить нулевую гипотезу')
    else:
        print ('Можем отклонить нулевую гипотезу')
    DF['interval'] = K * DF['SE']
    a = plt.boxplot([sample1, sample2], vert=True, patch_artist=True,
                    labels=['Выборка 1'
                    ,
                    'Выборка 2'
                    ])
    plt.show()
    b = plt.errorbar(
        x=['Выборка 1'
           ,
           'Выборка 2'
           ],
        y=DF['Mx'],
        yerr=DF['interval'],
        capsize=3,
        mfc='red',
        mec='black',
        fmt='o',
        )
    plt.show()
    return p


def histogramma(sample):
    plt.hist(sample, bins=np.arange(min(sample), max(sample) + 1, 1))
    plt.show()
    plt.hist(sample, bins=np.arange(min(sample), max(sample) + 1, 1),
             density=True)
    plt.show()


def Z_graph(sample):
    """
    Z-change of general data graphic
    """
    xsr = avarage(sample)
    zmassive = []
    for i in range(len(sample)):
        z_value_i = (sample[i] - xsr) / sd(sample)
        zmassive.append(z_value_i)
    sns.histplot(data=sample, kde=True)
    plt.show()
    return zmassive


def diagramma_boxplot(sample):
    plt.boxplot(sample, showfliers=1)
    plt.show()

def allavarage(sample):
    """
    one-way analysis of variance method to see mean of ALL data
    """
    n=sample.size
    cnt=0
    a=sample.sum()
    for i in a:
        cnt+=i
    return cnt/n

def SST(sample):
    """
    Sum of Squares Total
    """
    cnt=0
    xsr=allavarage(sample)
    a=sample.to_numpy()
    for i in range(len(a)):
        for j in range(len(a[i])):
            cnt+=((a[i][j]-xsr)**2)
    DF=sample.size-1
    return cnt

def SSW(sample):
    """
    Sum of Squares Between groups. Return SS, Degree of Freedom
    """
    a=sample.to_numpy()
    a=a.T
    cnt=0
    for i in range(len(a)):
        xsr=sum(a[i])/len(a[i])
        for j in range(len(a[i])):
            cnt+=((a[i][j]-xsr)**2)
    a=a.T
    DF=sample.size-i-1
    return cnt,DF

def SSB(sample):
    """
    Sum of Squares Within groups. Return SS, Degree of Freedom
    """
    a=sample.to_numpy()
    cnt=0
    srednee=allavarage(sample)
    a=a.T
    for i in range(len(a)):
        xsr=sum(a[i])/len(a[i])
        cnt+=(len(a[i])*((xsr-srednee)**2))
    a=a.T
    DF=i
    return cnt,DF

def f_val(sample):
    """
    Returns Value of F, Degree of Freedom Between, Degree of Freedom Within group
    """
    q,w=SSB(sample)
    o,i=SSW(sample)
    f_v= (q/w)/(o/i)
    return f_v, w,i

def factor_dispersion_analys(sample):
    """
    Returns P-value of F-test
    """
    f_value,dfb,dfw= f_val(sample)
    print(f_value,dfb,dfw)
    p=stats.f.sf(f_value,dfb,dfw)
    return p
