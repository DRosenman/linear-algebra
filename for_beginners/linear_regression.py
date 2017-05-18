# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:05:56 2017

@author: rosen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

class dataset:

    def __init__(self,x,y = []):

        if type(x[0]) == tuple:
           x,y = zip(*x)

           self.x = np.array(x)
           self.y = np.array(y)

        elif type(x[0]) == list:
            temp = np.array(x)
            self.x = temp[:,0]
            self.y = temp[:,1]

        else:
            self.x = np.array(x)
            self.y = np.array(y)
        self.slope = slope(self.x,self.y)
        self.slope_error = slope_error(self.x,self.y)
        self.intercept = intercept(self.x,self.y)

        self.slope_00 = slope(self.x,self.y,through_origin = True)
        self.slope_error00 = slope_error(self.x,self.y,through_origin = True)

        self.xlabel = 'x'
        self.xunits = ''
        self.ylabel = 'y'
        self.yunits = ''




    def __str__(self):
        return ("slope = " + str(self.slope) + "\nslope std. error = "
              + str(self.slope_error) + "\nintercept = " + str(self.intercept))
    def results(self,with_through_origin = False,through_origin = False):
        if with_through_origin == False:
            print_summary_stats(self.x,self.y,through_origin)
        else:
            print_summary_stats_combined(self.x,self.y)


    def graph(self,title = '',through_origin = False):
        fig = plt.figure()
        if self.x.min() < 0:
            x_min = self.x.min() *.98
        else:
            x_min = 0.0
        x_max = self.x.max() * 1.02
        x = np.linspace(x_min,x_max,100)

        if through_origin == False:
            m = self.slope
            c = self.intercept
            name = 'Best Fit Line'
            equation = self.ylabel + " = " + str(round(m,3)) + '$\\cdot$' + self.xlabel + ' + ' + str(round(c,3))
        else:
            m = self.slope_00
            c = 0.0
            name = 'Best Fit Line Through Origin'
            equation = self.ylabel + " = " + str(round(m,3)) + '$\\cdot$' + self.xlabel

        if self.xunits == '':
            xlabel = self.xlabel
            ylabel = self.ylabel
        else:
            xlabel = self.xlabel + ' ' + '(' + self.xunits + ')'
            ylabel = self.ylabel + ' ' + '(' + self.yunits + ')'
        plt.grid()
        axis_font = {'family': 'serif','color':  'darkred','weight': 'bold','size': 12,
        }
        font_big = {'family': 'serif','color':  'darkred','weight': 'bold','size': 16,
        }

        plt.xlim(x_min,1.05*x_max)
        plt.xlabel(xlabel,fontdict = axis_font)
        plt.ylabel(ylabel,fontdict = axis_font)

        plt.scatter(self.x,self.y,label = 'Measurement', color = 'darkred')
        plt.plot(x,x*m+c,label = name, )
        plt.title(title,fontdict = font_big)
        ax = plt.gca()
        plt.text(0.5, 0.75,equation,color = 'darkred', ha='center', va='center', transform=ax.transAxes)
        plt.legend();

def slope(x,y,through_origin = False):
    x = np.array(x)
    y = np.array(y)
    if through_origin == False:
        x = np.array(x)
        y = np.array(y)
        return ((x-x.mean())*y).sum()/((x-x.mean())**2).sum()
    else:
        return ((x*y).sum())/(x**2).sum()

def intercept(x,y,through_origin = False):
    if through_origin == False:
        x = np.array(x)
        y = np.array(y)
        return y.mean() - slope(x,y)*x.mean()
    else:
        return 0.0

def slope_error(x,y,through_origin = False):
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    if through_origin == False:
        residuals_squared = (y - slope(x,y)*x - intercept(x,y))**2
        residuals_squared_sum = residuals_squared.sum()
        D = ((x-x.mean())**2).sum()
        return np.sqrt((1/(n-2))*residuals_squared_sum/D)
    else:
        residuals_squared = (y - slope(x,y,True)*x)**2
        return np.sqrt(residuals_squared.sum()/((n-1)*(x**2).sum()))

def intercept_error(x,y,through_origin = False):
    if through_origin == False:
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        D = ((x-x.mean())**2).sum()
        residuals_squared = (y - slope(x,y)*x - intercept(x,y))**2
        return np.sqrt(((1/n) + (x.mean()**2)/D)*residuals_squared.sum()/(n-2))
    else:
        return 'NaN'

def summary_stats(x,y,through_origin = False):
    return (slope(x,y,through_origin),intercept(x,y,through_origin),
            slope_error(x,y,through_origin),intercept_error(x,y,through_origin),
                       correlation_coefficient(x,y,through_origin))

def summary_stats_df(x,y,both = False,through_origin = False):
    x = np.array(x)
    y = np.array(y)
    if both == False:
        if through_origin == False:
            df = pd.DataFrame([[slope(x,y),intercept(x,y),slope_error(x,y),intercept_error(x,y),correlation_coefficient(x,y)]],
                            columns = ['Slope', 'Intercept','Std. Error, Slope','Std Error, Intercept','R'],
                                       index = [''])
            return df
        else:
            df = pd.DataFrame([[slope(x,y,through_origin),intercept(x,y,through_origin = True),slope_error(x,y,through_origin),correlation_coefficient(x,y,through_origin)]],
                            columns = ['Slope', 'Intercept','Std. Error, Slope','R'],
                                       index = [''])
            return df
    if both == True:
        df = pd.DataFrame([[slope(x,y),intercept(x,y),slope_error(x,y),intercept_error(x,y),correlation_coefficient(x,y)],
                                [slope(x,y,True),intercept(x,y,True),slope_error(x,y,True),'NA',correlation_coefficient(x,y,True)]],
                                columns = ['Slope', 'Intercept','Std. Error, Slope','Std Error, Intercept','R'],
                                index = ['Regular Linear Regression','Regression Line Through (0,0)'])
        return df

def print_summary_stats(x,y,through_origin = False):
    if through_origin == False:
        display(pd.DataFrame([[slope(x,y),intercept(x,y),slope_error(x,y),intercept_error(x,y),correlation_coefficient(x,y)]],
                            columns = ['Slope', 'Intercept','Std. Error, Slope','Std. Error, Intercept', 'r'],
                                       index = ['']))

    if through_origin == True:
        display(pd.DataFrame([[slope(x,y,through_origin = True),intercept(x,y,through_origin = True),correlation_coefficient(x,y,through_origin = True),
                            slope_error(x,y,through_origin = True)]],
                            columns = ['Slope', 'Intercept','Std. Error, Slope','r'],
                            index = ['']))


def print_summary_stats_combined(x,y):
     display(pd.DataFrame([[slope(x,y),intercept(x,y),slope_error(x,y),intercept_error(x,y),correlation_coefficient(x,y)],
                          [slope(x,y,through_origin = True),intercept(x,y,through_origin = True),slope_error(x,y,through_origin = True),'---',correlation_coefficient(x,y,through_origin = True)]],
                            columns = ['Slope', 'Intercept','Std. Error, Slope','Std. Error, Intercept', 'r'],
                                       index = ['Regular Linear Regression','Regression Line Through (0,0)']))


def correlation_coefficient(x,y,through_origin = False):
    x = np.array(x)
    y = np.array(y)
    if through_origin == False:
        n = len(x)
        xy = x*y
        return (n*(xy.sum()) - (x.sum())*(y.sum()))/((((n*(x**2).sum()) - (x.sum())**2)**.5)*((n*((y**2).sum()) - (y.sum())**2)**.5))
    else:
        return np.sqrt(1.0 - (((y - x*slope(x,y,through_origin))**2).sum())/((y**2).sum()))


def r(x,y,through_origin = False):
    return correlation_coefficient(x,y,through_origin)

def r_squared(x,y,through_origin = False):
    return (correlation_coefficient(x,y,through_origin))**2

import scipy.stats as stats

def p_value(x,y,through_origin = False):
    x = np.array(x)
    y = np.array(y)
    if through_origin == True:
        k = 1
    else:
        k = 2
    df = len(x) - k
    m = slope(x,y,through_origin)
    error = slope_error(x,y,through_origin)
    t = m/error
    return stats.t.sf(np.abs(t), df)*2

def import_csv(file):
    df  = pd.read_csv(file, header = None)
    return df.values[:,0],df.values[:,1]
    
  
