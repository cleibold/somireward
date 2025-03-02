#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:45:37 2020

@author: physiol1
"""

import pandas as pd
from statsmodels.stats.anova import AnovaRM  #pingouin.rm_anova can also be used with python version >3.0
import re
import sys
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison 
import numpy as N

def ANOVA_RM_1way(group1,group2,group3,fig_to_save,uname,fn):
    
    #operate one way repeated measures ANOVA with panda dataframe of data
    #Repeated measures Anova using least squares regression
    
    #The full model regression residual sum of squares is
    #used to compare with the reduced model for calculating the
    #within-subject effect sum of squares [1].
    #Currently, only fully balanced within-subject designs are supported.
    #Calculation of between-subject effects and corrections for violation of
    #sphericity are not yet implemented.
    #Parameters
    #----------
    #data : DataFrame
    #depvar : str
    #    The dependent variable in `data`
    #subject : str
    #    Specify the subject id
    #within : list[str]
    #    The within-subject factors
    #between : list[str]
    #    The between-subject factors, this is not yet implemented
    #aggregate_func : {None, 'mean', callable}
    #    If the data set contains more than a single observation per subject
    #    and cell of the specified model, this function will be used to
    #    aggregate the data before running the Anova. `None` (the default) will
    #    not perform any aggregation; 'mean' is s shortcut to `numpy.mean`.
    #    An exception will be raised if aggregation is required, but no
    #    aggregation function was specified.
    
    statistic,pvalue1=stats.shapiro(group1)    #Shapiro-Wilk test for normality
    statistic,pvalue2=stats.shapiro(group2)
    statistic,pvalue3=stats.shapiro(group3)
    
    if pvalue1>0.05 and pvalue2>0.05 and pvalue3>0.05:
        print 'Datasets normaly distributed'
    else:
        print 'Error! Datasets not normaly distributed'
        print 'group1 pvalue', pvalue1
        print 'group2 pvalue', pvalue2
        print 'group3 pvalue', pvalue3
    
    sub_id=[i+1 for i in xrange(len(group1))]*3   #sub_id: individual trials; independent variables
    iv=N.repeat([1, 2, 3], len(group1))         #comparing group ids; dependent variables
    rt=N.concatenate((group1,group2,group3))    #dataset to compare
    
    df=pd.DataFrame({'id':sub_id, 'RT':rt, 'groups':iv})
    
    
    #df = pd.DataFrame({'patient': N.repeat([1, 2, 3, 4, 5], 4),
     #              'drug': N.tile([1, 2, 3, 4], 5),
      #             'response': [30, 28, 16, 34,
       #                         14, 18, 10, 22,
        #                        24, 20, 18, 30,
         #                       38, 34, 20, 44, 
          #                      26, 28, 14, 30]})
    
    a=AnovaRM(data=df,depvar='RT', subject='id', within=['groups']).fit() 
    print a
    #with print a the result with Fvalue, NumDF (degree of freedom), DENDF, Pr>F
    
    # the F test-statistic is 24.7589 and the corresponding p-value is 0.0000.
    #Since this p-value is less than 0.05, we reject the null hypothesis and 
    #conclude that there is a statistically significant difference in mean response 
    #times between the four drugs.
    
    #    A one-way repeated measures ANOVA was conducted on 5 individuals to 
    #examine the effect that four different drugs had on response time.

    #Results showed that the type of drug used lead to statistically significant 
    #differences in response time (F(3, 12) = 24.75887, p < 0.001).
    
    #Details under link: https://www.statology.org/repeated-measures-anova-python/
    
    #write the ANOVA result in text file to get single values
    orig_stdout = sys.stdout
    f = open('%s_u%s_%s_out.txt' %(fig_to_save,uname,fn), 'w')
    sys.stdout = f
    print a
    sys.stdout = orig_stdout
    f.close() 
    
    '''
    with open('out.txt') as f:
    if 'F Value' in f.read():
        print("true")
        '''
    
    #get the values of Fvalue, NumDF, DENDF, Pr>F
    with open('%s_u%s_%s_out.txt' %(fig_to_save,uname,fn)) as f:
        data=f.read()
        #data=data.replace('=','')     #optional: if necessary, remove the lines and so on.
        #data=data.replace('-','')
        #data=data.replace('\n','')
        #data=data.replace('Anova','')
        #data=data.replace('','')
    num = [float(s) for s in re.findall(r'-?\d+\.?\d*', data)]  #find all digital values
    #[24.7589, 3.0, 12.0, 0.0]
    if num[-1]<0.05:
        print 'Significant difference between groups: F(%s,%s):%s, p=%s' %(num[1],num[2],num[0],num[-1])
    else:
        print 'No significance between groups: F(%s,%s): %s, p=%s' %(num[1],num[2],num[0],num[-1])
    
    pvalue = num[-1]
    Num_df = num[1]
    DEN_df=num[2]
    F_value=num[0]
    
    #Multicomparison with Holm-Bonferroni method
    MultiComp = MultiComparison(df['RT'],df['groups'])
    comp = MultiComp.allpairtest(stats.ttest_rel, method='Holm')
    print comp[0]
    [pvalue12,pvalue13,pvalue23]=comp[1][2]
    
    return pvalue, Num_df, DEN_df, F_value, pvalue12, pvalue13, pvalue23

def ANOVA_RM_1way_or_kruskal(group1,group2,group3,fig_to_save,uname,fn,script):
    
    #operate one way repeated measures ANOVA with panda dataframe of data
    #Repeated measures Anova using least squares regression
    
    #The full model regression residual sum of squares is
    #used to compare with the reduced model for calculating the
    #within-subject effect sum of squares [1].
    #Currently, only fully balanced within-subject designs are supported.
    #Calculation of between-subject effects and corrections for violation of
    #sphericity are not yet implemented.
    #Parameters
    #----------
    #data : DataFrame
    #depvar : str
    #    The dependent variable in `data`
    #subject : str
    #    Specify the subject id
    #within : list[str]
    #    The within-subject factors
    #between : list[str]
    #    The between-subject factors, this is not yet implemented
    #aggregate_func : {None, 'mean', callable}
    #    If the data set contains more than a single observation per subject
    #    and cell of the specified model, this function will be used to
    #    aggregate the data before running the Anova. `None` (the default) will
    #    not perform any aggregation; 'mean' is s shortcut to `numpy.mean`.
    #    An exception will be raised if aggregation is required, but no
    #    aggregation function was specified.
    
    statistic,pvalue1=stats.shapiro(group1)    #Shapiro-Wilk test for normality
    statistic,pvalue2=stats.shapiro(group2)
    statistic,pvalue3=stats.shapiro(group3)
    
    sub_id=[i+1 for i in xrange(len(group1))]*3   #sub_id: individual trials; independent variables
    iv=N.repeat([1, 2, 3], len(group1))         #comparing group ids; dependent variables
    rt=N.concatenate((group1,group2,group3))    #dataset to compare
    
    df=pd.DataFrame({'id':sub_id, 'RT':rt, 'groups':iv})
     #df = pd.DataFrame({'patient': N.repeat([1, 2, 3, 4, 5], 4),
     #              'drug': N.tile([1, 2, 3, 4], 5),
      #             'response': [30, 28, 16, 34,
       #                         14, 18, 10, 22,
        #                        24, 20, 18, 30,
         #                       38, 34, 20, 44, 
          #                      26, 28, 14, 30]})
    
    if pvalue1>0.05 and pvalue2>0.05 and pvalue3>0.05:
        print 'Datasets normaly distributed'
        a=AnovaRM(data=df,depvar='RT', subject='id', within=['groups']).fit() 
        print a
        #with print a the result with Fvalue, NumDF (degree of freedom), DENDF, Pr>F
        
        # the F test-statistic is 24.7589 and the corresponding p-value is 0.0000.
        #Since this p-value is less than 0.05, we reject the null hypothesis and 
        #conclude that there is a statistically significant difference in mean response 
        #times between the four drugs.
        
        #    A one-way repeated measures ANOVA was conducted on 5 individuals to 
        #examine the effect that four different drugs had on response time.
        
        #Results showed that the type of drug used lead to statistically significant 
        #differences in response time (F(3, 12) = 24.75887, p < 0.001).
        
        #Details under link: https://www.statology.org/repeated-measures-anova-python/
        
        #write the ANOVA result in text file to get single values
        orig_stdout = sys.stdout
        f = open('%s_u%s_%s_%s_ANOVARM.txt' %(fig_to_save,uname,fn,script), 'w')
        sys.stdout = f
        print a
        sys.stdout = orig_stdout
        f.close() 
        
        '''
        with open('out.txt') as f:
            if 'F Value' in f.read():
                print("true")
                '''
                
        #get the values of Fvalue, NumDF, DENDF, Pr>F
        with open('%s_u%s_%s_%s_ANOVARM.txt' %(fig_to_save,uname,fn,script)) as f:
            data=f.read()
            #data=data.replace('=','')     #optional: if necessary, remove the lines and so on.
            #data=data.replace('-','')
            #data=data.replace('\n','')
            #data=data.replace('Anova','')
            #data=data.replace('','')
        num = [float(s) for s in re.findall(r'-?\d+\.?\d*', data)]  #find all digital values
        #[24.7589, 3.0, 12.0, 0.0]
        
        if len(num)==4:
            pvalue = num[-1]
            Num_df = num[1]
            DEN_df=num[2]
            F_value=num[0]
        else:
            pvalue = N.nan
            Num_df = num[0]
            DEN_df=num[1]
            F_value=N.nan
            
        if  pvalue<0.05:
            print 'Significant difference between groups: F(%s,%s):%s, p=%s' %(Num_df,DEN_df,F_value,pvalue)
        else:
            print 'No significance between groups: F(%s,%s): %s, p=%s' %(Num_df,DEN_df,F_value,pvalue)
        
        
        
        #Multicomparison with Holm-Bonferroni method
        MultiComp = MultiComparison(df['RT'],df['groups'])
        comp = MultiComp.allpairtest(stats.ttest_rel, method='Holm')
        print comp[0]
        [pvalue12,pvalue13,pvalue23]=comp[1][2]
        
        orig_stdout = sys.stdout
        f = open('%s_u%s_%s_%s_ANOVARM.txt' %(fig_to_save,uname,fn,script), 'w')
        sys.stdout = f
        print a
        print comp[0]
        sys.stdout = orig_stdout
        f.close() 
        
    else:
        #print 'Attention! Datasets not normaly distributed, Kruskal-walllis H test'
        print 'Attention! Datasets not normaly distributed, Friedman test'
        print 'group1 pvalue', pvalue1
        print 'group2 pvalue', pvalue2
        print 'group3 pvalue', pvalue3
        #statistic,pvalue=stats.kruskal(group1,group2,group3)      #kruskal-wallis H test for not normaly distributed samples
        statistic,pvalue=stats.friedmanchisquare(group1,group2,group3)   #friedman test for dependent samples, not normaly distributed
        #Multicomparison with Holm-Bonferroni method
        MultiComp = MultiComparison(df['RT'],df['groups'])
        comp = MultiComp.allpairtest(stats.wilcoxon, method='Holm') #dependent samples, not normally distributed
        print comp[0]
        [pvalue12,pvalue13,pvalue23]=comp[1][2]
        '''
        if (group1==group2)==True or (group2==group3)==True or (group1==group3)==True:
            [pvalue12,pvalue13,pvalue23]=[N.nan,N.nan,N.nan]
        else:
            MultiComp = MultiComparison(df['RT'],df['groups'])
            comp = MultiComp.allpairtest(stats.wilcoxon, method='Holm') #dependent samples, not normally distributed
            #comp = MultiComp.allpairtest(stats.mannwhitneyu, method='Holm')
            print comp[0]
            [pvalue12,pvalue13,pvalue23]=comp[1][2]
        '''
        Num_df = N.nan
        DEN_df=N.nan
        F_value=N.nan
        
        orig_stdout = sys.stdout
        f = open('%s_u%s_%s_%s_ANOVARM.txt' %(fig_to_save,uname,fn,script), 'w')
        sys.stdout = f
        #print 'Attention! Datasets not normaly distributed, Kruskal-walllis H test'
        print 'Attention! Datasets not normaly distributed, Friedman test'
        print 'group1 pvalue', pvalue1
        print 'group2 pvalue', pvalue2
        print 'group3 pvalue', pvalue3
        print 'pvalue Kruskal-wallis H', pvalue
        if (group1==group2)==True or (group2==group3)==True or (group1==group3)==True:
            print 'pvalue nan with identical number in groups'
        else:
            print comp[0]
        sys.stdout = orig_stdout
        f.close() 
        
    return pvalue, Num_df, DEN_df, F_value, pvalue12, pvalue13, pvalue23
