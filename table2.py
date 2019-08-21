#!/usr/bin/env python3

'''
Author: Daniel M. Low
license: Apache 2.0

'''

import pandas as pd
import numpy as np

'''
descriptive stats for each disorder
it will consider rows that start with the disorder name (i.e., not reviews that have the disorder name second)
'''

if __name__ == "__main__":
    input_dir = 'data/inputs/'
    output_dir = 'data/outputs/'
    input_file = 'speech_psychiatry.xlsx'
    path = input_dir+input_file
    df = pd.read_excel(path, sheet_name='main_table',header=2)
    # For each disorder, otain stats
    disorders = ['depression', 'PTSD', 'schizophrenia', 'anxiety', 'bipolar', 'bulimia', 'anorexia', 'OCD']
    totals = []
    stats = []
    for disorder in disorders:
        df_disorder = df[df['Disorder and search notes'].str.startswith(disorder).fillna(False)] #this leaves out rows such as reviews
        df_disorder = df_disorder [~df_disorder ['Disorder and search notes'].str.contains('allintitle').fillna(False)]  # this leaves out subtitles
        total = df_disorder.shape[0]
        sample_median = np.nanmedian(df_disorder['Total sample size']) #this leaves out rows such as reviews
        sample_min = np.nanmin(df_disorder['Total sample size'])
        sample_max = np.nanmax(df_disorder['Total sample size'])
        clinical = np.nansum(df_disorder['clinical (1) or self-report (0)'])
        clinical_perc = np.nansum(df_disorder['clinical (1) or self-report (0)'])/total
        predictive= total-np.nansum(df_disorder['validation (leave-out development set, k-fold CV, leave-one-out CV, bootstrapping) or leave-out test set']=='null-hypothesis testing')
        predictive_perc = predictive / total
        totals.append(total)
        stats.append([
            str(int(sample_median)) + ' ('+str(int(sample_min))+'-'+str(int(sample_max))+')',
            str(int((clinical_perc)*100)) +' ('+str(int(clinical))+')',
            str(int((predictive_perc) * 100)) + ' (' + str(int(predictive)) + ')',
        ])

    total_articles = np.sum(totals)
    totals_perc = [str(np.round(n/total_articles*100,1))+' ('+str(int(n))+')' for n in totals]
    # Create table
    index = ['Depression', 'PTSD', 'Schizophrenia', 'Anxiety', 'Bipolar', 'Bulimia', 'Anorexia', 'OCD']
    table = pd.DataFrame(stats, index = index, columns = ['Median sample size (range)', 'Clinical assessment % (N)', 'Predictive models % (N)'])
    table_totals = pd.DataFrame(totals_perc, index = index, columns=['Articles % (N)'])
    table2 = pd.concat([table_totals,table], axis=1)
    table2.index.name = 'Disorder'
    table2.to_csv(output_dir + 'table2.csv')

    original = pd.read_csv(output_dir+'table2_original.csv')
    print(table2)
    print('====================')
    print(original)
    