# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:09:23 2019

@author: Paulo
"""

import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import scipy.interpolate

from matplotlib import patches

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

from matplotlib.patches import Ellipse

## script to save multiple txt files from Renishaws maps to excel file

import os, glob
import matplotlib.pyplot as plt
import pandas as pd
import datetime


import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy import optimize
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker
import pandas as pd
from scipy.linalg import solveh_banded
from scipy import signal


# From MatrixExp
def matrix_exp_eigen(U, s, t, x):
    exp_diag = np.diag(np.exp(s * t), 0)
    return U.dot(exp_diag.dot(U.transpose().dot(x)))

# From LineLaplacianBuilder
def get_line_laplacian_eigen(n):
    assert n > 1
    eigen_vectors = np.zeros([n, n])
    eigen_values = np.zeros([n])

    for j in range(1, n + 1):
        theta = np.pi * (j - 1) / (2 * n)
        sin = np.sin(theta)
        eigen_values[j - 1] = 4 * sin * sin
        if j == 0:
            sqrt = 1 / np.sqrt(n)
            for i in range(1, n + 1):
                eigen_vectors[i - 1, j - 1] = sqrt
        else:
            for i in range(1, n + 1):
                theta = (np.pi * (i - 0.5) * (j - 1)) / n
                math_sqrt = np.sqrt(2.0 / n)
                eigen_vectors[i - 1, j - 1] = math_sqrt * np.cos(theta)
    return eigen_vectors, eigen_values

def smooth2(t, signal):
    dim = signal.shape[0]
    U, s = get_line_laplacian_eigen(dim)
    return matrix_exp_eigen(U, -s, t, signal)


def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e6,
                 max_iters=10, conv_thresh=1e-5, verbose=False):
  '''Computes the asymmetric least squares baseline.
  * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
  smoothness_param: Relative importance of smoothness of the predicted response.
  asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                       Setting p=1 is effectively a hinge loss.
  '''
  smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
  # Rename p for concision.
  p = asymmetry_param
  # Initialize weights.
  w = np.ones(intensities.shape[0])
  for i in range(max_iters):
    z = smoother.smooth(w)
    mask = intensities > z
    new_w = p*mask + (1-p)*(~mask)
    conv = np.linalg.norm(new_w - w)
    if verbose:
      print (i+1, conv)
    if conv < conv_thresh:
      break
    w = new_w
  else:
    print ('ALS did not converge in %d iterations' % max_iters)
  return z


class WhittakerSmoother(object):
  def __init__(self, signal, smoothness_param, deriv_order=1):
    self.y = signal
    assert deriv_order > 0, 'deriv_order must be an int > 0'
    # Compute the fixed derivative of identity (D).
    d = np.zeros(deriv_order*2 + 1, dtype=int)
    d[deriv_order] = 1
    d = np.diff(d, n=deriv_order)
    n = self.y.shape[0]
    k = len(d)
    s = float(smoothness_param)

    # Here be dragons: essentially we're faking a big banded matrix D,
    # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
    diag_sums = np.vstack([
        np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant')
        for i in range(1, k+1)])
    upper_bands = np.tile(diag_sums[:,-1:], n)
    upper_bands[:,:k] = diag_sums
    for i,ds in enumerate(diag_sums):
      upper_bands[i,-i-1:] = ds[::-1][:i+1]
    self.upper_bands = upper_bands

  def smooth(self, w):
    foo = self.upper_bands.copy()
    foo[-1] += w  # last row is the diagonal
    return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)



font = {'family' : 'Arial',
        'size'   : 22}

plt.rc('font', **font)
##############################################################################
# PLOT ANALYZER ##############################################################
##############################################################################

files = ['gold_before_and_after_sugars.txt',
         'ST75_before_and_after_sugars.txt',
         'ST95_before_and_after_sugars.txt',
         'DSA_before_and_after_sugars.txt',
         'DSAC_before_and_after_sugars.txt']


sugars = ['Melezitose','Stachyose','Raffinose','All sugars']

table = []
cluster_dif = []
#melezitose = []
dist_matrix = []

cutoff = 840

df = pd.read_csv(r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\Molecular Imprint surfaces\molecular imprinting\19_07_2019\SiAuDSAComp\SiAuDSAComp_512map_static950_1sec_10%_1acc_785nm_pinhole_0111.txt', header = None, sep = '\t')
center = df.iloc[:1015,2]

for k in range(len(files)):
    
    target = ['Gold Substrate',
          'ST75 Substrate',
          'ST95 Substrate',
          'DSA Substrate',
          'DSAC Substrate']

    color = ['r','g','b','k','k']


    df0 = pd.read_csv(files[k], header=None,sep='\t')
    
    df = df0.iloc[:,:cutoff]
    
    targets = [target[k],'Melezitose','Stachyose','Raffinose','All sugars']
    
    colors = [color[k],'orange','sienna','mediumpurple','deepskyblue']

##############################################################################
# PCA ANALYZER ###############################################################
##############################################################################

    b = []
    j = 1
    
    for i in range(df.shape[0]):
        if i/512 < j:
            b.append(targets[j-1])
        else:
            b.append(targets[j])
            j = j +1
        
    
    df['target'] = pd.DataFrame(b)
    
    y = df['target']
    
    X = df.iloc[:,:df.shape[1]-1]
    
    rel_x = []
    
    for i in range(X.shape[0]):
        rel_x.append((X.iloc[i,:]-X.iloc[i,:].min())/(X.iloc[i,:].max()-X.iloc[i,:].min()))
        
    
    rel_x = pd.DataFrame(rel_x)
    
    plt.figure(figsize=(9,9/1.618))
    for m in range(5):

        final_I_avg = X[m*512:(m+1)*512].mean(axis=0).reset_index(drop=True)
        baseline = als_baseline(final_I_avg)
        intensity = final_I_avg-baseline
        intensity = (intensity-intensity.min())/(intensity.max()-intensity.min())


        
        plt.plot(center[:cutoff],intensity,label=targets[m], color=colors[m],lw=2)
    plt.ylabel('Intensity (a.u.)')
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.legend(loc='best',frameon=False)
    plt.show()

    plt.savefig('plot '+target[k]+'.svg' , dpi=300, transparent=True)
    
    
    x = StandardScaler().fit_transform(rel_x.T)
    
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(rel_x)
    columns = ['principal component '+str(i) for i in range(1,3)]
    principalDf = pd.DataFrame(data = principalComponents , columns = columns)
    
    ########################################
    #loadings plot
    ################################
    
    plt.figure(figsize=(9,9/1.618))
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), columns = ['LPC1','LPC2'])
    loadings['LPC1'] = -loadings['LPC1']
    plt.plot(center[:cutoff],loadings['LPC1'],color=color[k])
    plt.plot(center[:cutoff],loadings['LPC2'],color=color[k],ls='--')
    plt.legend(['LPC1','LPC2'],loc='best',frameon=False,ncol=5)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Loadings')
    plt.show()
    
    
    
    
    finalDf = pd.concat([principalDf, df['target']], axis = 1)
    
    from scipy import stats
    
    z = np.abs(stats.zscore(finalDf.iloc[:,:2]))
    finalDf = finalDf[(z < 1).all(axis=1)]
    finalDf.reset_index(drop=True)
    
    
    from collections import OrderedDict
    from sklearn.mixture import GaussianMixture
    
    position = []
    
    fig = plt.figure(figsize=(9,9/1.618))
    ax = fig.add_subplot(1,1,1)
    plt.xlabel('PC 1 ('+str(round(pca.explained_variance_ratio_[0]*100,2))+' %)')
    plt.ylabel('PC 2 ('+str(round(pca.explained_variance_ratio_[1]*100,2))+' %)')
    plt.title(target[k]+' before and after sugar')
    plt.xlim(-6,6,1)
    plt.ylim(-6,6,1)
    
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        
        gmm = GaussianMixture(n_components=1).fit(pd.concat([finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']],axis=1).values)
    
       
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            
            if covar.shape == (2, 2):
                U, s, Vt = np.linalg.svd(covar)
                angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
                width, height = 2 * np.sqrt(s)
            else:
                angle = 0
                width, height = 2 * np.sqrt(covar)
            
    #draw the 2sigma region
        ax.add_patch(Ellipse(pos,2*width,2*height,angle,alpha=0.3,color=color))
        
        new_x = finalDf.loc[indicesToKeep, 'principal component 1']
        new_y = finalDf.loc[indicesToKeep, 'principal component 2']
        
        position.append([pos,np.sqrt(width/2),np.sqrt(height/2),target])
        
        range1s = (((new_x < pos[0]+np.sqrt(width/2)) & (new_x > pos[0]-np.sqrt(width/2))) | 
                ((new_y < pos[1]+np.sqrt(height/2)) & (new_y > pos[1]-np.sqrt(height/2))))
        ax.scatter(new_x[range1s], new_y[range1s], c = color , s =100,alpha=1,label=target,marker='x')
    

    
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(loc='best',ncol=5,frameon=False)
    
    plt.show()
    plt.savefig(targets[0]+'_PCA.svg' , dpi=300, transparent=True)
    
    for m in range(1,len(position)):
        distance = np.sqrt((position[0][0][0]-position[m][0][0])**2+ (position[0][0][1]-position[m][0][1])**2)
        
        err1 = (position[0][0][0]-position[m][0][0])/distance
        err2 = -(position[0][0][0]-position[m][0][0])/distance
        err3 = (position[0][0][1]-position[m][0][1])/distance
        err4 = -(position[0][0][1]-position[m][0][1])/distance
        
        errt = err1**2*position[0][1]**2+err2**2*position[m][1]**2+err3**2*position[0][2]+err4**2*position[m][2]
        
        table.append([position[0][3] + ' - ' + position[m][3],distance,errt])
    

    target = ['Gold Substrate','ST75 Substrate','ST95 Substrate','DSA Substrate','DSAC Substrate']
    indicesToKeep = finalDf['target'] == target[k]
    df_substrate = finalDf[indicesToKeep]
    df_sugars = finalDf[[not elem for elem in indicesToKeep]]
    
    dist_substrate = []
    for i in range(df_substrate.shape[0]):
        dist_substrate.append([np.sqrt(df_substrate.iloc[i,0]**2+df_substrate.iloc[i,1]**2),df_substrate.iloc[i,2]])
    dist_substrate = pd.DataFrame(dist_substrate)
    
    
    dist_sugars = []
    for i in range(df_sugars.shape[0]):
        dist_sugars.append([np.sqrt(df_sugars.iloc[i,0]**2+df_sugars.iloc[i,1]**2),df_sugars.iloc[i,2]])
    dist_sugars = pd.DataFrame(dist_sugars)
      

    
    for j in range(dist_substrate.shape[0]):
        dist_matrix.append([dist_substrate.iloc[j,0],dist_substrate.iloc[j,1]])
            
    for i in range(dist_sugars.shape[0]):
        if  (dist_sugars.iloc[i,1] == 'Melezitose'):
            dist_matrix.append([dist_sugars.iloc[i,0],dist_sugars.iloc[i,1] + ' ' + dist_substrate.iloc[j,1]])

############################################################
        #PLOT THE DISTANCE BETWEEN CLUSTERS
        ###################################################
                
from scipy.stats import norm 

from scipy.stats import chisquare

     
dfmelezitose = pd.DataFrame(dist_matrix,columns = ['Distance','Substrate'])


from statannot import add_stat_annotation

from scipy.stats import f_oneway

box_pairs = [('Gold Substrate', 'Melezitose Gold Substrate'),
             ('ST75 Substrate', 'Melezitose ST75 Substrate'),
             ('ST95 Substrate', 'Melezitose ST95 Substrate')
             ]
y = 'Distance'
x = 'Substrate'

index = dfmelezitose.iloc[:,1] == 'DSA Substrate'
pos = np.argmax(index)
df_plot = dfmelezitose.iloc[:pos]

test_short_name = 'f_oneway'
pvalues = []
for pair in box_pairs:
    data1 = df_plot.groupby(x)[y].get_group(pair[0])
    data2 = df_plot.groupby(x)[y].get_group(pair[1])
    stat, p = f_oneway(data1, data2)
    print("Performing Bartlett statistical test for equal variances on pair:",
          pair, "stat={:.2e} p-value={:.2e}".format(stat, p))
    pvalues.append(p)
print("pvalues:", pvalues)
plt.figure(figsize=(9,9/1.618))
#ax = sns.boxplot(x="Substrate", y="Distance", data=df_plot,showfliers=False)
ax = sns.barplot(x="Substrate", y="Distance", data=df_plot,capsize=.2)

test_results = add_stat_annotation(ax, data=df_plot, x=x, y=y,
                                   box_pairs=box_pairs,
                                   perform_stat_test=False, pvalues=pvalues, test_short_name=test_short_name,
                                   text_format='star', verbose=2)
plt.xticks(fontsize=14)
plt.savefig('bar plot anova.svg' , dpi=300, transparent=True)
order = dfmelezitose.iloc[:,1].unique()






























