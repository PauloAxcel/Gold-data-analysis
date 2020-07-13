import os, glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import solveh_banded
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn import svm

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


font = {'family' : 'arial',
#        'weight' : 'bold',
        'size'   : 24}

plt.rc('font', **font)

directory = r'C:\\Users\\paulo\\Desktop\\birmingham_02\\Molecular Imprint surfaces\\report\\report for SERS SMI\\figure 3\\'

os.chdir(directory)


file_names = glob.glob('*.txt') # gets all .txt files in the folder

#    
dataset2 = []
peak_data = []

min_val = []
max_val = []

###############################################################################
######## PREPROCESSING DIFFERENT WAVENUMBER SPECTRA ###########################
############################################################################### 

for j in range(len(file_names)):
    dataset = pd.read_csv(file_names[j], sep = '\t', header = None) 
    df = dataset.reset_index(drop=True)
    n_sample = sum(df.iloc[0,0]==df.iloc[:,0])
    n_row = df.iloc[:,0].shape[0]//n_sample
    wavenumber = df.iloc[:n_row,0]
    min_val.append(min(wavenumber))
    max_val.append(max(wavenumber))

min_val = max(min_val)
max_val = min(max_val)

min_val = 550

for j in range(len(file_names)):
    dataset = pd.read_csv(file_names[j], sep = '\t', header = None)
    df = dataset.reset_index(drop=True)
    n_sample = sum(df.iloc[0,0]==df.iloc[:,0])
    n_row = df.iloc[:,0].shape[0]//n_sample
    wavenumber = df.iloc[:n_row,0]
    x = wavenumber[wavenumber>min_val][wavenumber<max_val]
    
###############################################################################
######## AVERAGE GRAPH PLOY ###################################################
############################################################################### 
    
spectra = []

colors = ['k','grey','g','lime','b','cyan']
targets = ['DSAC','DSACS','ST75','ST75S','ST95','ST95S']
k=-1
for j in range(len(file_names)):
    if j%2 == 0:
        plt.figure(figsize=(9, 9/1.618))
        k = k+1

    dataset = pd.read_csv(file_names[j], sep = '\t', header = None)
    
    df = dataset.reset_index(drop=True)


    n_sample = sum(df.iloc[0,0]==df.iloc[:,0])
    
    n_row = df.iloc[:,0].shape[0]//n_sample
    
    wavenumber = df.iloc[:n_row,0]

    dataset = []
     
    
    for i in range(df.shape[0]//n_row):
        pillar_df = df.iloc[i*n_row:n_row*(i+1),1].reset_index(drop=True)
        dataset.append(pillar_df)

    
    final_df = pd.concat([wavenumber,pd.concat(dataset,axis=1)],axis=1)
    
    #    final_df.to_csv(file_names[j]+'.csv',sep=';',index=False)
    
    final_I = final_df.iloc[:,1:]
    

    #AVERAGE    
    
    final_I_avg = final_I.mean(axis=1)
    
    y = final_I_avg[wavenumber>min_val][wavenumber<max_val]
    
    new_x = wavenumber[wavenumber>min_val][wavenumber<max_val]
    
    new_y = np.interp(x,sorted(new_x),y[::-1])
    
#    y = y[:800]
#    x = x[:800]
    
    final_I_std = final_I.std(axis=1)
    
    y_std = final_I_std[wavenumber>min_val][wavenumber<max_val]
    
#    y_std = y_std[:800]
        
    #    windows = 10
    
    #    region = wavenumber[0:2900]
    
#    for k in range(final_I.shape[1]):
#    
#        x = wavenumber[170:970]
#        y = final_I.iloc[:,k]-final_I.iloc[:,k].min()
#        y = final_I_avg[170:970]
    
    #BASELINE    
    
#    y = final_I_avg
#    x = wavenumber
    
    baseline = als_baseline(new_y)
    y = new_y-baseline
    y = (y-y.min())/(y.max()-y.min())
    
    for m in range(final_I.shape[1]):
        x2 = wavenumber
        y2 = final_I.iloc[:,m]
        yf = np.interp(x,sorted(x2),y2[::-1])
        baseline = als_baseline(yf)
        yf = yf-baseline
        yf = (yf-yf.min())/(yf.max()-yf.min())
        spectra.append(yf)
    
    #    peak.append(y[round(x)==833])
    
    #NORMALISATION
    
    
    plt.plot(x,y,label=targets[j][:20],color=colors[j],lw=2.5)
    
#    plt.fill_between(x,y,y+y_std,label=file_names[j][:15],ls='--')
    
    plt.ylabel('Intensity (a.u.)')
    plt.xlabel('Raman shift (cm$^{-1}$)')

    plt.legend(loc='best')
    plt.show()
    
###############################################################################
######## PCA AND LOADING CALCULATION ##########################################
############################################################################### 

spectra = pd.DataFrame(spectra)
labels = pd.DataFrame(['DSAC']*100+['DSACS']*100+['ST75']*300+['ST75S']*300+['ST95']*300+['ST95S']*300)
labels.columns = ['label']


colors = ['k','grey','g','lime','b','cyan']

table = []

region = [0,200,800,1400]


for n in range(len(region)-1):
    spectra2 = spectra[region[n]:region[n+1]]
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(spectra2)
    columns = ['principal component '+str(i) for i in range(1,3)]
    principalDf = pd.DataFrame(data = principalComponents , columns = columns)
    
    ########################################
    #loadings plot
    ################################
       
    plt.figure(figsize=(9,9/1.618))
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), columns = ['LPC1','LPC2'])
    loadings['LPC1'] = -loadings['LPC1']
    plt.plot(x,loadings['LPC1'],color='darkmagenta',lw=2.5)
    plt.plot(x,loadings['LPC2'],color='darkmagenta',ls='--',lw=2.5)
    plt.legend(['LPC1','LPC2'],loc='best')
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Loadings')
    plt.show()
    
    #    plt.savefig('C:\\Users\\paulo\\Desktop\\birmingham_02\\Molecular Imprint surfaces\\report\\report on flat smi\\'+targets[0]+'_loadings_v02.svg' , dpi=300, transparent=True)
    
    labels2 = labels[region[n]:region[n+1]].reset_index(drop=True)
    
    
    finalDf = pd.concat([principalDf, labels2], axis = 1)
    

    
    z = np.abs(stats.zscore(finalDf.iloc[:,:2]))
    finalDf = finalDf[(z < 3).all(axis=1)]
    finalDf.reset_index(drop=True)
    
    

    
###############################################################################
######## GAUSSIAN MIXTURE CLUSTERING ##########################################
###############################################################################    

    
    position = []
    
    fig = plt.figure(figsize=(9,9/1.618))
    ax = fig.add_subplot(1,1,1)
    plt.xlabel('PC 1 ('+str(round(pca.explained_variance_ratio_[0]*100,2))+' %)')
    plt.ylabel('PC 2 ('+str(round(pca.explained_variance_ratio_[1]*100,2))+' %)')
    #plt.title(target[k]+' before and after sugar')

    targets2 = targets[n*2:(n+1)*2]
    colors2 = colors[n*2:(n+1)*2]
    
    
    for target, color in zip(targets2,colors2):
        indicesToKeep = finalDf['label'] == target
        
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
        
    #    ax.scatter(new_x, new_y, c = color , s =100,alpha=1,label=target,marker='x')
        position.append([pos,np.sqrt(width/2),np.sqrt(height/2),target])
        
        range1s = (((new_x < pos[0]+np.sqrt(width/2)) & (new_x > pos[0]-np.sqrt(width/2))) | 
                ((new_y < pos[1]+np.sqrt(height/2)) & (new_y > pos[1]-np.sqrt(height/2))))
        ax.scatter(new_x[range1s], new_y[range1s], c = color , s =100,alpha=1,label=target,marker='x')
    
    #    plot_gmm(gmm)
    
        handles, labels2 = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels2, handles))
        plt.legend(loc='best',ncol=5)

    plt.show()
    
    for m in range(1,len(position)):
        distance = np.sqrt((position[0][0][0]-position[m][0][0])**2+ (position[0][0][1]-position[m][0][1])**2)
        
        err1 = (position[0][0][0]-position[m][0][0])/distance
        err2 = -(position[0][0][0]-position[m][0][0])/distance
        err3 = (position[0][0][1]-position[m][0][1])/distance
        err4 = -(position[0][0][1]-position[m][0][1])/distance
        
        errt = err1**2*position[0][1]**2+err2**2*position[m][1]**2+err3**2*position[0][2]+err4**2*position[m][2]
        
        table.append([distance,errt,position[0][3]+'-'+position[m][3]])
    
###############################################################################
######## BAR PLOT EUCLIDEAN CLUSTER DISTANCE ##################################
###############################################################################
        
table = pd.DataFrame(table)
#table.index=table.iloc[:,-1]
table = table.iloc[:,:2]
table.columns=['distance','err']

x2 = np.arange(0,3,1)

x = ['DSAC','ST75','ST95']
color = ['k','g','b']
sugars = ['Stachyose']


Y = table['distance']
Y_err = table['err']


#plt.rcParams['hatch.linewidth'] = 3.0 
fig = plt.figure(figsize=(9, 9/1.618))

for i in range(len(x)):
    p1 = plt.bar(x2[i] ,Y[i],yerr=Y_err[i],color='orange',label=sugars[0],align='edge', capsize=7)
    plt.legend(loc='best')   
    plt.grid(True, which='both', axis='y')
        
plt.ylabel('Cluster Distance')
plt.xticks(list(x2+0.4),x)
handles, labelss = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labelss, handles))
plt.legend(by_label.values(), by_label.keys())
plt.yticks(np.arange(0,Y.max()+Y_err.max()+1,0.5))

plt.ylim(0,round(Y.max()+Y_err.max()))
#plt.savefig('cluster difference bar plot.svg',  dpi=300, transparent=True, bbox_inches='tight')


###############################################################################
######## ROC CURVE PLOT #######################################################
###############################################################################

classes = ['DSAC','DSACS','ST75','ST75S','ST95','ST95S']

for n in range(len(region)-1):
    spectra2 = spectra[region[n]:region[n+1]]
    labels2 = labels[region[n]:region[n+1]]
    
    X = spectra2.values
    y = np.array(labels2.values).ravel()
    
    # Binarize the output
    y_n = label_binarize(y,classes=classes[n*2:(n+1)*2])
    
    y_n_exp = []
    
    for k in range(len(y_n)):
        if y_n[k]==0:
            y_n_exp.append([1,0])
        else:
            y_n_exp.append([0,1])
    
    y_n_exp = pd.DataFrame(y_n_exp).values.astype('int')
    
    n_classes = y_n_exp.shape[1]
    
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    
        # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_n_exp, test_size=.4,  random_state=0)
    
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
 
    
    plt.figure(figsize=(9, 9/1.618))
    lw = 2
    plt.plot(fpr['micro'], tpr['micro'], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc['micro'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    
#    # shuffle and split training and test sets
#    X_train, X_test, y_train, y_test = train_test_split(X, y_n_exp.values, test_size=.5,  random_state=0)
#    
#    # Learn to predict each class against the other
#    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#                                     random_state=random_state))
#
#    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
#    
#    # Compute ROC curve and ROC area for each class
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
#    for i in range(n_classes):
#        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
#        roc_auc[i] = auc(fpr[i], tpr[i])
#    
#    # Compute micro-average ROC curve and ROC area
#    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#    
#    lw = 2
#    
#    
#    # First aggregate all false positive rates
#    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#    
#    # Then interpolate all ROC curves at this points
#    mean_tpr = np.zeros_like(all_fpr)
#    for i in range(n_classes):
#        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#    
#    # Finally average it and compute AUC
#    mean_tpr /= n_classes
#    
#    fpr["macro"] = all_fpr
#    tpr["macro"] = mean_tpr
#    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#    
#    # Plot all ROC curves
#    plt.figure(figsize=(9,9/1.618))
#    plt.plot(fpr["micro"], tpr["micro"],
#             label='micro-average ROC curve (area = {0:0.2f})'
#                   ''.format(roc_auc["micro"]),
#             color='deeppink', linestyle=':', linewidth=4)
#    
#    plt.plot(fpr["macro"], tpr["macro"],
#             label='macro-average ROC curve (area = {0:0.2f})'
#                   ''.format(roc_auc["macro"]),
#             color='navy', linestyle=':', linewidth=4)
#    
#    colors = cycle(['orange','sienna','mediumpurple'])
#    for i, color in zip(range(n_classes), colors):
#        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                 label='ROC curve of class {0} (area = {1:0.2f})'
#                 ''.format(i, roc_auc[i]))
#    
#    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Some extension of Receiver operating characteristic to multi-class')
#    plt.legend(loc="lower right")
#    plt.show()


