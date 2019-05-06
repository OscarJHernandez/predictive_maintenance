import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import emcee
import scipy.optimize as op
from scipy.signal import savgol_filter # This is to smoothen the signal
from scipy.signal import argrelextrema
from scipy.signal import chirp, find_peaks, peak_widths
from sklearn.preprocessing import StandardScaler
import os

# # Read in the data, label it choose an engine number to examine


def detect_anomaly(in_data,N_clusters,eng_id,threshold,N_features,n_min=60,steps=80):
	"""
	N_features: The number of features to extract from the PCA vector
	n_min = 60 # Minimum place to start the line fit
	steps = 80 # How many steps to take in fitting the line
	"""
	
	# Some fixed parameters
	savgol_window_size = 81
	out_data = 'savgol_eng_'+str(eng_id)+"/"
	#n_min = 60 # Minimum place to start the line fit
	#threshold = 0.5 # In units of sigma
	
		
	try:
		# Create target Directory
		os.mkdir(out_data)
		#print("Directory " , out_data ,  " Created ") 
	except FileExistsError:
		print("Directory " , out_data,  " already exists")
	
	
	# Read in the data 
	data = pd.read_csv('data/'+in_data,header=None,delim_whitespace=True)
	
	# Now we label the columns
	settings = ['operational_setting_1','operational_setting_2','operational_setting_3']
	sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
	           'sensor_9','sensor_10', 'sensor_11','sensor_12', 'sensor_13','sensor_14', 'sensor_15','sensor_16',
	           'sensor_17','sensor_18', 'sensor_19','sensor_20', 'sensor_21']
	
	cols = ['engine_num','time_cycles']+settings+sensors
	data.columns = cols
	
	
	sensor_data = data.drop(settings,axis=1)
	sensor_data = sensor_data[sensor_data['engine_num']==eng_id]
	sensor_data = sensor_data[sensors]
	
	
	
	# Now we examine the correlations 
	eng1_data = sensor_data
	
	# These three sensors are flat lines
	eng1_data = eng1_data.drop(["sensor_1"], axis=1)
	eng1_data = eng1_data.drop(["sensor_18"], axis=1)
	eng1_data = eng1_data.drop(["sensor_19"], axis=1)
	
	corr = eng1_data.corr()
	corr = np.abs(corr)
	
	# plot the heatmap
	plt.clf()
	sns.heatmap(corr, 
	        xticklabels=corr.columns,
	        yticklabels=corr.columns)
	plt.title("Engine Number: " + str(eng_id))
	plt.plot()
	plt.savefig(out_data+"corr_data_full_"+in_data+"_sensor_"+str(int(eng_id))+'.pdf',bboxes='tight')
	
	
	# Now we examine the correlations 
	eng1_data = sensor_data
	
	# These three sensors are flat lines
	eng1_data = eng1_data.drop(["sensor_1"], axis=1)
	eng1_data = eng1_data.drop(["sensor_18"], axis=1)
	eng1_data = eng1_data.drop(["sensor_19"], axis=1)
	
	# Drop these correlated sensors
	eng1_data = eng1_data.drop(["sensor_5"], axis=1)
	eng1_data = eng1_data.drop(["sensor_6"], axis=1)
	eng1_data = eng1_data.drop(["sensor_10"], axis=1)
	eng1_data = eng1_data.drop(["sensor_16"], axis=1)
	corr = np.abs(eng1_data.corr())
	
	# plot the heatmap
	plt.clf()
	sns.heatmap(corr, 
	        xticklabels=corr.columns,
	        yticklabels=corr.columns)
	plt.plot()
	plt.title("Engine Number: " + str(eng_id))
	plt.savefig(out_data+"corr_data_sub_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight')
	
	#================================================================
	# Choose the N clusters
	#================================================================
	
	N_ind_sensors_label = []
	N_ind_sensors_name = [] 
	
	corr = np.abs(eng1_data.corr())
	M = np.asarray(corr.iloc[:,:])
	Z = linkage(M,'single' )
	plt.figure(figsize=(25, 10))
	labelsize=20
	ticksize=15
	plt.title('Hierarchical Clustering Dendrogram for Sensor Data', fontsize=labelsize)
	plt.xlabel('stock', fontsize=labelsize)
	plt.ylabel('distance', fontsize=labelsize)
	dendrogram(
	    Z,
	    leaf_rotation=90.,  # rotates the x axis labels
	    leaf_font_size=8.,  # font size for the x axis labels
	    labels = corr.columns
	)
	plt.yticks(fontsize=ticksize)
	plt.xticks(rotation=-90, fontsize=ticksize)
	plt.savefig(out_data+"sensor_dendrogram_sub_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight')
	
	# Lets generate three clusters based on the data
	agglo = cluster.FeatureAgglomeration(n_clusters=N_clusters)
	agglo.fit(M)
	M_reduced = agglo.transform(M)
	
	cluster_label = agglo.labels_
	data_col = corr.columns
	
	
	# Now we find representatives of the N clusters
	
	# Initialize our array
	N_ind_sensors_label.append(cluster_label[0])
	N_ind_sensors_name.append(data_col[0])
	
	for k in range(1,len(cluster_label)):
	    
	    if(cluster_label[k] not in N_ind_sensors_label):
	        N_ind_sensors_label.append(cluster_label[k])
	        N_ind_sensors_name.append(data_col[k])
	        
	
	
	# Now we examine the correlations 
	eng1_data_ind = sensor_data[N_ind_sensors_name]
	corr = np.abs(eng1_data_ind.corr())
	
	
	# plot the heatmap
	plt.clf()
	sns.heatmap(corr, 
	        xticklabels=corr.columns,
	        yticklabels=corr.columns)
	plt.title("Engine Number: " + str(eng_id))
	plt.plot()
	plt.savefig(out_data+"_corr_data_N_sensors_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight')
	
	
	def rms(y):
	    
	    s = np.dot(y,y)
	    s = s/float(len(y))
	    s =np.sqrt(s)
	    
	    return s
	
	def max_peak(y):
	    
	    s = np.max(y)
	    
	    return s
	
	def line_int(y):
	    
	    s = 0.0
	    
	    for i in range(1,len(y)):
	        s+= np.abs(y[i]-y[i-1])
	    
	    return s
	
	def energy(y):
	    
	    y = y -np.mean(y)
	    s = np.dot(y,y)
	    
	    return s
	
	def std(y):
	    
	    s = np.std(y)
	    
	    return s
	
	def compute_property(func,y_vec):
	    
	    N = len(y_vec)
	    y_func = np.zeros(N)
	    
	    for i in range(1,N+1):
	        yi = y_vec[0:i]
	        fi = func(yi)
	        y_func[i-1] = fi
	    
	    
	    
	    return y_func
	    
	
	
	
	# # Next, we compute all features for all of the sensors and combine all of the data into a large feature matrix
	# Generate a new data frame with all of these new features
	X =  pd.DataFrame()
	
	energy_set = ['energy_'+str(int(k)) for k in range(0,N_clusters)]
	rms_set = ['rms_'+str(int(k)) for k in range(0,N_clusters)]
	line_set = ['line_'+str(int(k)) for k in range(0,N_clusters)]
	max_set = ['max_'+str(int(k)) for k in range(0,N_clusters)]
	std_set = ['std_'+str(int(k)) for k in range(0,N_clusters)]
	
	for k in range(len(energy_set)):
	    feature_name_1 = energy_set[k]
	    feature_name_2 = rms_set[k]
	    feature_name_3 = line_set[k]
	    feature_name_4 = max_set[k]
	    feature_name_5 = std_set[k]
	    X[feature_name_1] = compute_property(energy, eng1_data_ind.iloc[0:,k].values)
	    X[feature_name_2] = compute_property(rms, eng1_data_ind.iloc[0:,k].values)
	    X[feature_name_3] = compute_property(line_int, eng1_data_ind.iloc[0:,k].values)
	    X[feature_name_4] = compute_property(max_peak, eng1_data_ind.iloc[0:,k].values)
	    X[feature_name_5] = compute_property(std, eng1_data_ind.iloc[0:,k].values)
	
	
	all_features = energy_set+rms_set+line_set+max_set+std_set
	
	
	# # The feature matrix has high dimensionality, use PCA to find the first two Priciple components of the feature matrix
	
	# In[9]:
	
    #====================================================================================
    # PCA Analysis
    #====================================================================================	
	pca = PCA(n_components=2)
	
	# Scale all of the features 
	X = X.loc[:, all_features].values
	X = StandardScaler().fit_transform(X)
	
	principalComponents = pca.fit_transform(X)
	
	principalDf = pd.DataFrame(data = principalComponents
	             , columns = ['pc1', 'pc2'])
	
	print('Explained Variance: ', pca.explained_variance_ratio_)
	
	V1 = principalDf['pc1']
	V2 = principalDf['pc2']
	
	plt.clf()
	plt.title("PCA 1")
	plt.plot(V1)
	plt.xlabel("Cycles", size=20)
	plt.ylabel("PCA [unitless]", size=20)
	plt.savefig(out_data+"PCA1_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight')
	
	
    #====================================================================================
    # Bayesian Fit
    #====================================================================================
	
	#savgol_filter(y, 11, 3) # window size 51, polynomial order 3
	x = range(len(V1))
	y1 = savgol_filter(V1,savgol_window_size,3)
	y2 = savgol_filter(V2,savgol_window_size,3)
	
	plt.clf()
	plt.title("PCA 1 Savgol Filter")
	plt.plot(x,y1)
	plt.xlabel("Cycles", size=20)
	plt.ylabel("PCA [unitless]", size=20)
	plt.savefig(out_data+"PCA1_filter_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight')
	
	def bayesian_fit(x,y): 
	
	    def lnlike(theta, x, y):
	        a1, b,sigma = theta
	        model = a1*x + b
	        inv_sigma2 = 1.0/sigma**2
	        return -0.5*(np.sum((y-model)**2*inv_sigma2- np.log(inv_sigma2)))
	
	    def lnprob(theta, x, y):
	        #lp = lnprior(theta)
	
	        #return lp + lnlike(theta, x, y, yerr)
	        return lnlike(theta, x, y)
	
	
	    nll = lambda *args: -lnlike(*args)
	    result = op.minimize(nll, [1.0,1.0,1.0], args=(x, y))
	    a1_ml, b_ml, sigma_ml = result["x"]
	
	    ndim, nwalkers = 3, 8
	    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
	
	    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y))
	
	    sampler.run_mcmc(pos, 100)
	
	    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
	    
	    return samples
	
	
	
	Nk = []
	a1_k = []
	a2_k = []
	a2_error_k = []
	a1_error_k = []
	sse_k = []
	

	n_max = len(y1)
	

	dh = int(float(n_max-n_min)/float(steps))
	
	
	
	for N in range(n_min,n_max,dh):
	    
	    x_N = x[0:N]
	    y_N = y1[0:N]
	    
	    samples =  bayesian_fit(x_N,y_N)
	    
	    
	    a1_N = np.mean(samples[:,0])
	    b_N = np.mean(samples[:,1])
	    
	    a1_k.append(a1_N)
	    
	    model_k = b_N + a1_N*np.asarray(x_N)
	    sse = np.dot(y_N-model_k,y_N-model_k)
	    
	    a1_error_k = np.std(samples[:,0])
	    sse_k.append(sse)
	    Nk.append(N)
	
	plt.clf()
	plt.title('Linear coefficient')
	plt.plot(Nk,a1_k)
	plt.ylabel('$a_1$',size=20)
	plt.xlabel('Cycles',size=20)
	plt.savefig(out_data+"linear_coeff_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight')    
	
	
	plt.clf()
	plt.plot(Nk,sse_k,'-o',color='blue')
	plt.xlabel("Cycles", size=20)
	plt.ylabel('Residuals',size=20)
	plt.savefig(out_data+"residuals_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight')    
	
	plt.clf()
	sse2_k = np.gradient(sse_k,2)
	plt.title("Residual Acc. vs Cycle",size=20)
	plt.plot(Nk,sse2_k,'-o',color='green')
	plt.xlabel("Cycles", size=20)
	plt.ylabel("Residual Acceleration", size=20)
	plt.savefig(out_data+"residual_acc_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight')    
	
	
	
	#============================================================================================
	# Now we determine the range of the anomalies	
	# Apply the standard Scaler to the SSE-acceleration data
	sse2_k_median = np.median(sse2_k)
	sse2_k_std = np.std(sse2_k)
	sse2_k_scaled = np.abs(sse2_k-sse2_k_median)/sse2_k_std
	
	# Now normalize from zero to one
	sse2_k_max = np.max(sse2_k_scaled)
	sse2_k_min = np.min(sse2_k_scaled)
	sse2_k_scaled = (sse2_k_scaled-sse2_k_min)/(sse2_k_max-sse2_k_min)
	
	# Define a possible threshold for the failure region
	Nk_anom =[]
	
	for i in range(len(sse2_k)):
	    
	    # Find all points that have an anomaly
	    if(sse2_k_scaled[i] >= threshold):
	        Nk_anom.append(Nk[i])
	
	if(len(Nk_anom)!=0):
		plt.clf()
		[plt.axvline(Ni,alpha=1.0,color='red',linewidth=2.0) for Ni in Nk_anom]
		plt.axvspan(Nk_anom[0],Nk_anom[-1],alpha=0.2, color='purple')
		plt.plot(Nk,sse2_k_scaled,'-o',color='green')
		plt.title("Residual Acc. Scaled vs Cycle",size=20)
		plt.xlabel("Cycles", size=20)
		plt.ylabel("Residual Acceleration", size=20)
		plt.savefig(out_data+"_scaled_residual_acc_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight') 
	    
	
	
	for k in range(0,len(N_ind_sensors_name)):
		plt.clf()
		plt.title(N_ind_sensors_name[k],size='20')
		plt.plot(eng1_data_ind.iloc[:,k].values)
		
		if(len(Nk_anom)!=0):
			[plt.axvline(Ni,alpha=0.5,color='red',linewidth=3.0) for Ni in Nk_anom]
		#	plt.axvspan(Nk_anom[0],Nk_anom[-1],alpha=0.2, color='purple')
		
		plt.xlabel("Cycles",size=20)
		plt.legend()
		plt.savefig(out_data+"sensor_"+str(N_ind_sensors_name[k])+"_anomaly_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight')
	    
	    
	plt.clf()
	plt.plot(x,y1)
	if(len(Nk_anom)!=0):
		[plt.axvline(Ni,alpha=0.3,color='red',linewidth=3.0) for Ni in Nk_anom]
		#plt.axvspan(Nk_anom[0],Nk_anom[-1],alpha=0.2, color='purple')
	plt.ylabel("PCA1 filter",size=20)
	plt.xlabel("Cycles",size=20)
	plt.savefig(out_data+"PCA_"+in_data+"_eng_"+str(int(eng_id))+'.pdf',bboxes='tight')
	
	y_feature = x[-1]-x[N_features]
	x_feature = y1[0:N_features]
	return x_feature,y_feature
