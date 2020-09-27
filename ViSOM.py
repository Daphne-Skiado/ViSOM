import os
import numpy as np
from PIL import Image
import math
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.cluster import KMeans
from matplotlib.gridspec import GridSpec
import matplotlib._color_data as mcd


def normalize_data(x):
	norm_data=np.copy(x)
	scaler = MinMaxScaler()
	norm_data = scaler.fit_transform(norm_data)

	return norm_data

def create_image_embedding(image):
	emb = []
	for color in range(3):
		image_color = np.copy(image[:, :, color])
		color_emb = [0]*11
		for i in range(11):
			items_on_the_bin = np.where((image_color // 25) == i)[0].size
			color_emb[i] = items_on_the_bin
		color_emb[9] += color_emb[10]
		emb = emb + color_emb[:10]
	return emb

def load_dataset(path, images_files):
	images_embeddings = np.zeros((len(images_files),30))
	emb_index = 0
	for image_name in images_files:
		img = Image.open(path+image_name)
		img_array = np.array(img)
		embedding = create_image_embedding(img_array)
		images_embeddings[emb_index] = np.array(embedding)
		emb_index += 1
	return images_embeddings

################################################
############Performance Evaluation##############
################################################

def find_pairwise_TP(cluster_labels, true_labels):
	TP = 0
	for i in range(len(cluster_labels)-1):
		for j in range(i+1,len(cluster_labels)):
			if (cluster_labels[i] == cluster_labels[j]) and (true_labels[i] == true_labels[j]):
				TP += 1
	return TP

def find_pairwise_FP(cluster_labels, true_labels):
	FP = 0
	for i in range(len(cluster_labels)-1):
		for j in range(i+1,len(cluster_labels)):
			if (cluster_labels[i] == cluster_labels[j]) and (true_labels[i] != true_labels[j]):
				FP += 1
	return FP

def find_pairwise_TN(cluster_labels, true_labels):
	TN = 0
	for i in range(len(cluster_labels)-1):
		for j in range(i+1,len(cluster_labels)):
			if (cluster_labels[i] != cluster_labels[j]) and (true_labels[i] != true_labels[j]):
				TN += 1
	return TN

def find_pairwise_FN(cluster_labels, true_labels):
	FN = 0
	for i in range(len(cluster_labels)-1):
		for j in range(i+1,len(cluster_labels)):
			if (cluster_labels[i] != cluster_labels[j]) and (true_labels[i] == true_labels[j]):
				FN += 1
	return FN

def evaluate_performance(cluster_labels, true_labels):
	TP = find_pairwise_TP(cluster_labels, true_labels)
	FP = find_pairwise_FP(cluster_labels, true_labels)
	TN = find_pairwise_TN(cluster_labels, true_labels)
	FN = find_pairwise_FN(cluster_labels, true_labels)
	if TP == FP == 0:
		precision = 0
	else:
		precision = TP / (TP + FP)
	if TP == FN == 0:
		recall = 0
	else:
		recall = TP / (TP + FN)
	if precision == recall == 0:
		f1_measure = 0
	else:
		f1_measure = 2 * (precision * recall) / (precision + recall)
	if (TP == TN == FP == FN == 0):
		accuracy = 0
	else:
		accuracy = (TP + TN) / (TP + TN + FN + FP)

	print('F1 measure : ' + str(f1_measure))
	print('Accuracy : ' + str(accuracy))
	return precision, recall, f1_measure, accuracy

#Function to create clustering plots
def plot_map(M, clustering_labels, true_labels, category_imgs):
	c_l = np.array(clustering_labels)
	t_l = np.array(true_labels)
	index = 0
	label_names = np.unique(t_l)

	all_samples_in_same_neuron = [0,0,0,0,0,0]
	labels_max = {}
	for i in range(M):
		for j in range(M):
			k = (i,j)
			samples = np.where(c_l == index)[0].tolist()
			samples_labels = t_l[samples]
			neurons_labels = [0 for n in range (np.unique(t_l).size)]
			max_label_index = -1
			
			for s in samples_labels :
				neurons_labels[s] += 1
			temp = neurons_labels[-1]
			if (temp == 1):
				neurons_labels[-1] = 5
			for n in range(len(neurons_labels)):
				all_samples_in_same_neuron[neurons_labels[n]] += 1
				if neurons_labels[n] < 3:
					neurons_labels[n] = 0
				else:
					max_label_index = n
			neurons_labels[-1] = temp	
			labels_max[k] = max_label_index
			index += 1
	print('Categories with all their samples assigned to the same neuron : ' + str(all_samples_in_same_neuron[5]))
	print('Categories with 4 of their samples assigned to the same neuron : ' + str(all_samples_in_same_neuron[4]))
	print('Categories with 3 of their samples assigned to the same neuron : ' + str(all_samples_in_same_neuron[3]))

	rows = M
	columns = M
	f, axarr = plt.subplots(rows, columns, figsize=(M+4,M+4))
	f.tight_layout(pad=0.5)
	for axi in axarr.ravel():
		axi.set_axis_off()
	index = 0
	for r in range(rows):
		for c in range(columns):
			neuron_max = labels_max[(r,c)]
			if neuron_max == -1:
				continue
			path = category_imgs[neuron_max]
			image = Image.open(path)
			axarr[r,c].imshow(image)
			axarr[r,c].set_title(str((r,c)))
			
	f.savefig('ViSOM_map_'+str(M)+'.png')

################################################
#############ViSOM Implementation###############
################################################

#Weights are initialized to random vectors 1x30 with values from [0.0, 0.1]
def init_random_weights(grid_x, grid_y, data):
	weights = np.random.rand(grid_x*grid_y, data.shape[1]) / 10.0

	return weights 

def sigma_decay(sigma_0, i, sigma_exp_value):
	new_sigma = sigma_0 * np.exp(-i / sigma_exp_value)
	return new_sigma

def alpha_decay(alpha_0, i, num_of_iters):
	new_alpha = alpha_0 * np.exp(-i / num_of_iters)
	return new_alpha

#Return index of weight that has the smaller distance from the chosen sample
def find_winner(sample, weights):
	total_dists = np.linalg.norm(np.subtract(sample,weights),axis=-1)
	winner = np.argmin(total_dists)
	return winner

#For a fiven neuron find its neighbours on the grid
#Compute the neighbourhood function (Gaussian) for each neighbour
#Return dicitonary with neighbour's indices as keys and their neighbourhood function values as values
def find_neighbours(winners_coords, sigma, grid):
	dists = np.linalg.norm(np.subtract(winners_coords, grid), axis = 1)
	winner_dist_index = grid.index(winners_coords)
	dists[winner_dist_index] = np.Inf
	#Compute Gaussian neighbourhood function values
	h_func_values = np.exp(-np.divide(np.power(dists,2), 2*(sigma**2)))
	#Find neighbours (using sigma as the neighbourhood radius)
	neighs_indices = np.where(np.power(dists,2) <= sigma)[0].tolist()
	neighbours = {i : h_func_values[i] for i in neighs_indices}

	return neighbours

#Update the winner's weight to pull it closer to the chosen sample
def update_winner(sample, weights, winners_coords, alpha):
	winner_weight = weights[winners_coords]
	diverse = np.subtract(sample,winner_weight)
	new_weight = winner_weight + alpha*diverse

	return new_weight

#Updates the weights of the neurons in the neighbourhood of the winner
def update_neighbours(neighbourhood, grid, weights, alpha, winner_index, sample, lamda):
	winner_coords = grid[winner_index]
	winner_weight = weights[winner_index]
	winner_sample_dist = np.subtract(sample, winner_weight)
	new_weights = np.copy(weights)
	neighs = list(neighbourhood.keys())
	for n in neighs:
		h_value = neighbourhood[n] #neighbourhood function value
		coords = grid[n]           #coordinates on the grid
		n_weight = weights[n]      #neighbour's weight
		n_winner_dist = np.subtract(winner_weight, n_weight)

		d_vk = np.linalg.norm(n_winner_dist)
		D_vk = np.linalg.norm(np.array(winner_coords)-np.array(coords))
		beta = (d_vk / ((D_vk*lamda)-1))

		n_new_weight = n_weight + alpha*h_value*(winner_sample_dist + beta*n_winner_dist)
		new_weights[n] = n_new_weight

	return new_weights


def ViSOM_alg(grid_x, grid_y, data, true_labels, category_imgs):
	weights = init_random_weights(grid_x, grid_y, data)

	iters = 4000
	alpha_0 = 0.8
	sigma_0 = grid_x / 2.0
	sigma_exp_value = iters / np.log(sigma_0)
	labels = [-1 for i in range(data.shape[0])]
	grid = [(i,j) for i in range(grid_x) for j in range(grid_y)]

	lamda = 4.0
	
	for i in range(iters):

		alpha = alpha_decay(alpha_0, i, iters)
		sigma = sigma_decay(sigma_0, i, sigma_exp_value)

		sample_index = random.randint(0,data.shape[0]-1)
		sample = data[sample_index]

		winner_index = find_winner(sample, weights)
		winners_coords = grid[winner_index]
		labels[sample_index] = winner_index

		old_weights = np.copy(weights)
		neighbourhood = find_neighbours(winners_coords, sigma, grid)

		new_winner_weight = update_winner(sample, old_weights, winner_index, alpha)
		new_neighbours_weights = update_neighbours(neighbourhood, grid, old_weights, alpha, winner_index, sample, lamda)

		weights = np.copy(new_neighbours_weights)

		weights[winner_index] = np.copy(new_winner_weight)

		if np.sum(euclidean_distances(old_weights, weights)) < 0.01:
			print('Convergence at iteration : ' + str(i))

	
	print('Number of created clusters : ' + str(np.unique(np.array(labels)).size))
	evaluate_performance(labels, true_labels)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	plot_map(grid_x, labels, true_labels, category_imgs)


#################################################
#################Load Dataset####################
#################################################

path = './images/'
images_files = sorted([f for f in os.listdir(path) if f.endswith('.JPG')])

images_labels = [(i // 5) for i in range(265)]

#lists used to plot clustering map results
labels_names = images_files.copy()
del labels_names[215]
category_imgs = [path+labels_names[i] for i in range(0,265,5)]
category_imgs.append(path+'image215.JPG')

#data and labels arrays creation
images_labels.insert(215,53)
images_embeddings = load_dataset(path, images_files)
images_embeddings = normalize_data(images_embeddings)
images_labels_arr = np.array([images_labels]).reshape(266,1)

labels = images_labels
data = images_embeddings


ViSOM_alg(8, 8, data, labels, category_imgs)

ViSOM_alg(16, 16, data, labels, category_imgs)
