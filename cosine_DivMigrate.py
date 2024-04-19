import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import sys
import time
import imageio
import os


# Check if the correct number of command line arguments is provided
if len(sys.argv) != 3:
    print("Usage: python script.py <method> <simulation>")
    print("Method writing: nm, g, d")
    sys.exit(1)

method = sys.argv[1]
sim = int(sys.argv[2])

matrix1=[]
matrix2=[]

for i in range(1,sim+1):
	# input the sim data
	infile = f"running_sim{i}_{method}"
	data = np.array(pd.read_table(infile))
	listdata = []
	for j in range(len(data)):
		for l in range(len(data)):
			if pd.isna(data[j,l]):
				data[j,l] = 0
			listdata.append(data[j,l])
	# save list matrix
	matrix1.append(listdata)
	# input the sim with significant data 
	listdata2 = []
	infile2 = f"running_sim{i}_sig_{method}"
	data2 = np.array(pd.read_table(infile2))
	for j in range(len(data2)):
		for l in range(len(data2)):
			if pd.isna(data2[j,l]):
				data2[j,l] = 0
			listdata2.append(data2[j,l])
	# save list matrix
	matrix2.append(listdata2)

print(f"Input the data 1: \n {matrix1}")
print("")
print(f"Initial matrixs processed succesfully")
print("")
print(f"Input the data 1: \n {matrix2}")
print("")
print(f"Simulation matrixs processed succesfully")
print("")

# Calculate cosine similarity between the given matrix and all other matrices
print(f"Global testing of {sim} using Cosine similarity")
similarities = [1 - cosine(matrix1[0], matrix) for matrix in matrix2]
for idx, similarity in enumerate(similarities):
	if idx % 100 == 0:
		print(f"Similarity every 100 lines of Sim_test_{idx}: {similarity}")
	time.sleep(0.01)

# runnin t-test in indidual list
pvalue = []
ttest = []
for i in range(0,len(matrix1)):
	t_stat, p_val = ttest_ind(matrix1[i], matrix2[i])
	print(f" Simulation {i+1}: T-statistic: {t_stat},P-value: {p_val}")
	pvalue.append(p_val)
	ttest.append(t_stat)


#convert to pd
merged_df = pd.concat([pd.DataFrame({'Similarities': similarities}),pd.DataFrame({'P_Values': pvalue}),pd.DataFrame({'T_Test': ttest})],axis=1).sort_values("Similarities",ascending=False)
bestOptions = merged_df[merged_df['Similarities'] == similarities[np.argmax(similarities)]]
merged_df.to_csv("All_Cosine_similarities.csv")
bestOptions.to_csv("Best_Options.csv")

# PCA 3D for cosine similarity in distance
print("Plot a point in 3D of using cosine distance")
data = []
data.append(matrix1[0])
data.extend(matrix2)
pca = PCA(n_components=3)
# Perform PCA to reduce to 3 dimensions
data_3d = pca.fit_transform(data)

# Create a function to generate the plot
def plot_3d_pca(data_3d, angle):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA Plot')
    ax.view_init(30, angle)  # Set the viewpoint angle
    return fig

# Create GIF frames
print("")
print("Making gif file from 3D pics..")
angles = np.linspace(0, 360, 36)  # Generate 36 frames with 10 degrees interval
frames = []
for angle in angles:
    fig = plot_3d_pca(data_3d, angle)
    filename = f'frame_{int(angle)}.png'
    fig.savefig(filename)
    plt.close(fig)
    frames.append(imageio.imread(filename))

# Save frames as GIF
imageio.mimsave('3d_pca_plot.gif', frames, duration=1)

# Cleanup: remove temporary image files
for filename in os.listdir('.'):
    if filename.startswith('frame_'):
        os.remove(filename)

print("Generating the single pic for 3D plot.")
# Create a 3D scatter plot
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA Plot')
# Save the plot as an image file
plt.savefig('3d_pca_plot.png')


print("")
print("Generating all the highest simulation in 3D cube...")
# plot in math3D
data2 = []
for i in range(0,len(list(bestOptions.index))):
     data2.append(matrix2[i])

pca2 = PCA(n_components=3)
data2_3d = pca2.fit_transform(data2)

# Extract vectors for each data point
vectors = np.zeros_like(data2_3d)
for i in range(len(data2_3d)):
    vectors[i] = data2_3d[i] / np.linalg.norm(data2_3d[i]) 

# Create a Plotly 3D scatter plot with vectors
fig = go.Figure()

# Add data points
fig.add_trace(go.Scatter3d(x=data2_3d[:, 0], y=data2_3d[:, 1], z=data2_3d[:, 2], mode='markers'))

# Add vectors
for i in range(len(data2_3d)):
    x, y, z = data2_3d[i]
    vx, vy, vz = vectors[i]
    fig.add_trace(go.Scatter3d(x=[0, vx], y=[0, vy], z=[0, vz], mode='lines', line=dict(color='blue')))

# Set layout
fig.update_layout(scene=dict(aspectmode='cube'))

fig.write_html("plotly_figure.html")
# Save as PNG
fig.write_image("plotly_figure.png", width=1500, height=1200)

print("")


# Display the highest value with the position index
highest_similarity_index = list(bestOptions.index)
print("")
print("#####################################################")
print("Best similarities: " + str(highest_similarity_index))
print(f"There is a {len(highest_similarity_index)} best simulation options, similarity: {similarities[np.argmax(similarities)]}, P-value: {pvalue[np.argmax(similarities)]}, t-test: {ttest[np.argmax(similarities)]}")
print("____________________________________________")
print("")


# testing mean of square for choosing the best one
MSE_value=[]
initial = matrix1[0]
for i in highest_similarity_index:
	mse = mean_squared_error(initial, matrix2[i])
	print(f"Sim {i}, MSE : {mse}")
	MSE_value.append(mse)

print("Mean of squared running test..")
MSE_value
MSE_value.to_csv("MSE_value.csv")
print("")

# Pulling the values
print("Pulling model for the best simulation...")
bestSim_initial = matrix1[0]
bestSim_models = matrix2[highest_similarity_index+1]
print("All cases for comparing in between the populations")
print("")
print(f"Positions-Matrix_:_Initial-Migration_:_Simulation-{highest_similarity_index+1}")
print("__________________________________________________________")
for i,(l,k) in enumerate(list(zip(bestSim_initial,bestSim_models))):
	print(f"Pos {i+1}  :  {l}  :  {k}")


# Create a regression plot with custom colors
print("")
print("Plotting the highest similarities...")
plt.figure(figsize=(13, 8))
sns.regplot(x=list(range(len(similarities))), y=similarities, scatter_kws={"s": 20, "color": "blue", "alpha": 0.5}, line_kws={"color": "green"}, ci=95, scatter=True)

# Highlight the highest point within the regression plot
for i in highest_similarity_index:
	sns.regplot(x=[i], y=[similarities[i]], scatter_kws={"s": 50, "color": "red"}, fit_reg=False)

plt.title(f"Regression Plot of Cosine Similarities")
plt.xlabel("Matrix Index")
plt.ylabel("Similarity")
plt.savefig(f"DivMigrate_simBoots.png",dpi=300)
