import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import sys

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

# Calculate cosine similarity between the given matrix and all other matrices
similarities = [1 - cosine(matrix1[0], matrix) for matrix in matrix2]

# Annotate the highest point with red marker and display its index as text
highest_similarity_index = np.argmax(similarities)
print(f"Best boostrap simulation : {highest_similarity_index}")

# Create a regression plot with custom colors
plt.figure(figsize=(13, 8))
sns.regplot(x=list(range(len(similarities))), y=similarities, scatter_kws={"s": 20, "color": "blue", "alpha": 0.5}, line_kws={"color": "green"}, ci=95, scatter=True)

# Highlight the highest point within the regression plot
sns.regplot(x=[highest_similarity_index], y=[similarities[highest_similarity_index]], scatter_kws={"s": 50, "color": "red"}, fit_reg=False)

# annotation 
plt.annotate(f'Bootstrap: {highest_similarity_index}', 
             xy=(highest_similarity_index, similarities[highest_similarity_index]), 
             xytext=(highest_similarity_index + 0.2, similarities[highest_similarity_index] + 0.08),
             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=0.5'),
             color='red')

plt.title("Regression Plot of Cosine Similarities with Custom Colors")
plt.xlabel("Matrix Index")
plt.ylabel("Similarity")
plt.savefig(f"DivMigrate_simBoots_{highest_similarity_index}.png",dpi=300)
