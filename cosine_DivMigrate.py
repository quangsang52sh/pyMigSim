import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
import sys
import time


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

MSE_value.to_csv("MSE_value.csv")


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
plt.figure(figsize=(13, 8))
sns.regplot(x=list(range(len(similarities))), y=similarities, scatter_kws={"s": 20, "color": "blue", "alpha": 0.5}, line_kws={"color": "green"}, ci=95, scatter=True)

# Highlight the highest point within the regression plot
for i in highest_similarity_index:
	sns.regplot(x=[i], y=[similarities[i]], scatter_kws={"s": 50, "color": "red"}, fit_reg=False)

# annotation 
#plt.annotate(f'Bootstrap: {highest_similarity_index+1}', 
#             xy=(highest_similarity_index, similarities[highest_similarity_index]), 
#             xytext=(highest_similarity_index - 0.01, similarities[highest_similarity_index] - 0.08),
#             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=0.5'),
#             color='red')

plt.title(f"Regression Plot of Cosine Similarities")
plt.xlabel("Matrix Index")
plt.ylabel("Similarity")
plt.savefig(f"DivMigrate_simBoots.png",dpi=300)
