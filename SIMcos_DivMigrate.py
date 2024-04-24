import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.formula.api import ols
import plotly.graph_objs as go
import sys
import time
import imageio
import os
import subprocess


class cosine_runningStats:
    print("""
	###############################################################
	Thank you for chossing my script......
	Running script after 5s ..
	Please wait a sec !
	_________________________
	
    """)
    time.sleep(5)
    def __init__(self):
        self.matrix1=[]
        self.matrix2=[]
        self.similarities = []
        self.pvalue = []
        self.ttest = []
        self.data = []
        self.highest_similarity_index = []
        self.MSE_value=[]
        
        
    
    def read_infile(self,sim,method):
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
            self.matrix1.append(listdata)
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
            self.matrix2.append(listdata2)
        #print(f"Input the data 1: \n {self.matrix1}")
        print("")
        print(f"Initial matrixs processed succesfully")
        print("")
        #print(f"Input the data 1: \n {self.matrix2}")
        print("")
        print(f"Simulation matrixs processed succesfully")
        print("")
    
    
    def cosine_similarities(self,sim):
        # Calculate cosine similarity between the given matrix and all other matrices
        print(f"Global testing of {sim} using Cosine similarity")
        print("___________________________________________________________")
        self.similarities = [1 - cosine(self.matrix1[0], matrix) for matrix in self.matrix2]
        for idx, similarity in enumerate(self.similarities):
            if idx % 200 == 0:
                print(f"Similarity every 200 lines of Sim_test_{idx}: {similarity}")
            time.sleep(0.01)
    
    def ttest_stats(self):
        # runnin t-test in indidual list
        for i in range(0,len(self.matrix1)):
            t_stat, p_val = ttest_ind(self.matrix1[i], self.matrix2[i])
            print(f" Simulation {i+1}: T-statistic: {t_stat},P-value: {p_val}")
            self.pvalue.append(p_val)
            self.ttest.append(t_stat)
    
    def convert_dataFrame(self):
        #convert to pd
        merged_df = pd.concat([pd.DataFrame({'Similarities': self.similarities}),pd.DataFrame({'P_Values': self.pvalue}),pd.DataFrame({'T_Test': self.ttest})],axis=1).sort_values("Similarities",ascending=False)
        bestOptions = merged_df[merged_df['Similarities'] == self.similarities[np.argmax(self.similarities)]]
        bestOptions = bestOptions[bestOptions['P_Values'] <= 0.05]
        merged_df.to_csv("All_Cosine_similarities.csv")
        bestOptions.to_csv("Best_Options.csv")
        return bestOptions
    
    def PCA_plot(self, bestOptions):
        # PCA 3D for cosine similarity in distance
        print("Plot the points in 3D of cosine distance in PCA")
        self.data.append(self.matrix1[0])
        self.data.extend(self.matrix2)
        pca = PCA(n_components=3)
        # Perform PCA to reduce to 3 dimensions
        data_3d = pca.fit_transform(self.data)
        
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
        imageio.mimsave('3d_pca_plot.gif', frames, duration=3)
        
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
        print("Generating all the highest simulations in 3D cube...")
        # plot in math3D
        data2 = []
        for i in range(0,len(list(bestOptions.index))):
             data2.append(self.matrix2[i])
        
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
    
    def display_highestVal(self,bestOptions): 
        # Display the highest value with the position index
        self.highest_similarity_index = list(bestOptions.index)
        print("")
        print("#####################################################")
        print("Best similarities: " + str(self.highest_similarity_index))
        print(f"There are {len(self.highest_similarity_index)} best simulation options, similarity: {self.similarities[np.argmax(self.similarities)]}, P-value: {self.pvalue[np.argmax(self.similarities)]}, t-test: {self.ttest[np.argmax(self.similarities)]}")
        print("____________________________________________")
        print("")
        print("Running the mean of squared error for looking the best simulation...")
        print("_____________________________________________________________________")
    
    def MSE_stats(self):
        # testing mean of square for choosing the best one
        initial = self.matrix1[0]
        for i in self.highest_similarity_index:
            mse = mean_squared_error(initial, self.matrix2[i])
            print(f"Sim {i}, MSE : {mse}")
            self.MSE_value.append(mse)
        
        self.MSE_value
        pd.DataFrame({'mse':self.MSE_value}).to_csv("MSE_value.csv")
        print("")
    
    def pulling_bestmatrix(self):
        # Pulling the values
        print("Pulling model for the best simulations...")
        bestSim_initial = self.matrix1[0]
        print("All cases for comparing in between the populations")
        print("")
        for i in range(0,len(self.highest_similarity_index)):
            bestSim_models = self.matrix2[i+1]
            print(f"Positions-Matrix_:_Initial-Migration_:_Simulation-{i+1}")
            print("__________________________________________________________")
            for j,(l,k) in enumerate(list(zip(bestSim_initial,bestSim_models))):
                print(f"Pos {j+1}  :  {l}  :  {k}")
                print("_______________________________________")
    
    
    def regplot(self):
        # Create a regression plot with custom colors
        print("")
        print("Plotting the highest similarities...")
        plt.figure(figsize=(13, 8))
        sns.regplot(x=list(range(len(self.similarities))), y=self.similarities, scatter_kws={"s": 20, "color": "blue", "alpha": 0.5}, line_kws={"color": "green"}, ci=95, scatter=True)
        
        # Highlight the highest point within the regression plot
        for i in self.highest_similarity_index:
            sns.regplot(x=[i], y=[self.similarities[i]], scatter_kws={"s": 50, "color": "red"}, fit_reg=False)
        
        plt.title(f"Regression Plot of Cosine Similarities")
        plt.xlabel("Matrix Index")
        plt.ylabel("Similarity")
        plt.savefig(f"DivMigrate_simBoots.png",dpi=300)
    
    
    def OLS_stats(self):
        # Generating a new matrix based on the running stats in Python
        #matrix3=[]
        if len(self.highest_similarity_index) > 1:
            print("Define all the best simulations for running stats")
            for i,j in enumerate(self.highest_similarity_index):
                # Define the data
                name = f"sim_{j+1}"
                globals()[name] = self.matrix2[j]
            
            # Combine the data into a numpy array
            data_new = np.array([globals()[f"sim_{i+1}"] for i in self.highest_similarity_index]).T
            # Ensure data_new is a 2-dimensional array
            data_new = np.atleast_2d(data_new)
            # create dataframe
            df_new = pd.DataFrame(data_new, columns=[f"sim_{i+1}" for i in self.highest_similarity_index])
            # add constant
            df_new['const'] = 1
            df_new = df_new.astype(float)
            # Perform OLS
            print("")
            print("OLS running ....")
            print("________________________")
            # Define independent and dependent variables
            X = df_new.drop(columns=['const'])  # Independent variables (excluding the 'const' column)
            y = df_new['const']  # Dependent variable
            # Add a constant to the independent variables
            X = sm.add_constant(X)
            # Fit the OLS model
            model = sm.OLS(y, X).fit()
            # Extract significant values
            print(model.summary2())
            #pval_ols = model.pvalues
            #pd.DataFrame(np.array(pval_ols.index),np.array(pval_ols)).to_csv("bestData_OLS_pval.csv")
            #olsdata = pd.DataFrame(np.array(pval_ols.index),np.array(pval_ols))
            #pval_ols_mod = pd.DataFrame(olsdata[0][olsdata.index != 0.0])
            # Create a DataFrame to store the results
            results_df = pd.DataFrame({
                    'Coefficients': model.params
            })
            # Save the best result to a CSV file
            results_df = results_df.sort_values('Coefficients',ascending=False)
            results_df.to_csv('bestData_ols_Coefficient.csv')
            # Find the index corresponding to the highest fitted value
            index_of_max_value = results_df['Coefficients'].iloc[1:].idxmax()
            # Retrieve the row with the highest fitted value
            best_result = results_df.loc[index_of_max_value]
            print("")
            print("###########################")
            print("Your final simulation after running OLS was:")
            print(best_result)
            print("")
            print("__________________________________________")
            print("End of running simulation...")	
        else:
            print(f"The best simulation is {matrix2[highest_similarity_index+1]} : sim {highest_similarity_index+1}")
            print("__________________________________________")
            print("End of running simulation...")

def main(method, sim_count):
    CR = cosine_runningStats()
    CR.read_infile(sim_count,method)
    CR.cosine_similarities(sim_count)
    CR.ttest_stats()
    bestOpt = CR.convert_dataFrame()
    CR.PCA_plot(bestOpt)
    CR.display_highestVal(bestOpt)
    CR.MSE_stats()
    CR.pulling_bestmatrix()
    CR.regplot()
    CR.OLS_stats()


if __name__ == "__main__":
    """Check if the correct number of command line arguments is provided"""
    if len(sys.argv) != 3:
        print("Usage: python script.py <method> <simulation>")
        print("Method writing: nm, g, d")
        sys.exit(1)
    
    method = sys.argv[1]
    sim_count = int(sys.argv[2])
    main(method, sim_count)
