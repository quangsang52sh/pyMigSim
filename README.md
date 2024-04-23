# pyMigSim
The DivMigrate simulation in Python (`pyMigSim`) runs `10,000` replications to select the highest values, employing cosine distance calculation. The method computes the cosine distance between two vectors in space using the SciPy module in Python. <br/>
To run the simulation: <br/>
- Execute migration using DivMigrate in RStudio (`python pyDivMigSim.py <R_script> <genpop_file> <method [nm|d|g]> <simulation>`)
- Determine the highest value of the simulation number using cosine distance (`python cosine_DivMigrate.py <method [nm|d|g]> <simulation>`)
- Compute statistics for all the best simulations to select the optimal one (`python bestMigrateSim.py`)

# Require package
```
Rstudio 4.2.2
r-diveRsity
python 3.8.15
numpy
pandas
seaborn
matplotlib
scipy
statsmodels
sklearn
mpl_toolkits
plotly
imageio
```
