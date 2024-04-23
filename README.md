# pyMigSim
DivMigrate simulation in python (`pyMigSim`) running 10000 replications for chossing the highest values apllied the distance in `cosine`. The method compute cosine distance between two vectors in spatial using `scipy` module in python. <br/>
To run simulation: <br/>
- Running migration using DivMigrate in R studio (`python pyDivMigSim.py <R_script> <genpop_file> <method [nm|d|g]> <simulation>`)
- Running cosine distance to figure out the highest value of simulation number (`python cosine_DivMigrate.py <method [nm|d|g]> <simulation>`)
- Running statistics of all the best simulations to select the best one (`python bestMigrateSim.py`) 
