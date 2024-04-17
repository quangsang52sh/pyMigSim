import subprocess
import sys

# Check if the correct number of command line arguments is provided
if len(sys.argv) != 3:
    print("Usage: python script.py <R_script> <genpop_file>")
    sys.exit(1)

# generation command for running in R

infile_Rscript = sys.argv[1]
genpop_file = sys.argv[2]


for i in range(1,10000):
	command = f'Rscript {infile_Rscript} {genpop_file} SimTest_{i} running_sim{i} running_sim{i}_sig'
	print(f"______ Running simulation DivMigration of {i} _________")
	subprocess.run(command, check=True, shell = True)
	print(f"___ End of running sim {i} ____ ")
	print(f" ")
	

