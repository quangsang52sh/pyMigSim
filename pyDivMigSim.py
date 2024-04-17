import subprocess
import sys

# Check if the correct number of command line arguments is provided
if len(sys.argv) != 5:
    print("Usage: python script.py <R_script> <genpop_file> <method> <simulation>")
    print("Method writing: nm, g, d")
    sys.exit(1)

# generation command for running in R

infile_Rscript = sys.argv[1]
genpop_file = sys.argv[2]
method = sys.argv[3]
sim = int(sys.argv[4])


for i in range(1,sim+1):
	command = f'Rscript {infile_Rscript} {genpop_file} SimTest_{i} running_sim{i}_{method} running_sim{i}_sig_{method}'
	print(f"______ Running simulation DivMigration of {i} _________")
	subprocess.run(command, check=True, shell = True)
	print(f"___ End of running sim {i} ____ ")
	print(f" ")
	

