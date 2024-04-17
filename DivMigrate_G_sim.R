library(diveRsity)


if (length(commandArgs(trailingOnly = TRUE)) < 4) {
  print("")
  print("######################################################################################################")
  print("Usage: Rscript DivMigrate_sim.R <input.gen> <outFolderDivMigrate> <outMigRate> <outMigRateSig> ")
  print("######################################################################################################")
  stop()
}


genpopFile = commandArgs(trailingOnly = TRUE)[1]
outDivMigrate = commandArgs(trailingOnly = TRUE)[2]
outMigRate = commandArgs(trailingOnly = TRUE)[3]
outMigRateSig = commandArgs(trailingOnly = TRUE)[4]

pdf(paste0(outDivMigrate,".pdf"),width=10,height=8)
run <- divMigrate(infile = genpopFile, plot_network=T, boots=0.05)
dev.off()
print("divMigrate running with input genpop file successfully.....")
write.table(run$gRelMig, outMigRate, sep="\t",row.names=F)
write.table(run$gRelMigSig,outMigRateSig, sep="\t",row.names=F)
print("Writing the migration rate in matrix .....")
