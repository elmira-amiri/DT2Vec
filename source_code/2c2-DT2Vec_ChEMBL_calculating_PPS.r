library(protr)
library(dplyr)
library(GOSemSim)
library(Biostrings)
library(foreach)
library(doParallel)

df_DTI <- read.csv(file = "Entrez_protein_DTI.csv", header = FALSE)
df_DTI$V1 <- as.character(df_DTI$V1)
genelist = c(df_DTI$V1)
sprintf("Number of genes in DTI:  %s", length(genelist))

protlist_temp <- read.csv(file = "uniprot_Seq.csv")
head(protlist_temp,3)

protlist <- c(dplyr::pull(protlist_temp,Sequence))

PPS_matrix = parSeqSim(protlist, cores = 8, batches = 4, type = "local", submat = "BLOSUM62")

write.csv(PPS_matrix, file = sprintf("PPS(seq)_known_ChEMBLid_P%s.csv", length(genelist)))
