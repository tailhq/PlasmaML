#! /usr/bin/Rscript
args <- commandArgs(trailingOnly = TRUE)
year <- args[1]
setwd("data/")
prefix <- "omni2_"
system(paste("wget ftp://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/",
             prefix, year, ".dat", sep = ""))
system(paste("sed -i 's/ \\{1,\\}/,/g' ", prefix, year, ".dat", sep = ""))


df <- read.csv(paste(prefix, year, ".dat", sep = ""), 
               header = FALSE, stringsAsFactors = FALSE, 
               colClasses = rep("numeric",55), 
               na.strings = c("99", "999.9", 
                              "9999.", "9.999", "99.99", 
                              "9999", "999999.99", 
                              "99999.99", "9999999."))

library(MonetDB.R)
bat.file <- monetdb.server.setup(paste(getwd(), "monetdb", sep="/"),
                                 monetdb.program.path = "~/monetdb/bin/", 
                                 dbname = "omni", dbport = 50000)
correct.pid <- monetdb.server.start(bat.file)

conn <- dbConnect(MonetDB.R(), "monetdb://localhost:50000/omni")

monetdb.read.csv(conn, paste(prefix, year, ".dat", sep = ""), "omniHourly", 
                 nrows=8760, header=FALSE,
                 locked=FALSE, 
                 na.strings=c("99", "999.9", 
                              "9999.", "9.999", "99.99", 
                              "9999", "999999.99", 
                              "99999.99", "9999999."), 
                 nrow.check=500, delim=",", 
                 newline = "\\n", quote = "\"", 
                 create=TRUE)

monetdb.server.stop(correct.pid, wait = TRUE)