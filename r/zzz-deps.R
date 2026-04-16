
library(arrow)
library(xgboost)
library(data.table)
library(ggplot2)
library(ggpattern)
library(ggdendro)
library(ggtext)
library(latex2exp)
library(glue)
library(withr)

f <- function(...) as.character(glue::glue(..., .envir = parent.frame()))

expit <- function(x) exp(x) / (1 + exp(x))

sync_data <- function() {
  dataset <- "fgai"
  remote <- "ault"
  system(f("rsync -avz --update {remote}:{dataset}/data/ data/"))
  
  remote <- "clariden"
  system(f("rsync -avz --update {remote}:{dataset}/data/ data/"))
}

DATASETS <- c("nsduh", "brfss", "census_income")

DATASET_NAMES <- c(nsduh = "NSDUH (Marijuana)", brfss = "BRFSS (Diabetes)", 
                   census_income = "ACS Census (Income)")

MODELS <- c("llama3_8b", "ministral3_8b", "gemma3_4b", "qwen35_9b",
            "deepseek_7b", "phi4", "qwen35_27b", "gemma3_27b", "deepseek_r1", 
            "llama3_70b")

MODEL_NAMES <- c(
  llama3_8b      = "Llama 3 8B",
  llama3_70b     = "Llama 3.3 70B",
  ministral3_8b  = "Ministral 3 8B",
  gemma3_4b      = "Gemma 3 4B",
  gemma3_27b     = "Gemma 3 27B",
  qwen35_9b     = "Qwen 3.5 9B",
  qwen35_27b    = "Qwen 3.5 27B",
  deepseek_7b    = "DeepSeek 7B",
  deepseek_r1    = "DeepSeek-R1 32B",
  phi4           = "Phi-4"
)
