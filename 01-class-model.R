options(stringsAsFactors = FALSE)
library(tidyverse)
library(VGAM)
library(kernlab)

comp <- read.csv('complaint-topics.csv')
colnames(comp) <- gsub('\\.', '_', tolower(colnames(comp)))
View(head(comp))

large_prods <- comp %>% 
  group_by(company) %>% 
  count() %>%
  ungroup() %>% 
  filter(n > 100)

comp2 <- comp %>%
  filter(company %in% large_prods$company)

m <- vglm(topic10 ~ product + company,
          data = comp2,
          family = 'multinomial')

m <- ksvm(as.factor(topic10) ~ product + company,
          data = comp2)

comp2$fits <- fitted(m)

tb <- table(comp2$topic10, comp2$fits)

tb %>% prop.table(1) %>% round(3) %>% plot()
tb %>% prop.table(1) %>% round(3) %>% diag()
tb %>% prop.table(2) %>% round(3) %>% diag()
