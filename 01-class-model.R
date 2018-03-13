options(stringsAsFactors = FALSE)
library(tidyverse)
library(VGAM)
library(kernlab)

comp <- read.csv('complaint-topics.csv')
colnames(comp) <- gsub('\\.', '_', tolower(colnames(comp)))

productLookup <- c(
  'Bank account or service' = 'Bank Account',
  'Checking or savings account' = 'Bank Account',
  'Consumer Loan' = 'Consumer Loan',
  'Credit card' = 'Credit card',
  'Credit card or prepaid card' = 'Credit card',
  'Credit reporting' = 'Credit reporting',
  'Credit reporting, credit repair services, or other personal consumer reports' = 'Credit reporting',
  'Debt collection' = 'Debt collection',
  'Money transfer, virtual currency, or money service' = 'Money transfers',
  'Money transfers' = 'Money transfers',
  'Mortgage' = 'Mortgage',
  'Other financial service' = 'Other',
  'Payday loan' = 'Payday loan',
  'Payday loan, title loan, or personal loan' = 'Payday loan',
  'Prepaid card' = 'Credit card',
  'Student loan' = 'Student loan',
  'Vehicle loan or lease' = 'Vehicle loan or lease',
  'Virtual currency' = 'Money transfers'
)

comp$productgrp = productLookup[comp$product]
table(comp$product, comp$productgrp)
tb <- table(comp$productgrp, comp$topic11)
tb %>% prop.table(1) %>% round(3)
tb %>% prop.table(2) %>% round(3)

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
