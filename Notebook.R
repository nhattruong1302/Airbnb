# Load libraries
library(estimatr)
library(gplots)
library(plm)

library(lmtest)
library(olsrr)
library(fastDummies)

library(faraway)
library(usdm)
library(car)

library(glmnet)

# graphical packages
library(ggplot2)
library(GGally)

# library for count models
library(MixAll)
library(pscl)
library(MASS) 
library(rsq)

# data manipulation
library(dplyr)
library(gplots)

# lib for confusion matrix 
library(caret)
library(e1071)

###################################################################
#### GOAL 1: Examine what factors affect Airbnb listing prices ####
###################################################################

# Load listing data and omit n/a values
listing <- read.csv("airbnb_listing_v2.csv", na.strings = "?")

listing <- na.omit(listing)

# Visualize variable price
hist(listing$price, xlab = "price", ylab = "Frequency",
     main = "Histogram of Price Distribution")

# Visualize variable price_log
hist(listing$log_price, xlab = "price_log", ylab = "Frequency",
     main = "Histogram of price_log Distribution")

# data for regression
colnames(listing)
listing_1 <- listing[,-c(1, 2, 4, 5)]

# Create dummy variables for categorical variables
listing_2 <-dummy_cols(listing_1, select_columns = c("host_age", "host_gender", "host_race", "city"), 
                          remove_first_dummy = TRUE, remove_selected_columns=TRUE)

# Run multi-regression of 'log_price' ~ all variables
model_2 <- lm(log_price ~ ., data=listing_2)
summary(model_2)

# model diagnostics
par(mfrow = c(2,2))
plot(model_2)
par(mfrow = c(1,1))

# VIF Stepwise (for nummerical variables only)
colnames(listing_2)
usdm::vifstep(listing_2, th=5)

#Fitting the Lasso model
X=as.matrix(listing_2[,-c(1)])
y=as.matrix(listing_2["log_price"])
lasso_10 <- glmnet(X, y, alpha =1 ,lambda= 10) # alpha = 0 for Ridge, 1 for Lasso 
coef_10 <- round(coef(lasso_10),4)
coef_10

# which columns are non-zeros (selected columns)
names(lasso_10)
lasso_10$beta
which(lasso_10$beta!=0)

# run regression model without host_race_white
colnames(listing_2)
model_3 <- lm(log_price ~ ., data=listing_2[,-c(24)])
summary(model_2)


############################################################
#### GOAL 2: Examine how listing price affects demand  ####
############################################################

# Quick check "total_guests" ~ "log_price"
pois1 <- glm(total_guests ~ log_price, data = listing, family = poisson)
summary(pois1)
exp(pois1$coefficients)

AIC(pois1)
BIC(pois1)
logLik(pois1)

# Create data for analysis, including outcome variables "log_price", "log_total_guests", and other variables of characteristics
colnames(listing)
listing_3 <- listing[,-c(1, 2, 5)]
listing_4 <-dummy_cols(listing_3, select_columns = c("host_age", "host_gender", "host_race", "city"), 
                       remove_first_dummy = TRUE, remove_selected_columns=TRUE) 

# multivariate poisson regression
pois1m <- glm(total_guests ~ ., data = listing_4, family = poisson)
summary(pois1m)
exp(pois1m$coefficients)

# assess multicolinearity
usdm::vifstep(listing_4[,-2])

## compare model fit
models <- list("Pois1" = pois1, "Pois1m" = pois1m)
rbind(AIC = sapply(models, function(x) AIC(x)), 
      BIC = sapply(models, function(x) BIC(x)), 
      logLik = sapply(models, function(x) logLik(x)))

#################################################################################################
#### GOAL 3: Examine where there is discrimination of host response towards gender and race  ####
#################################################################################################

# Get the data
response <- read.csv("airbnb_request_response_v2.csv")

# Combine data to df
df <- merge(listing, response, by = "listing_id")

# summarize by group: delay by airlines
summarise(group_by(response, guest_gender, guest_race), acceptance_prob = mean(host_response_yes))

# prepare data for Logistic regression
colnames(df)
df1 <- df[,-c(1,2,3,4,5)]
colnames(df1)
df2 <-dummy_cols(df1, select_columns = c("host_age", "host_gender", "host_race", "city", "guest_race", "guest_gender"), 
                 remove_first_dummy = TRUE, remove_selected_columns=TRUE) 
colnames(df2)

# fit Logistic regression model with guest characteristics
log0 <- glm(host_response_yes ~ guest_race_white + guest_gender_male,  data=df2, family =binomial)
summary(log0)

# fit Logistic regression model
log_1 <- glm(host_response_yes ~ .,  data=df2, family =binomial)
summary(log_1)

# Goodness of fit: AIC, BIC, logLikelihood Pseudo R2 
# logLik(log_1)
# AIC(log_1)
# BIC(log_1)

# Pseudo Rsquare
#1- log_1$deviance/log_1$null.deviance

models <- list("log_0" = log0, "log_1" = log_1)
rbind(AIC = sapply(models, function(x) AIC(x)), 
      BIC = sapply(models, function(x) BIC(x)), 
      logLik = sapply(models, function(x) logLik(x)),
      Pseudo_R2 = sapply(models, function(x) 1- log_1$deviance/log_1$null.deviance))


# Create four categories based on race and gender of guests: black female, black male, white female, and white male
colnames(df2)
df2$race_gender <- 'N/A'
df2$race_gender[df2$guest_race_white == 0 & df2$guest_gender_male == 0] <- "black female"
df2$race_gender[df2$guest_race_white == 0 & df2$guest_gender_male == 1] <- "black male"
df2$race_gender[df2$guest_race_white == 1 & df2$guest_gender_male == 0] <- "white female"
df2$race_gender[df2$guest_race_white == 1 & df2$guest_gender_male == 1] <- "white male"
table(df2$race_gender)

# fit Logistic regression model with race and gender categories
log_2 <- glm(host_response_yes ~ as.factor(race_gender),  data=df2, family =binomial)
summary(log_2)

exp(log_2$coefficients)

# Goodness of fit: AIC, BIC, logLikelihood Pseudo R2 
models <- list("log_0" = log0, "log_1" = log_1, "log_2" = log_2)
rbind(AIC = sapply(models, function(x) AIC(x)), 
      BIC = sapply(models, function(x) BIC(x)), 
      logLik = sapply(models, function(x) logLik(x)),
      Pseudo_R2 = sapply(models, function(x) 1- log_1$deviance/log_1$null.deviance))

# Prediction: P(host_response_yes=1 (M)) 
#df2$pred_prob <- predict(log_1, df2, type="response")
#head(df2$pred_prob)

# Check statistic parameter of column 'pred_prob'
#summary(df2$pred_prob)
#hist(df2$pred_prob, xlab = "pred_prob", ylab = "Frequency",
#     main = "Histogram of pred_prob Distribution")

# Convert prediction to (0,1) prediction with cutoff = 0.5
#df2$pred <- 0
#df2$pred[df2$pred_prob>0.5] <- 1
#str(df2)

# Factorize df2$pred as logical
#df2$pred <- as.logical(df2$pred)
#df2$host_response_yes <- as.logical(df2$host_response_yes)
#str(df2)

# Confusion matrix to test quality of prediction/model fit
#conf_table <- confusionMatrix(as.factor(df2$pred), as.factor(df2$host_response_yes),positive="TRUE")
#conf_table
