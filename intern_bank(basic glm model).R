rm(list = ls(all = T))
getwd()

setwd("C:/Users/Hanu/Documents/insofe/internship/bank/bank-additional")

#-----------------------------------------------------------------------------------------

bank_m<- read.csv("bank-additional-full.csv", sep = ";")

str(bank_m)

sum(is.na(bank_m))

summary(bank_m)

hist(bank_m$age)

hist(bank_m$euribor3m)

hist(bank_m$duration)

library(caret)
library(dplyr)
library(DMwR)
library(tidyr)
library(ggplot2)
library(e1071)
library(tidyverse)

#-------------------------------------univariate analysis--------------------------------

plot(x = bank_m$age, y = bank_m$euribor3m, "h" )

plot(x = bank_m$age, y = bank_m$duration, type = "h") #right skewed

kurtosis(bank_m$age)


sd(bank_m$age)

range(bank_m$age)

boxplot(bank_m$age)


bank_m %>% 
  ggplot(aes(x = age, 
             y = duration)) + 
  geom_point(size = 2, alpha = 0.5)


bank_m %>% 
  ggplot(aes(x = age,
             y = duration))  +
  geom_point(size = 2, 
             alpha = 0.5, aes(color = age)) + 
  labs(title = "age vs duration",
       x = "age",
       y = "duration per minutes") +
  theme_minimal()+scale_color_gradient(low = "red", high = "blue")




#---------------------------categorical variables-----------------------------
#---------------------------seasons in  which loans were taken-----------
moth<- table(bank_m$month)

barplot(moth, main = "month distribution", xlab ="months" )   

#----------------------------------no of calls in a single campaign-------
campaign<- table(bank_m$campaign)

barplot(campaign, main = "no of calls done in this campaign", xlab = "campaign")

#---------------------------job demographics-----------------
job<- table(bank_m$job)

barplot(job, main = "job demographics", xlab = "jobs")

#--------------------------days of the week--------------------------

day<- table(bank_m$day_of_week)

barplot(day, main = "days of the week", xlab = "days")

#-------------------------------has housing loan---------------------

house<- table(bank_m$housing)

barplot(house, main = "has housing loan", xlab = "housing loan")

#--------------------------------no of people having loan-------------

loan<- table(bank_m$loan)

barplot(loan, main = "no of people having loan", xlab = "loan")

#--------------------------------bi variate analysis------------------

library(car)

par(mfrow=c(4,2))

par(mar = rep(2, 4))

scatterplot(age ~ euribor3m | duration, data=bank_m,
            xlab="euribor3m", ylab="age",
            main="Enhanced Scatter Plot",
            labels=row.names(bank_m)) 

plot(bank_m$age, bank_m$euribor3m, main="Scatterplot Example",
     xlab="age ", ylab="euro ", pch=19) 

library(corrplot)


cor_data<- cor(bank_m, method = "s")

#-----------------------------------------------------------------------

#install.packages("factoextra") 
# if(!require(devtools)) install.packages("devtools")
# devtools::install_github("kassambara/factoextra")

library(factoextra)
# Use the get_dist() function from the factoexrtra to calculate inter-observation distances
distance <- get_dist(bank_m)

# The fviz_dist() function plots a visual representation of the inter-observation distances
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
# The gradient argument, helps us define the color range for the distance scale
#------------------------------------------------------------------------

devtools::install_github("wilkelab/cowplot")

install.packages("devtools")

install.packages("ggpmisc")

library(ggplot2)
library(ggpubr)
theme_set(
  theme_minimal() +
    theme(legend.position = "top")
)

df<- bank_m

df$euribor3m<- as.factor(df$euribor3m)

b <- ggplot(df, aes(x = age, y = duration))
# Scatter plot with regression line
b + geom_point()+
  geom_smooth(method = "lm") 

# Add a loess smoothed fit curve
b + geom_point()+
  geom_smooth(method = "loess") 

#-------------------------------pre processing---------------------------
#------------------------------------------------------------------------

std_m = preProcess(bank_m[, !(names(bank_m) %in% "y")], method = c("center","scale"))

bank_m <- predict(std_m, bank_m)



#---------------------------------train test-----------------------------

library(caret)

set.seed(123)

train_rows<- createDataPartition(bank_m$y, p = 0.7, list = F)

train_data<- bank_m[train_rows,]

test_data<- bank_m[-train_rows, ]

#--------------------------------------model------------------------------

log_reg<- glm(y~., data = train_data, family = binomial)

summary(log_reg)

prob_train <- predict(log_reg, type = "response")

library(ROCR)

pred <- prediction(prob_train, train_data$y)

perf <- performance(pred, measure="tpr", x.measure="fpr")

plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))

perf_auc <- performance(pred, measure="auc")

# Access the auc score from the performance object

auc <- perf_auc@y.values[[1]]

print(auc)


prob_test <- predict(log_reg, test_data, type = "response")

preds_test <- ifelse(prob_test > 0.15, "yes", "no")


test_data_labs <- test_data$y

conf_matrix <- table(test_data_labs, preds_test)

print(conf_matrix)


#--------------------------------specificity---------------------------
specificity <- conf_matrix[1, 1]/sum(conf_matrix[1, ])

print(specificity)


#-------------------------------sensitivity---------------------------
sensitivity <- conf_matrix[2, 2]/sum(conf_matrix[2, ])

print(sensitivity)


#-------------------------------accuracy------------------------------
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)

print(accuracy)


library(caret)

# Using the argument "Positive", we can get the evaluation metrics according to our positive referene level

confusionMatrix(preds_test, test_data$y, mode = "prec_recall" ,positive = "yes")

