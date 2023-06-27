library(readxl)
library(car)
library(tidyverse)
library(caret)

car_data <- read_excel("~/Fall 2022/ECON/ECON453_used_car_data.xlsx")
View(car_data)
summary(car_data)

### Declare variables for binary/dummy parameters ###
car_data$BIN_leather = ifelse((car_data$leather_interior == "yes"),1,0)
car_data$BIN_sunroof = ifelse((car_data$sunroof == "yes"),1,0)
car_data$BIN_accident = ifelse((car_data$accident_reported == "yes"),1,0)

# Car type binary parameters
car_data$BIN_coupe = ifelse((car_data$car_type == "coupe"),1,0)
car_data$BIN_truck = ifelse((car_data$car_type == "truck"),1,0)
car_data$BIN_sedan = ifelse((car_data$car_type == "sedan"),1,0)
car_data$BIN_suvx = ifelse((car_data$car_type == "suv/crossover"),1,0)

car_data$BIN_trans = ifelse((car_data$transmission == "automatic"),1,0)

# Car color binary parameters
car_data$BIN_gray = ifelse((car_data$exterior_color == "gray"),1,0)
car_data$BIN_silver = ifelse((car_data$exterior_color == "silver"),1,0)
car_data$BIN_white = ifelse((car_data$exterior_color == "white"),1,0)
car_data$BIN_blue = ifelse((car_data$exterior_color == "blue"),1,0)
car_data$BIN_red = ifelse((car_data$exterior_color == "red"),1,0)

### First Model ###

linear_model1 = lm(data = car_data, 
                   asking_price~current_mileage + year + BIN_coupe + BIN_truck 
                   + BIN_suvx + BIN_trans + mpg_city + mpg_hwy
                   + BIN_white + BIN_gray + BIN_silver + BIN_blue + BIN_red
                   +BIN_leather + BIN_sunroof + owners + BIN_accident)

summary(linear_model1)

# Confidence interval for all variables for linear model
confint(linear_model1)

# IMPLEMENT ANALYSIS HERE #

# We are testing if the color has any effect on the asking price in relation to
# the other variables mentioned. So, set all color binary variables to 0

linearHypothesis(linear_model1,c("BIN_red=0","BIN_blue=0","BIN_white=0",
                 "BIN_gray=0","BIN_silver=0"))

# Now let's make some predictions...
# For this example were going for a coupe with most features included. Expecting
# the asking price to be on the higher end.
linear_pred <- predict(linear_model1, data.frame(current_mileage = 40000, year = 2020, 
                                  mpg_city = 20,mpg_hwy = 23, BIN_coupe = 1, 
                                  BIN_truck = 0,BIN_suvx = 0, BIN_accident = 0, 
                                  BIN_trans = 1,BIN_red = 1, BIN_white = 0, 
                                  BIN_blue = 0, BIN_gray = 0,BIN_silver = 0, 
                                  BIN_leather = 1, BIN_sunroof = 0, owners = 1),
                                  interval = "predict",level = 0.95)

# Confirming that the prediction is reasonable...

# average price of cars in data set
average_price = mean(car_data$asking_price) 
average_price

### Second Model ###

log_model1 = glm(data = car_data, asking_price~log(current_mileage)+log(year)
                 + BIN_coupe + BIN_truck + BIN_suvx + BIN_trans + mpg_city 
                 + mpg_hwy +BIN_white + BIN_gray + BIN_silver + BIN_red 
                 + BIN_blue + BIN_leather + BIN_sunroof + owners + BIN_accident)

summary(log_model1)

linearHypothesis(log_model1, c("BIN_red=0","BIN_blue=0","BIN_white=0",
                 "BIN_gray=0","BIN_silver=0"))

# Make the same prediction as before but now using the log model
log_pred <- predict(log_model1, data.frame(current_mileage = 40000, year = 2020, 
                                  mpg_city = 20,mpg_hwy = 23, BIN_coupe = 1, 
                                  BIN_truck = 0,BIN_suvx = 0, BIN_accident = 0, 
                                  BIN_trans = 1,BIN_red = 1, BIN_white = 0, 
                                  BIN_blue = 0, BIN_gray = 0,BIN_silver = 0, 
                                  BIN_leather = 1, BIN_sunroof = 0, owners = 1),
                                  interval = "predict", level = 0.95)

# Confidence interval for log model
# This gives the confidence interval for every variable in the log model
confint(log_model1,level = 0.95)

### Cross Validation ### 

# Partition our data
training_set = car_data[1:140,]
validation_set = car_data[141:200,]

# creating new models that are using the training set
linear_model1_cv <- lm(data = training_set, 
                          asking_price~current_mileage + year + BIN_coupe + BIN_truck 
                          + BIN_suvx + BIN_trans + mpg_city + mpg_hwy
                          + BIN_white + BIN_gray + BIN_silver + BIN_blue + BIN_red
                          +BIN_leather + BIN_sunroof + owners + BIN_accident)

log_model1_cv <- glm(data = training_set, asking_price~log(current_mileage)+log(year)
                     + BIN_coupe + BIN_truck + BIN_suvx + BIN_trans + mpg_city 
                     + mpg_hwy +BIN_white + BIN_gray + BIN_silver + BIN_red 
                     + BIN_blue + BIN_leather + BIN_sunroof + owners + BIN_accident)

# new predictions using the new models
pred1 <- predict(linear_model1_cv,validation_set)

# RMSE calculation for pred1
sqrt(mean((validation_set$asking_price-pred1)^2))

pred2 <-predict(log_model1_cv,validation_set)

# RMSE calculation for pred2
sqrt(mean((validation_set$asking_price-pred2)^2))

# when comparing results, we see that pred2 is a bit higher 
# than pred1, but within reason (about 2000)
