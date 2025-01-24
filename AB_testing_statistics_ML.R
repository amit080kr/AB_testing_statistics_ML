# A/B Testing using Statistical Inference

# Libraries
library(tidyverse)
library(broom)
library(car)

# Load Data
control <- read_csv("control_data.csv")
experiment <- read_csv("experiment_data.csv")

##########################################################################################################################

# Exploratory Data Analysis (EDA)
# Summarize key statistics
summary(control)
summary(experiment)

# Check for missing data
control %>% summarize_all(~ sum(is.na(.)))
experiment %>% summarize_all(~ sum(is.na(.)))

# Remove rows with missing values in Enrollments and Payments
control <- control %>% filter(!is.na(Enrollments) & !is.na(Payments))
experiment <- experiment %>% filter(!is.na(Enrollments) & !is.na(Payments))

# Visualize key metrics
control %>%
  ggplot(aes(x = Enrollments)) +
  geom_histogram(binwidth = 5, fill = "blue", alpha = 0.6) +
  labs(title = "Control Group: Enrollments Distribution")

experiment %>%
  ggplot(aes(x = Enrollments)) +
  geom_histogram(binwidth = 5, fill = "green", alpha = 0.6) +
  labs(title = "Experiment Group: Enrollments Distribution")

#########################################################################################################################

# Hypothesis Formulation

# Null Hypothesis (H₀): There is no difference in enrollments between the control and experiment groups.
# Alternative Hypothesis (H₁): The experiment group has a different mean enrollments compared to the control group.

#########################################################################################################################

# Static Hypothesis Testing

# Test for Normality using Shapiro-Wilk

# Control group normality

shapiro.test(control$Enrollments)

# Since p-value ≥ 0.05, we fail to reject the null hypothesis.
# The control group's Enrollments data appears to follow a normal distribution.

# Experiment group normality

shapiro.test(experiment$Enrollments)

# Since p-value ≥ 0.05, we fail to reject the null hypothesis.
# The experiment group's Enrollments data also appears to follow a normal distribution.

# Both are normally distributed

# Test for Equal Variances
# Using Levene’s Test(robust to outlier and can be used for both normal or not normally distributed) or Bartlett’s Test(for normally distributed data):

leveneTest(Enrollments ~ group,
           data = bind_rows(control %>% mutate(group = "control"),
                            experiment %>% mutate(group = "experiment")))

# The variances of the control and experiment groups' Enrollments data are equal 
# we can use statistical homogeneity variances test

# Perform a T-Test (Difference in Means)
# Since the data is normally distributed and have equal variance we can use t-test if the data is not normally distributed we would have use non-parametric Mann-Whitney U test

t_test_result <- t.test(control$Enrollments,
                        experiment$Enrollments,
                        alternative = "two.sided",
                        var.equal = TRUE)
t_test_result

# Analyze Proportions (CTR)

# Create a contingency table

click_table <- matrix(c(
  sum(control$Clicks, na.rm = TRUE),  # Total clicks in control group
  sum(experiment$Clicks, na.rm = TRUE)  # Total clicks in experiment group
), nrow = 2, byrow = TRUE)

# Inspect the table
click_table

# Perform Chi-Square Test
chisq_test_result <- chisq.test(click_table)
chisq_test_result

# Since p-value ≥ 0.05 No significant difference in click behavior

#########################################################################################################################

# Confidence Intervals
# Confidence interval for difference in means

t_test_result$conf.int

# Confidence interval for proportions
prop.test(
  x = c(sum(control$Clicks, na.rm = TRUE), sum(experiment$Clicks, na.rm = TRUE)),
  n = c(sum(control$Pageviews, na.rm = TRUE), sum(experiment$Pageviews, na.rm = TRUE))
)

#Proportions (CTR):
 
# prop 1 (control): 8.15% CTR
# prop 2 (experiment): 8.17% CTR
# Since p-value > 0.05, we fail to reject the null hypothesis.
# There is no statistically significant difference in CTR between the control and experiment groups.
# X-squared = 0.030969 This value is small, indicating minimal deviation between the observed and expected proportions.

#########################################################################################################################

# Power Analysis
# Estimate sample size or power for detecting a significant effect

library(pwr)

# Power analysis for t-test
pwr.t.test(d = 0.2, sig.level = 0.05, power = 0.8, type = "two.sample")

# We need approximately 394 samples in each group (control and experiment) to detect a small effect size with 80% power at a 5% significance level.
# If the sample size is smaller than this, the test may lack sensitivity to detect meaningful differences.

# Power analysis for proportions
pwr.2p2n.test(
  h = ES.h(0.2, 0.3),
  n1 = nrow(control),
  n2 = nrow(experiment),
  sig.level = 0.05
)

# The power is very low, meaning there is only a 12.33% chance of detecting a true difference in proportions between groups.
# This suggests the current sample size is insufficient for this test.

#########################################################################################################################

# Plot Distributions
combined_data <- bind_rows(control %>% mutate(group = "control"),
                           experiment %>% mutate(group = "experiment"))

combined_data %>%
  ggplot(aes(x = Enrollments, fill = group)) +
  geom_density(alpha = 0.6) +
  labs(title = "Distribution of Enrollments: Control vs Experiment")

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

# A/B Testing using ML

# Load Additional Libraries
library(tidyquant)

# Modeling packages
library(recipes)
library(broom)

# Connector packages
library(rpart)
library(rpart.plot)
library(xgboost)

library(rsample)
library(parsnip)
library(yardstick)

# Import data
 
# Control population
control <- read_csv("control_data.csv")

# Experiment population
experiment <- read_csv("experiment_data.csv")

# Investigate the Data
control %>% head(5)
control %>% glimpse()
experiment %>% glimpse()

##########################################################################################################################

# Data Quality Check

# Check for Missing Data in control population
 
control %>%
   map_df(~ sum(is.na(.))) %>%
   gather(key = "feature", value = "missing_count") %>%
   arrange(desc(missing_count))

# There are 14 missing data points from Enrollments and payments column
 
# Check for Missing Data in experiment population
 
experiment %>% 
   map_df(~ sum(is.na(.))) %>%
   gather(key = "feature", value = "missing_count") %>%
   arrange(desc(missing_count))

# Control and Experiment population are consistent i.e 14 missing point in both the group
 
# Missing values
control %>% filter(is.na(Enrollments))

experiment %>%filter(is.na(Enrollments))

# We dont have enrollment and payment data from Nov third we need to remove it

# Combine and Format Data for ML
control <- control %>% mutate(check = "control")
experiment <- experiment %>% mutate(check = "experiment")

set.seed(123)

data_formatted <- control %>%

# Combine with Experiment data
bind_rows(experiment, .id = "Experiment") %>%
mutate(Experiment = as.numeric(Experiment) - 1)

data_formatted <- data_formatted %>%

# Add row id
mutate(row_id = row_number())

data_formatted <- data_formatted %>%
  # Create a Day of Week feature
  mutate(DOW = str_sub(Date, start = 1, end = 3) %>% 
           factor(levels = c("Sun", "Mon", "Tue", "Wed","Thu", "Fri", "Sat"))
          ) %>% 
  select(-Date, -Payments) %>%
  # Remove missing data
  filter(!is.na(Enrollments))  %>%
  # Shuffle the data
  sample_frac(size = 1)

data_formatted <- data_formatted %>%
  # Reorganize columns
  select(row_id, Enrollments, Experiment, everything())

data_formatted %>% glimpse()

##########################################################################################################################

# Split into Training and Testing Sets
set.seed(123)
split_obj <- data_formatted %>% initial_split(prop = 0.8, strata = "Experiment")
train_col <- training(split_obj)
test_col <- testing(split_obj)
train_col %>% glimpse()
test_col %>% glimpse()

##########################################################################################################################

# Building the model
library(parsnip) # For ML packages: glmnet, xgboost, sparklyr
library(yardstick) # For metrics()

# Model 1: Linear Regression

#Enrollment is our target variable
# Training the linear regression model

linear_regression <- linear_reg("regression") %>%
  set_engine("lm") %>%
  fit(Enrollments ~ ., data = train_col %>% select(-row_id, check))

# Testing the linear regression model

linear_regression %>%
  predict(new_data = test_col) %>%
  bind_cols(test_col %>% select(Enrollments)) %>%
  metrics(truth = Enrollments, estimate = .pred) %>%
  knitr::kable() # used for creating the tables

# Visualizing the data

linear_regression %>%
  # Format Data
  predict(test_col) %>%
  bind_cols(test_col %>% select(Enrollments)) %>%
  mutate(observation = row_number() %>% as.character()) %>%
  gather(key = "key", value = "value", -observation, factor_key = TRUE) %>%
 
  # Visualize
  ggplot(aes(x = observation, y = value, color = key)) + 
  geom_point() + 
  expand_limits(y = 0) + 
  theme_tq() +
  scale_color_tq() + 
  labs(title = "Enrollments: Prediction vs Actual",
  subtitle = "Model 1: Linear Regression (Baseline)")

linear_regression_model_terms <- linear_regression$fit %>%
  tidy() %>%
  arrange(p.value) %>%
  mutate(term = as_factor(term) %>% fct_rev()) 

# knitr::kable() used for pretty tables
linear_regression_model_terms %>% knitr::kable()

# The model is having issue with the observation 1 and 5.

# CLicks, Pageviews and Experiment are strong predictors/features with a p-value less then 0.05 and We note that the
# coefficient of Experiment is -16.9, and because the term is binary (0 or 1) this can be interpreted as
# decreasing Enrollments by -16.9 per day when the Experiment is run

# Featur importance with p-value

linear_regression_model_terms %>%
  ggplot(aes(x = p.value, y = term)) + 
  geom_point(color = "#2C3E50") + 
  geom_vline(xintercept = 0.05, linetype = 2, color = "black") + 
  theme_tq() + 
  labs(title = "Feature Importance",
  subtitle = "Model 01: Linear Regression (Baseline)")

#Our model on average off by 28 enrollments(mean absolute error) and the test set R-squared is low at 0.34.

##########################################################################################################################


# Model 2: Decision Tree
decision_tree <- decision_tree(
  mode = "regression",
  cost_complexity = 0.001,
  tree_depth = 5,
  min_n = 4
) %>%
  set_engine("rpart") %>%
  fit(Enrollments ~ ., data = train_col %>% select(-row_id))

decision_tree %>%
  predict(new_data = test_col) %>%
  bind_cols(test_col %>% select(Enrollments)) %>%
  metrics(truth = Enrollments, estimate = .pred) %>%
  knitr::kable() # used for creating the tables

# The MAE of decision tree is 18 Enrollments per day

# Visualizing the data

decision_tree %>%
  # Format Data
  predict(test_col) %>%
  bind_cols(test_col %>% select(Enrollments)) %>%
  mutate(observation = row_number() %>% as.character()) %>%
  gather(key = "key", value = "value", -observation, factor_key = TRUE) %>%
 
  # Visualize
  ggplot(aes(x = observation, y = value, color = key)) + 
  geom_point()  +
  expand_limits(y = 0)  +
  theme_tq()  +
  scale_color_tq()  +
  labs(title = "Enrollments: Prediction vs Actual",
  subtitle = "Model 2: Decision tree")

# The model is having issue with the observation 1 and 5.

rpart.plot(
  decision_tree$fit,
  roundint = FALSE,
  cex = 0.5,  # Reduce font size further
  fallen.leaves = TRUE,
  extra = 101,
  main = "Model 02: Decision Tree"
  )

# The top features are the most important to the model (“Pageviews” and “Clicks”). 
# The decision tree shows that “Experiment” is involved in the decision rules.
# The rules indicate a when Experiment = 0.5, there is a drop in enrollments.

##########################################################################################################################

# Model 3: XGBoost

set.seed(123)
xgboost <- boost_tree(
  mode = "regression",
  mtry = 100,
  trees = 1000,
  min_n = 8,
  tree_depth = 6,
  learn_rate = 0.2,
  loss_reduction = 0.01,
  sample_size = 1
) %>%
  set_engine("xgboost") %>%
  fit(Enrollments ~ ., data = train_col %>% select(-row_id))

xgboost %>%
  predict(new_data = test_col) %>%
  bind_cols(test_col %>% select(Enrollments)) %>%
  metrics(truth = Enrollments, estimate = .pred)%>%
  knitr::kable() # used for creating the tables

# The MAE of decision tree is about 17.5 Enrollments per day

# Visualizing the model
xgboost %>%
  # Format Data
  predict(test_col) %>%
  bind_cols(test_col %>% select(Enrollments)) %>%
  mutate(observation = row_number() %>% as.character()) %>%
  gather(key = "key", value = "value", -observation, factor_key = TRUE) %>%
 
  # Visualize
  ggplot(aes(x = observation, y = value, color = key)) + 
  geom_point()  +
  expand_limits(y = 0)  +
  theme_tq()  +
  scale_color_tq()  +
  labs(title = "Enrollments: Prediction vs Actual",
       subtitle = "Model 3: XGBoost")

# The model is having issue with the observation 1.

# Feature Importance in XGBoost

xgboost_feature_importance <- xgboost$fit %>%
  xgb.importance(model = .) %>%
  as_tibble()%>%
  mutate(Feature = as_factor(Feature) %>% fct_rev())

xgboost_feature_importance %>% knitr::kable()

# Plot the feature importance

xgboost_feature_importance %>%
  ggplot(aes(x = Gain, y = Feature)) +
  geom_point(color = "#2C3E50") +
  geom_label(aes(label = scales::percent(Gain)), hjust = "inward", color = "#2C3E50")  +
  expand_limits(x = 0) +
  theme_tq() +
  labs(title = "XGBoost Feature Importance") 

# The information gain is 94% from Pageviews and clicks combined. 
# Experiment has about a 6% contribution to information gain, indicating it’s still predictive.

# This tells a story that if Enrollments are critical, Udacity should focus on getting Pageviews.
# Based on XGBoost model Udacity should be focusing on Page Views and secondarily Clicks
# to maintain or increase Enrollments. The features drive the system.

# Outcomes

# If Udacity wants alert people of the time commitment, the additional popup form is expected to decrease the number of enrollments. 
# The negative impact can be seen in the decision tree and in the linear regression model term (-16.9 Enrollments when Experiment = 1).