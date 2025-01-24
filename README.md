## Udacity A/B Testing Experiment: README  

### **Overview**  
This project evaluates an A/B test conducted by Udacity to assess the impact of introducing a time-commitment question for users enrolling in free trial courses. The goal was to determine whether setting clearer expectations for time commitment would reduce dropouts during the free trial period and improve student outcomes, without significantly affecting overall enrollments.

### **Experiment Description**  
- **Control Group**: Users could directly enroll in the free trial or access course materials without time commitment information.  
- **Experiment Group**: Users were asked how much time they could dedicate weekly. Those indicating less than 5 hours received a message suggesting accessing course materials for free.  

### **Hypothesis**  
- **Null Hypothesis (H₀)**: There is no difference in enrollments or click-through rates (CTR) between the control and experiment groups.  
- **Alternative Hypothesis (H₁)**: The time-commitment message impacts enrollments or CTR.  

---

### **Project Workflow**

#### **1. Data Preprocessing and EDA**  
- **Missing Data Handling**: Removed rows with missing values in `Enrollments` and `Payments`.  
- **Exploratory Data Analysis**: Visualized distributions and summarized key metrics for `Enrollments` and `Clicks`.  

#### **2. Statistical Inference**
- **Normality Check**: Both control and experiment group `Enrollments` passed the Shapiro-Wilk test (p ≥ 0.05), confirming normality.  
- **Variance Test**: Levene's test indicated equal variances between groups.  
- **t-Test for Means**: No significant difference in `Enrollments` (p > 0.05).  
- **Chi-Square Test for Proportions**: CTRs for control (8.15%) and experiment (8.17%) showed no significant difference (p = 0.8603).  
- **Confidence Intervals**: The 95% confidence intervals included 0 for both mean and proportion differences, reinforcing the conclusion of no significant difference.

#### **3. Machine Learning Analysis**
- **Linear Regression**: Baseline model with an R² of 0.34 and Mean Absolute Error (MAE) of 28 enrollments per day.  
- **Decision Tree**: Improved interpretability with an MAE of 18 enrollments, identifying key features like `Pageviews` and `Clicks`.  
- **XGBoost**: Achieved the best performance with an MAE of 17.5 enrollments and identified `Pageviews` (55% gain) and `Clicks` (38% gain) as the most important drivers.  

#### **4. Power Analysis**
- **t-Test Power**: 394 samples per group are required to detect small effects (Cohen's d = 0.2) at 80% power and 5% significance.  
- **Proportion Test Power**: Current sample size insufficient (12.33% power). Larger samples are needed for reliable conclusions.

---

### **Key Outcomes**
1. **Statistical Results**:
   - No significant difference in enrollments or CTR between groups.
   - The intervention did not harm enrollments or CTR but also did not produce measurable benefits.  

2. **Machine Learning Insights**:
   - `Pageviews` and `Clicks` are the primary drivers of enrollments, contributing 94% of feature importance in the XGBoost model.
   - The experiment feature (`Experiment = 1`) had a small but measurable negative impact, reducing enrollments by ~16.9 per day in the linear regression model.  

---

### **Final Recommendations**
- **Statistical Testing**: Redesign the experiment to target larger or more impactful changes. Ensure sufficient sample size for reliable statistical results.  
- **Focus on Key Metrics**: Optimize `Pageviews` and `Clicks`, as they are critical predictors of enrollments.  

---

This project combines **statistical inference** and **machine learning** to provide a robust analysis of the A/B test, offering actionable insights for Udacity's decision-making.
