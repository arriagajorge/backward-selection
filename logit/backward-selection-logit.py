# import stasmodels.api as sm

# Function for performing feature selection based on p-values
def backward_feature_elimination(X, y, significance_level=0.05):
  """
    Performs backward feature elimination using logistic regression.

    This function takes feature matrix X and target vector y and iteratively eliminates
    features from X based on their p-values from logistic regression. The process continues
    until no features with p-values above the specified significance level remain.

    Parameters:
        X (DataFrame): The feature matrix.
        y (Series): The target vector.
        significance_level (float, optional): The significance level for feature elimination.
            Defaults to 0.05.

    Returns:
        DataFrame: The feature matrix with selected features after elimination.
    """
  while len(X.columns) > 0:
      # Fit logistic regression model
      model = sm.Logit(y, X).fit(disp=0)
      # Get p-values
      p_values = model.pvalues
      max_p_value = p_values.max()
      # Eliminate feature if its p-value is above significance level
      if max_p_value > significance_level:
          excluded_feature = p_values.idxmax()
          X = X.drop(excluded_feature, axis=1)
      else:
          break
  return X
