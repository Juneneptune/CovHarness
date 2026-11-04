# Holdout portfolio turnover

Mean daily turnover of the unconstrained global-minimum-variance portfolio built from each model's forecast over the 500-day holdout. Turnover on a day is the L1 distance between the new weights and the previous day's weights after they drift with returns, and the first day is excluded. These numbers were computed after the holdout from the saved forecasts, and no model was refit. The HARQ-DRD portfolio traded about twice as much as the one built from the LSTM-BEKK trained on realized covariance, while their realized risk was statistically indistinguishable.
