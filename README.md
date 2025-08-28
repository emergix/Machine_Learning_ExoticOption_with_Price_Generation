# Machine_Learning_YetiPhenics_with_Generation_Price
Execute machine learning on Generated Prices
## Preface: Calibrating a Neural Pricer for Exotic Options in the Black-Scholes Model
Welcome to this blog series on advanced option pricing using machine learning! In the world of quantitative finance, pricing exotic options—such as barrier options, Asians, or multi-asset structures like Worst-of-N—can be computationally intensive, especially under stochastic models. Traditional methods like Monte Carlo simulations or finite difference schemes work well but scale poorly for real-time calibration or high-dimensional problems.
Here, we explore a modern approach: calibrating a neural network-based pricer for an exotic option (specifically, a "Worst-of-3" option) within the Black-Scholes framework. This involves training a deep learning model to approximate prices directly from input parameters, bypassing repeated simulations during calibration. The result? Faster inference times, making it ideal for inverse problems like implied volatility surface fitting or model parameter calibration.
This preface outlines our methodology at a high level, drawing from the provided Jupyter notebooks (YetiPhen_PrepareData.ipynb for data preparation and Yetiphoenix_BS_Learning.ipynb for model training). We'll dive deeper into code, results, and extensions in subsequent posts.
## Step 1: Data Preparation
The foundation of any ML-based pricer is high-quality training data. We generate or load a large dataset of option prices under the Black-Scholes model, varying key parameters to cover a realistic parameter space.

Model Assumptions: We assume a multi-asset Black-Scholes world with constant volatility, risk-free rate, and correlations between assets. For a Worst-of-3 option, the payoff depends on the minimum performance among three underlying assets (e.g., payout if the worst performer is above a strike).
Parameter Sampling:

Sample inputs like spot prices (S0 for each asset), strikes (K), maturities (T), volatilities (σ), correlations (ρ), and risk-free rates (r).
In the notebook, we load pre-generated data from CSV files (e.g., via Monte Carlo or closed-form approximations where possible).
Combine multiple files into a unified dataset (Xtotal), filtering for valid entries (e.g., positive volatilities).
Example parameters: Alpha (initial vol), beta (vol dynamics), rho (correlation), maturity, strike, etc. (adapted from an extended SABR-like setup, but simplified for Black-Scholes).


Data Splitting:

Split into training (90%) and validation (10%) sets using scikit-learn's train_test_split.
Save as CSVs for reproducibility (e.g., "LearningBaseFileB.CSV" for training).



This step ensures our dataset is diverse and representative, typically comprising millions of samples to capture edge cases like deep in/out-of-the-money scenarios.
## Step 2: Model Architecture
We use a neural network to learn the mapping from input parameters to option prices. The architecture is designed for efficiency and accuracy in regression tasks.

Input Features: A vector of parameters (e.g., 13 dimensions in the data prep notebook: alpha0, beta, beta2, d, gamma, nu, omega, lambda, rho, maturity, strike, option price, vol).
Network Design:

A convolutional neural network (CNN) variant with minmax scaling, leaky ReLU activations (leak factor ~0.2), and dropout (10%) for regularization.
Multiple dense layers (e.g., 11-13 layers with 8-32 neurons each), using 'tanh' activations internally and 'linear' for output.
Custom metrics: Quadratic error and "anti-quadratic" (possibly a negated or inverse metric for monitoring).
Optimizer: Adamax; Batch size: Large (e.g., 32k-64k) for efficiency on big data.
Early stopping via a custom callback monitoring loss improvement (e.g., halt if no progress beyond a fraction like 0.999).


Why Neural Networks? They excel at approximating complex, non-linear functions like option payoffs, often outperforming traditional interpolators in speed for calibration loops.

## Step 3: Training and Calibration
Training turns the network into a surrogate pricer, which we then "calibrate" by fine-tuning on market data or validating against benchmarks.
Training Loop:

Fit the model for 30 epochs (adjustable) with validation split.
Monitor loss (MSE-like) and custom metrics.
Example from notebook: Loss improves from ~-185 to ~-2024 over epochs, with validation loss reaching ~-3022, indicating good generalization.

## Yeti Phoenix Option Payoff Description

The Yeti Phoenix option is an exotic option that calculates its payoff based on the performance of multiple underlying assets over a series of observation dates. The payoff is determined by evaluating the worst-performing asset at each date and applying specific conditions related to predefined barriers and coupons. Below is a detailed description of how the payoff is computed:

Inputs

Trajec: A 2D array where rows represent observation dates and columns represent different underlying assets. Each element is the value of an asset at a specific date.
Bonus: A fixed amount added to the payoff at each observation date.

YetiBarrier: A threshold level for the worst-performing asset to determine eligibility for the Yeti coupon.

YetiCoupon: An additional amount added to the payoff if the worst-performing asset is at or below the Yeti barrier.

PhoenixBarrier: A threshold level for the worst-performing asset to trigger an early termination with a Phoenix coupon.

PhoenixCoupon: An amount added to the payoff if the worst-performing asset is at or above the Phoenix barrier, after which the payoff calculation terminates.

PDIbarrier: A threshold level for the worst-performing asset to activate the Put Down and In (PDI) component.

PDIGearing: A multiplier applied to the PDI payoff.

PDIStrike: The strike price used in the PDI payoff calculation.

PDItype: A parameter indicating the type of PDI option (e.g., 1 for call, -1 for put).

Forwards (output): An array storing the worst-performing asset's value at each observation date.

Payoff Calculation


Initialization:





The payoff starts at zero.



A flag (PDIFlag) is initialized to track whether the PDI condition is met.



An array (forwards) is created to store the worst-performing asset's value at each date.



Worst-Performing Asset:





For each observation date (starting from the second date, index 1), the code identifies the worst-performing asset by finding the minimum value across all assets at that date.



This minimum value is stored in the forwards array for that date.



Payoff Components:





For each observation date:





Bonus: The fixed Bonus amount is added to the payoff.



Yeti Coupon: If the worst-performing asset's value is at or below the YetiBarrier, the YetiCoupon is added to the payoff.



Phoenix Coupon: If the worst-performing asset's value is at or above the PhoenixBarrier, the PhoenixCoupon is added, and the payoff calculation stops (no further dates are evaluated).



PDI Flag: If the worst-performing asset's value is at or below the PDIbarrier, the PDIFlag is set to 1, indicating the PDI component is active.



PDI Payoff:





At the final date, if the PDIFlag is 1 (i.e., the PDI condition was triggered), an additional payoff is calculated as:





PDIGearing * max(0, PDItype * (final worst-performing asset value - PDIStrike)).



This represents a geared put or call option payoff, depending on PDItype, based on the final worst-performing asset value relative to the PDIStrike.



Final Payoff:





The total payoff is the sum of all bonuses, any Yeti coupons, any Phoenix coupon (if triggered), and the PDI payoff (if applicable).

Summary

The Yeti Phoenix option combines a fixed bonus at each observation date with conditional coupons based on the worst-performing asset's value relative to the Yeti and Phoenix barriers. It also includes a Put Down and In component that activates if the worst-performing asset falls below the PDI barrier, contributing a geared payoff based on the final asset value. The calculation stops early if the Phoenix barrier is met, making it a path-dependent exotic option with multiple payoff triggers.
