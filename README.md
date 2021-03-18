![tests](https://github.com/jennhsiao/ideotype/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/jennhsiao/ideotype/branch/main/graph/badge.svg?token=O47AEZGM6U)](https://codecov.io/gh/jennhsiao/ideotype)

# Ideotype

## Motivation
Over the next three decades rising population and changing dietary preferences are expected to increase food demand by 25–75%. At the same time climate is also changing — with potentially drastic impacts on food production. Changes in *__crop characteristics__* and *__management practices__* have the potential to mitigate some of the expected decrease in yield due to a changing climate. However, identifying optimum plant traits and management options for different growing regions through traditional breeding practices and agronomic field trials is time and resource intensive. Mechanistic crop simulation models can serve as powerful tools to help synthesize cropping information, set breeding targets, and develop adaptation strategies to sustain food production. 

In this project, we use a mechanistic crop simulation model (MAIZSIM) to explore how different crop traits and agricultural management options affect maize growth and yield, with the hope to identify ideal trait and management combinations, known as *__ideotypes__*, that maximize yield and minimize risk for different agro-climate regions in the US.

## Approach
### Identify key crop trait and management
*Simulation sites:*
We identified various sites across the US maize growing region that have several years of hourly growing season climate information available as environmental drivers for the MAIZSIM model. We further filtered these sites to include locations that have a maize planting area greater than 50,000 acers, and include only sites that are majority rain-fed. 

*Parameter selection:*
We selected several key model parameters that describe physiological, morphology, and phenological traits within a maize plant, as well as management practices such as planting dates and planting density. We set biologically reasonable ranges for the parameters and sampled within the boundaries following the Latin hypercube sampling method to create ensemble simulations of 1000 different parameter combinations.

*Sensitivity analysis:*
We use a sensitivity analysis framework to identify critical parameters in the MAIZSIM model through two complementary methods - by calculating the partial correlation coefficient (PCC), and by performing the Fourier amplitude sensitivity test (FAST). 

### Identify trait-management combinations with high yield and low volatility
By calculating mean yield over years normalized by yield variance over years due to natural climate variability, we identify key trait-management combinations that lead to high yield and low yield volatility for different maize-growing resgions within the US.

### Ideotype under current vs. future climate
We ran the MAIZSIM model with past climate as well as idealized projections of 2050 future climate to identify how ideotypes may shift with a changing climate. 

## Simulation sites
<p align="left">
  <img src="docs/sites_obsyield.png" width ="500">
</p>