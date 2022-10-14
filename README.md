## Functionality 

Detect anomaly for specified `gid` at one hour, using a particular `model`.

## Command
To detect anomalies from *starttime* to *endtime* on flow meter *gid* (on significance level 0.002), using:
- a) model *lstm_alpha_prob* \
`py main.py --gid [GID] --model lstm_alpha_prob --starttime ["YYYY-MM-DD HH:mm:SS"] --endtime ["YYY-MM-DD HH:mm:SS"] --alpha 0.002`

- b) model *lstm_alpha_quantile* \
`py main.py --gid [GID] --model lstm_alpha_quantile --starttime ["YYYY-MM-DD HH:mm:SS"] --endtime ["YYY-MM-DD HH:mm:SS"] --alpha 0.002`

_Note_
- `py` is required for `lstm_alpha_prob / lstm_alpha_quantile` models [due to tensorflow].
- `alpha` (the significance level) is by default `0.002` (in case it's not specified by the user), but can be set to any value specified in the python script.

*The available models are*: 

  * LSTM_err -- [model NAME]
    - lstm_alpha_prob -- [model TYPE] (*alpha* required)
    - lstm_alpha_quantile -- [model TYPE] (*alpha* required)

  * SM_stdev_4 -- [model NAME]
    - simplemath -- [model TYPE] (*alpha* NOT required)


# --------------------------------------------------------------------------------- 
# _How it works._

This LSTM-based anomaly detection flags as abnormal (or anomaly) data points which have the corresponding prediction error too unlikely to be due to chance (somewhat like the idea of p-value, but with a single observation).  

We have 2 very simialr models which only differ in the way they compute how 'extreme' a prediction error is (i.e. the probability of observing such a prediction error, or an even more extreme one.):

 - lstm_alpha_prob : this model is computing the *exact* probability of observing such a prediction error, or an even more extreme one (based on the empirical rule that the absolute prediction errors follow an exponential distribution with rate equal to the inverse of the sample size.)

 - lstm_alpha_quantile : this model is computing the  probability of observing such a prediction error, or an even more extreme one based on the quantile that
