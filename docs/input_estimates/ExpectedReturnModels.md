# Expected Returns
As stated in the docs of [PyPortOpt]():
```
Mean-variance optimization requires knowledge of the expected returns. In practice, these are rather difficult to know with any certainty. Thus the best we can do is to come up with estimates, for example by extrapolating historical data. This is the main flaw in mean-variance optimization – the optimization procedure is sound, and provides strong mathematical guarantees, given the correct inputs. 

Is importarnt to take into account that poor estimates of expected returns can do more harm than good. If predicting stock returns were as easy as calculating the mean historical return, we'd all be rich! For most use-cases, I would suggest that you focus your efforts on choosing an appropriate risk model (see risk-models).
````

<br>

This is one of the reasons why I have decided to re-structure the original PyPorfOpt library, and make the source code more modular. Users should be able to come up with their own superior models, add them in the ``pypo/input_estimates`` folder, and feed them into the optimization process. 

<br>

In future updates, documentation for all the expected return models will be detailed here. For the moment, visit [ PyPortOpt source documentation](https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html) for more details.