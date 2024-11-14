# Expected Returns
Mean-variance optimization requires knowledge of the expected returns. In practice, these are rather difficult to know with any certainty. Thus the best we can do is to come up with estimates, for example by extrapolating historical data, This is the main flaw in mean-variance optimization – the optimization procedure is sound, and provides strong mathematical guarantees, given the correct inputs. This is one of the reasons why I have emphasised modularity: users should be able to come up with their own superior models and feed them into the optimizer.

Caution!

Supplying expected returns can do more harm than good. If predicting stock returns were as easy as calculating the mean historical return, we'd all be rich! For most use-cases, I would suggest that you focus your efforts on choosing an appropriate risk model (see :ref:`risk-models`).

As of v0.5.0, you can use :ref:`black-litterman` to significantly improve the quality of your estimate of the expected returns.