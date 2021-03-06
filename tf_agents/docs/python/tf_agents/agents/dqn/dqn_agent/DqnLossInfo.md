<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.dqn.dqn_agent.DqnLossInfo" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="td_loss"/>
<meta itemprop="property" content="td_error"/>
</div>

# tf_agents.agents.dqn.dqn_agent.DqnLossInfo

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/agents/dqn/dqn_agent.py">View
source</a>

## Class `DqnLossInfo`

DqnLossInfo is stored in the `extras` field of the LossInfo instance.



<!-- Placeholder for "Used in" -->

Both `td_loss` and `td_error` have a validity mask applied to ensure that
no loss or error is calculated for episode boundaries.

td_loss: The **weighted** TD loss (depends on choice of loss metric and
  any weights passed to the DQN loss function.
td_error: The **unweighted** TD errors, which are just calculated as:

  ```
  td_error = td_targets - q_values
  ```

  These can be used to update Prioritized Replay Buffer priorities.

  Note that, unlike `td_loss`, `td_error` may contain a time dimension when
  training with RNN mode.  For `td_loss`, this axis is averaged out.

## Properties

<h3 id="td_loss"><code>td_loss</code></h3>

<h3 id="td_error"><code>td_error</code></h3>
