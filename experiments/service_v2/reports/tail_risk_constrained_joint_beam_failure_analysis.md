# Tail Risk Constrained Joint Beam Failure Analysis

## Status

No current failure: `tail_risk_constrained_joint_beam` passed eval `5`, `10`, and `20` in default anchor-locked mode.

## Guarded Failure Recap

`joint_accept_route_beam_guarded` still fails because it can improve local service counts while moving lateness into the tail:

- eval 5 max lateness: `39.300615` vs raw `31.124878`
- eval 10 max lateness: `39.918412` vs raw `33.856089`
- eval 20 max lateness: `48.364049` vs raw `35.936546`

The pattern confirms the original diagnosis: guarded scoring is still willing to buy visible service gains with a few very late orders.

## Residual Risk

The passing tail-risk method is conservative. It currently proves safety by anchoring to raw baseline rather than by finding many safe improvements. A future deviation mode should remain gated behind the same strict checks.

