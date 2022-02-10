## Cartpole-v1
**Action**: Push cart to left or right.

**Observations**: Cart's position, cart's velocity, pole's angle and pole's angular velocity.

**Reward**: Reward is 1 for every step taken.

**Initial Conditions**: All observations are assigned a uniform random value in range `[-0.05,0.05]`.

**Done**: Becomes true when pole is upright with zero velocity. Otherwise the episode terminates when one of the following occurs:

1. Pole's Angle is more than ±12°
2. Cart's Position is more than ±2.4 (center of the cart reaches the edge of the display)
3. Episode's length is greater than 500

## Pendulum-v1
**Action**: Torque applied to pendulum.

**Observations**: Position of pendulum (x,y cordinates calculated from pendulum angle) and pendulum's angular velocity.

**Reward**: Reward is calulated as:
```
Reward = -(theta^2 + 0.1*theta_dt^2 + 0.001*torque^2)
```
where `theta` is the pendulum's angle normalized between `[-pi, pi]`. Pendulum get a maximum reward of zero when its upright with zero velocity and no torque being applied.

**Initial Conditions**: Random angle in range `[-pi,pi]` and random angular velocity in range `[-1,1]`.

**Done**: Becomes true when reward becomes equal to zero. Otherwise the episode runs for `200` steps and terminates.
