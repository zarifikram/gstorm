import jax
from craftax.craftax_env import make_craftax_env_from_name

rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
rngs = jax.random.split(_rng, 3)

# Create environment
env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params

# Get an initial state and observation
obs, state = env.reset(rngs[0], env_params)

# Pick random action
action = env.action_space(env_params).sample(rngs[1])

# Step environment
obs, state, reward, done, info = env.step(rngs[2], state, action, env_params)

print(obs.shape)