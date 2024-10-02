import jax
import distrax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.flatten_util import ravel_pytree
from functools import partial 


HalfNormal = distrax.as_distribution(tfd.HalfNormal(1.0))


def inference_loop(rng_key, kernel, initial_state, num_samples):
    """
    Default Blackjax inference loop
    """
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def get_num_layers(params):
    """
    Obtain the number of layers given a set
    of params of a Flax model
    """
    num_layers = len((jax.tree_flatten(params, lambda x: "params" not in x))[0][0])
    return num_layers


def init_params(key, model, batch, n_tasks):
    """
    Initialise the parameters of the Hierarchical Bayesian Neural Network
    """
    key_tasks_init, key_params_init, key_sigma_init = jax.random.split(key, 3)
    keys_tasks_init = jax.random.split(key_tasks_init, n_tasks)

    params_tasks = jax.vmap(model.init, (0, None))(keys_tasks_init, batch)
    params_shared = model.init(key_params_init, batch)
    n_layers = get_num_layers(params_shared)
    params_sigma = HalfNormal.sample(seed=key_sigma_init, sample_shape=(n_layers, 2))

    model_treedef = jax.tree_structure(params_shared)

    params_init = {
        "task": params_tasks,
        "noise": params_sigma,
        "shared": params_shared,
    }

    return params_init, model_treedef


def build_sigma_tree(params, treedef):
    """
    Build a tree of parameters that satisfy
    the HBNN conditions and is flax-compatible
    """
    
    params_sigma_tree = jnp.expand_dims(params, 0) # (1, n_layers, 2)
    params_sigma_tree = jax.tree_util.build_tree(treedef, params_sigma_tree)

    return params_sigma_tree


@partial(jax.vmap, in_axes=(0, 0, 0, None))
def vmap_log_likelihood(params, X, y, model):
    """
    Hierarchical Bayesian neural network log-likelihood

    Parameters
    ----------
    params: pytree
    X: jnp.array(C, N, ...)
        Collection of observations
    y: jnp.array(C, N, ...)
        Collection of outputs
    """
    num_classes = y.shape[-1]
    logits = model.apply(params, X)
    log_likelihood = distrax.Multinomial(num_classes, logits=logits).log_prob(y).sum()
    return log_likelihood


def hierarchical_log_joint(params, X, y, model_treedef, model):
    """
    Hierarchical Bayesian neural network log-joint. The model
    assumes

    w^t_{ijl} = \mu_{ijl} + \varepsilon^t_{ijl} \sigma_l

    Parameters
    ----------
    parameters: flax.FrozenDict
    X: jnp.array(C, N, ...)
        Collection of observations
    y: jnp.array(C, N, ...)
        Collection of outputs

    Returns
    -------
    float: float
        Log-joint value
    """

    shared_tree = params["shared"]
    task_tree = params["task"]
    sigma_tree = build_sigma_tree(params["noise"], model_treedef)

    model_params = jax.tree.map(lambda mu, task, sigma: mu + sigma * task, shared_tree, task_tree, sigma_tree)

    # ** log-likelihood for all tasks **
    log_likelihood_collection = vmap_log_likelihood(model_params, X, y, model).sum()

    # ** log priors **
    # Tasks priors
    log_task_prior = jax.tree.map(distrax.Normal(0.0, 1.0).log_prob, task_tree)
    log_task_prior = ravel_pytree(log_task_prior)[0].sum()
    # Global prior
    log_shared_prior = jax.tree.map(distrax.Normal(0.0, 1.0).log_prob, shared_tree)
    log_shared_prior = ravel_pytree(log_shared_prior)[0].sum()
    # Sigma-layered prior
    log_sigma_prior = HalfNormal.log_prob(params["noise"]).sum()

    log_prior = log_task_prior + log_shared_prior + log_sigma_prior
    log_joint = log_likelihood_collection + log_prior
    return log_joint


def build_model_params(params, treedef):
    """
    Aggregate dictionary of parameters into a
    Flax-valid set of parameters.

    Parameters
    ----------
    params: dict
        Colletion of parameters for the HBNN as
        specified by `init_hbnn_params`

    Returns
    -------
    flax.FrozenDict: flax-compatible set of weights
    """
    shared_tree = params["shared"]
    task_tree = params["task"]
    sigma_tree = build_sigma_tree(params["noise"], treedef)

    model_params = jax.tree.map(lambda mu, task, sigma: mu + sigma * task,
                                shared_tree, task_tree, sigma_tree)
    return model_params


@partial(jax.vmap, in_axes=(0, None, None, None))
def vmap_eval_tasks(params, X, model, treedef):
    model_params = build_model_params(params, treedef)
    return jax.vmap(model.apply, 0)(model_params, X)


@partial(jax.vmap, in_axes=(None, 2, None,  None), out_axes=2)
@partial(jax.vmap, in_axes=(None, 1, None,  None), out_axes=1)
@partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0)
def eval_tasks_grid(params, X, model, treedef):
    """
    Evaluate the HBNN on a 2d grid of the input space over
    multiple samples.
    """
    model_params = build_model_params(params, treedef)
    return jax.vmap(model.apply, (0, None))(model_params, X)
