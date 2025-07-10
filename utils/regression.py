import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from typing import NamedTuple, List


class FeatureBatch(NamedTuple):
    """Container for regression features."""

    cat: jnp.ndarray
    gzip_bpb: jnp.ndarray
    total_flops: jnp.ndarray
    log_flops: jnp.ndarray
    num_tokens: jnp.ndarray
    num_parameters: jnp.ndarray
    n_cats: int

    def filter_by_category(self, cat_idx: int) -> "FeatureBatch":
        mask = self.cat[:, cat_idx] == 1
        return FeatureBatch(
            cat=self.cat[mask],
            gzip_bpb=self.gzip_bpb[mask],
            total_flops=self.total_flops[mask],
            log_flops=self.log_flops[mask],
            num_tokens=self.num_tokens[mask],
            num_parameters=self.num_parameters[mask],
            n_cats=self.n_cats,
        )


def create_feature_batch(df) -> FeatureBatch:
    cat = jnp.asarray(
        df["category"].astype("category").cat.codes.to_numpy(dtype=float)
    )
    cat = jax.nn.one_hot(cat, num_classes=int(cat.max()) + 1)
    return FeatureBatch(
        cat=cat,
        gzip_bpb=jnp.asarray(df["gzip_bpb"].to_numpy()),
        total_flops=jnp.asarray(df["total_flops"].to_numpy()),
        log_flops=jnp.asarray(df["log_flops"].to_numpy()),
        num_tokens=jnp.asarray(df["num_tokens"].to_numpy()),
        num_parameters=jnp.asarray(df["num_parameters"].to_numpy()),
        n_cats=cat.shape[-1],
    )


def mse_factory():
    @jax.jit
    def mse(preds, truth, reduction=jnp.sum):
        return reduction((truth - preds) ** 2)

    return mse


def huber_factory():
    @jax.jit
    def huber(preds, truth, reduction=jnp.sum, delta=1e-3):
        residuals = truth - preds
        cond = jnp.abs(residuals) <= delta
        loss = jnp.where(
            cond, 0.5 * residuals**2, delta * (jnp.abs(residuals) - 0.5 * delta)
        )
        return reduction(loss)

    return huber


def power_law_compute_factory(n_cats: int, axis_of_interest: str = "log_flops"):
    try:
        field_idx = FeatureBatch._fields.index(axis_of_interest)
    except ValueError as e:
        raise ValueError(f"{axis_of_interest!r} is not a field of FeatureBatch") from e

    @jax.jit
    def predict(theta, batch):
        a = theta[0:n_cats]
        b = theta[n_cats : 2 * n_cats]
        x = batch[field_idx]
        a_cat = batch.cat @ a
        b_cat = batch.cat @ b
        return 1 - (a_cat * (x**b_cat))

    def init():
        return np.concatenate([
            np.zeros(n_cats, dtype=float),
            np.zeros(n_cats, dtype=float),
        ]).tolist()

    return predict, init


def gzip_adjusted_factory(n_cats: int, axis_of_interest: str = "log_flops"):
    try:
        field_idx = FeatureBatch._fields.index(axis_of_interest)
    except ValueError as e:
        raise ValueError(f"{axis_of_interest!r} is not a field of FeatureBatch") from e

    @jax.jit
    def predict(theta, batch):
        a = theta[0:n_cats]
        b = theta[n_cats : 2 * n_cats]
        gzip_slope = theta[-1]
        x = batch[field_idx]
        a_cat = batch.cat @ a
        b_cat = batch.cat @ b
        return 1 - (a_cat * (x ** (b_cat + (gzip_slope * batch.gzip_bpb))))

    def init():
        return np.concatenate([
            np.zeros(n_cats, dtype=float),
            np.zeros(n_cats, dtype=float),
            np.zeros(1, dtype=float),
        ]).tolist()

    return predict, init


def chinchilla_compute_factory(n_cats: int):
    @jax.jit
    def predict(theta, batch):
        k_D = theta[0:n_cats]
        k_P = theta[n_cats:2*n_cats]
        e_D = theta[2*n_cats:3*n_cats]
        e_P = theta[3*n_cats:4*n_cats]

        kD_cat = batch.cat @ k_D
        kP_cat = batch.cat @ k_P
        eD_cat = batch.cat @ e_D
        eP_cat = batch.cat @ e_P

        D_opt = kD_cat * (jnp.log10(batch.num_tokens) ** eD_cat)
        P_opt = kP_cat * (jnp.log10(batch.num_parameters) ** eP_cat)
        return 1 - (D_opt + P_opt)

    def init():
        return np.concatenate([
            np.zeros(n_cats, dtype=float),
            np.zeros(n_cats, dtype=float),
            np.zeros(n_cats, dtype=float),
            np.zeros(n_cats, dtype=float),
        ]).tolist()

    return predict, init


def chinchilla_gzip_compute_factory(n_cats: int):
    @jax.jit
    def predict(theta, batch):
        k_D = theta[0:n_cats]
        k_P = theta[n_cats:2*n_cats]
        e_D = theta[2*n_cats:3*n_cats]
        e_P = theta[3*n_cats:4*n_cats]
        gzip_slope = theta[-1]

        kD_cat = batch.cat @ k_D
        kP_cat = batch.cat @ k_P
        eD_cat = batch.cat @ e_D
        eP_cat = batch.cat @ e_P

        D_opt = kD_cat * (batch.num_tokens ** (eD_cat + (gzip_slope * batch.gzip_bpb)))
        P_opt = kP_cat * (batch.num_parameters ** eP_cat)
        return 1 - (D_opt + P_opt)

    def init():
        return np.concatenate([
            np.zeros(n_cats, dtype=float),
            np.zeros(n_cats, dtype=float),
            np.zeros(n_cats, dtype=float),
            np.zeros(n_cats, dtype=float),
            np.zeros(1, dtype=float),
        ]).tolist()

    return predict, init


def fit_scaling_law(
    final_df,
    options,
    ratios,
    explode_metrics,
    *,
    loss_factory=mse_factory,
    predict_factory=power_law_compute_factory,
    domain_folder="eval/ICE/",
    baseline="USA",
):
    combined_df = explode_metrics(final_df, options, ratios, domain_folder=domain_folder, baseline=baseline)
    loss_fn = loss_factory()

    feature_batch = create_feature_batch(combined_df)
    n_cats = feature_batch.n_cats
    y = jnp.array(combined_df["relative_bpb"].astype(float))

    predict, init = predict_factory(n_cats=n_cats)

    def opt_fn(theta):
        return loss_fn(predict(theta, feature_batch), y)

    grad_fn = jax.grad(opt_fn)

    theta0 = init()

    res = minimize(
        opt_fn,
        theta0,
        jac=grad_fn,
        method="L-BFGS-B",
        options=dict(
            maxiter=500_000,
            maxfun=5000_000,
            maxcor=1000,
            ftol=1e-12,
            gtol=1e-12,
            maxls=200,
            disp=True,
        ),
    )
    if not res.success:
        raise RuntimeError(f"Fit did not converge: {res.message}")
    return res


def pretty_print_coefficients(theta_opt, combined_df):
    cats = sorted(combined_df["category"].unique())
    n_cats = len(cats)
    lets = ["a", "b", "c", "d", "e"]
    cat_coefs = {}
    for i in range(len(theta_opt) // n_cats):
        let = lets[i]
        vals = theta_opt[i * n_cats : (i + 1) * n_cats]
        cat_coefs[let] = vals
    gzip_slope = theta_opt[-1]
    coef_df = pd.DataFrame({"category": cats, **cat_coefs})
    loss = theta_opt[-1] if len(theta_opt) % n_cats else None
    return coef_df, gzip_slope, loss


def bootstrap_scaling_law(
    final_df,
    options,
    ratios,
    explode_metrics,
    *,
    loss_factory,
    predict_factory,
    n_boot: int = 1000,
    seed: int | None = None,
    block: bool = True,
    domain_folder="eval/ICE/",
    baseline="USA",
):
    base_combined = explode_metrics(final_df, options, ratios, domain_folder=domain_folder, baseline=baseline)
    cats = sorted(base_combined["category"].unique())
    n_cats = len(cats)

    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_boot):
        if not block:
            sample = final_df.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 2**32)))
        else:
            grouped = list(final_df.groupby(["width", "depth"], dropna=False))
            blocks: List[pd.DataFrame] = [g for _, g in grouped]
            n_blocks = len(blocks)
            if n_blocks == 0:
                raise ValueError("No blocks found: ensure 'width' and 'depth' columns exist and are not all NaN.")
            sampled_idx = rng.integers(0, n_blocks, size=n_blocks)
            resampled = pd.concat([blocks[i] for i in sampled_idx], ignore_index=True)
            sample = resampled.reset_index(drop=True)
        res = fit_scaling_law(
            sample,
            options,
            ratios,
            explode_metrics,
            loss_factory=loss_factory,
            predict_factory=predict_factory,
            domain_folder=domain_folder,
            baseline=baseline,
        )
        theta = res.x
        vals = []
        for i in range(len(theta) // n_cats):
            vals.append(theta[i * n_cats : (i + 1) * n_cats])
        for i in range(1, (len(theta) % n_cats) + 1):
            vals.append(theta[-i])
        lets = ["a", "b", "c", "d", "e"]
        row = {}
        for i in range(len(theta) // n_cats):
            let = lets[i]
            for c, v in zip(cats, vals[i]):
                row[f"{let}[{c}]"] = v
        for i in range(1, (len(theta) % n_cats) + 1):
            let = lets[i]
            row[f"{let}"] = vals[-i]
        rows.append(row)
    boot_df = pd.DataFrame(rows)
    summary = boot_df.describe(percentiles=[0.025, 0.5, 0.975])
    return boot_df, summary


def fit_and_bootstrap(
    final_df,
    options,
    ratios,
    explode_metrics,
    *,
    loss_factory,
    factory_fn,
    n_boot,
    seed,
    domain_folder="eval/ICE/",
    baseline="USA",
):
    final_df = final_df[~final_df["Run Name"].str.contains("OLMo", na=False)].copy()
    res = fit_scaling_law(
        final_df,
        options,
        ratios,
        explode_metrics,
        loss_factory=loss_factory,
        predict_factory=factory_fn,
        domain_folder=domain_folder,
        baseline=baseline,
    )
    if not res.success:
        raise RuntimeError(res.message)
    theta_opt = jnp.asarray(res.x)
    boot_df, _ = bootstrap_scaling_law(
        final_df,
        options,
        ratios,
        explode_metrics,
        loss_factory=loss_factory,
        predict_factory=factory_fn,
        n_boot=n_boot,
        seed=seed,
        domain_folder=domain_folder,
        baseline=baseline,
    )
    return theta_opt, boot_df
