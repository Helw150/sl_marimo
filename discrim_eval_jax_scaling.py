import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full", app_title="(No-SFT) Discrim Eval Scaling")


@app.cell
def _(mo):
    from utils.theme import apply_plotly_theme
    return (apply_plotly_theme,)


@app.cell
def _():
    from utils.wandb_utils import login_and_get_runs

    runs = login_and_get_runs(
        "marin-community/marin",
        {"displayName": {"$regex": "marin-us-central2.*discrim.*sbf.*"}, "state": "finished"},
    )

    print(list(runs))
    print(runs[0].summary.keys())
    options = [key for key in runs[0].summary.keys() if "lm_eval/" in key and ("bias" in key) and "averages" not in key]
    print(options)
    return options, runs


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Measuring Dialect Loss on the International Corpus of English across Scales

    All subsequent experiments are the product of Scaling Laws fit based on 149 Checkpoints from the University of Maryland [Gemstones Suite](https://arxiv.org/abs/2502.06857). These are Warmup-Stable-Decay Checkpoints for 22 models from 50M to 2B parameters, spanning 11 widths from 256 to 3072, 18 depths from 3 to 80, and 10 data volumes from 11B to 100B tokens. Importantly, each of these model runs are trained on Dolma 1.7 without intervention making them a model of "Naive" scaling with no dialect specific treatment or domain robustness treatement.

    We then evaluate these models as language models for the 12 samples of World English from the [International Corpus of English](https://en.wikipedia.org/wiki/International_Corpus_of_English) which are openly accessible. We keep only the written sub-corpora to avoid measuring transcription artifacts and remove all markup tags from the original data including any tags that indicate edits by the writer. This is only an evaluation of the **test** loss on these dialects, with the underlying training data held the same.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    from utils.flops import flops_per_token_gqa

    return flops_per_token_gqa, mo


@app.cell
def _(options, runs):
    import pandas as pd

    data = []

    for run in runs:
        name = run.displayName
        bpb_metrics = {}
        for metric_name in options:
            bpb_metric = run.summary.get(
                metric_name, None
            )
            bpb_metrics[metric_name] = bpb_metric
        if "OLMo" in name:
            if "tokenizer" not in name or "stage1" not in name or ("3896" not in name and "7B" in name):
                continue
            else:
                name = name.replace("tokenizer-fix-", "")
        data.append({"Run Name": name, **bpb_metrics})

    df = pd.DataFrame(data)
    return df, pd


@app.cell
def _(df, pd):
    import re


    def parse_model_name(run_name):
        match = re.search(r"-(\d+)x(\d+)_", run_name)
        if match:
            width = int(match.group(1))
            depth = int(match.group(2))
        else:
            width = None
            depth = None
        train_length_match = re.search(r"step_\d+_cooldown_(\d+)", run_name)
        if train_length_match:
            train_length = int(train_length_match.group(1))
        else:
            if "OLMo" in run_name:
                print(run_name)
                if "7B" in run_name:
                    width = 4096
                    depth = 32
                    tokens_per_step = 2e9 / 477.0
                    if "stage1" in run_name:
                        train_length = 3896e9 / tokens_per_step
                elif "13B" in run_name:
                    width = 5120
                    depth = 40
                    tokens_per_step = 2e9 / 477.0
                    if "stage1" in run_name:
                        train_length = 5000e9 / tokens_per_step
                    print(run_name)
            else:
                raise ValueError(f"Run Name {run_name} is not parseable")
        if not width:
            print(run_name, width, depth, train_length)
        return width, depth, train_length


    width_df = df
    width_df[["width", "depth", "train_length"]] = df["Run Name"].apply(
        lambda x: pd.Series(parse_model_name(x))
    )
    width_df = width_df.drop_duplicates(subset=["width", "depth", "train_length"])
    return (width_df,)


@app.cell
def _(df, flops_per_token_gqa, width_df):
    import numpy as np
    # Function to calculate total FLOPs for a run
    def calculate_total_flops(row):
        if (
            row["width"] is None
            or row["depth"] is None
            or row["train_length"] is None
        ):
            return None

        # Calculate FLOPs per token using the provided function
        if "OLMo" in row["Run Name"]:
            flops_per_token = flops_per_token_gqa(row["width"], row["depth"], vocab_size = 100278, queries_per_group=1, seq_len=4096)
        else:  
            flops_per_token = flops_per_token_gqa(row["width"], row["depth"])

        # Assuming train_length is in steps, convert steps to tokens
        # 477 steps = 2e9 tokens
        tokens_per_step = 2e9 / 477.0
        total_tokens = row["train_length"] * tokens_per_step

        # Total FLOPs = FLOPs per token * Total tokens
        total_flops = flops_per_token * total_tokens

        return total_flops


    # Apply the function to each row to get the total FLOPs
    flop_df = width_df
    flop_df["total_flops"] = df.apply(calculate_total_flops, axis=1)
    return flop_df, np


@app.cell
def _():
    from huggingface_hub import HfApi


    def get_model_num_parameters(model_id: str):
        """Return the number of parameters for a given model on the Hugging Face Hub."""
        api = HfApi()
        model_info = api.model_info(model_id)
        return model_info.safetensors.total
    return (get_model_num_parameters,)


@app.cell
def _(
    apply_plotly_theme,
    flop_df,
    flops_per_token_gqa,
    get_model_num_parameters,
    np,
    pd,
):
    def map_run_name_to_model_id(run_name):
        # Example: "ppl-eval-tomg-group-umd--Gemstone-256x23_cooldown--step_00002385_cooldown_00002622-7cba50"
        # maps to "tomg-group-umd/Gemstone-256x23_cooldown@step_00002385_cooldown_00002622"
        #MD3-SL-tomg-group-umd-Gemstone-256x23_cooldown@step_00002385_cooldown_00002622

        # Remove the "ppl-eval-" prefix
        model_id = run_name.replace("Domain-Scaling-Laws-", "")

        # Replace "--" with "/" and remove the last part (hash)
        model_id = "tomg-group-umd/" + model_id.split("umd--")[-1].split("_lmeval")[0].replace("--", "@")

        # Handle the step_ cooldown part
        #model_id = model_id.replace("--step_", "@step_")
        model_id = model_id.split("@")[0]

        return model_id


    final_df = flop_df
    # Apply the mapping function to create a new column 'model_id'
    final_df["model_id"] = flop_df["Run Name"].apply(map_run_name_to_model_id)


    # Function to get the number of parameters for each model ID
    def get_parameters_for_model(model_id):
        try:
            return get_model_num_parameters(model_id)
        except Exception as e:
            print(f"Could not get parameters for {model_id}: {e}")
            return None


    # Apply the function to the 'model_id' column to get the number of parameters
    final_df["num_parameters"] = flop_df["model_id"].apply(get_parameters_for_model)
    final_df = final_df.dropna(subset=["total_flops"])
    final_df["total_flops"] = pd.to_numeric(final_df["total_flops"])

    template = apply_plotly_theme()

    # Compute FLOPs per token
    final_df["flops_per_token"] = final_df.apply(
        lambda row: flops_per_token_gqa(row["width"], row["depth"])
        if pd.notnull(row["width"]) and pd.notnull(row["depth"])
        else None,
        axis=1,
    )
    final_df = final_df.dropna(subset=["flops_per_token"])
    final_df = final_df[final_df["flops_per_token"] != 0]

    # Compute derived metrics
    final_df["num_tokens"] = final_df["total_flops"] / final_df["flops_per_token"]
    final_df["num_tokens_billions"] = final_df["num_tokens"] / 1e9
    final_df["log_total_flops"] = np.log10(final_df["total_flops"])
    final_df
    return (final_df,)


@app.cell
def _():
    ratios = {}
    with open("/data/wheld3/dialect_sl/ratios.txt", "r") as file:
        for line in file.readlines():
            split, compress_ratio = line.strip().split(": ")
            ratios[split] = float(compress_ratio)
    return (ratios,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Fitting Relative Scaling Laws

    In the first work on Scaling laws for Neural Language Models ([Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)), they find that the loss of a language model can be reliably predicted using the functional form

    \[
    \text{Error} = \frac{\alpha}{F^{\beta}}
    \]

    where $F$ denotes the amount of compute used to train the model in FLOPs, $\alpha$ denotes the initial loss, and $\beta$ denotes the rate at which loss improves with more compute. Here we are interested in using this form to forecast an answer to the question **"Do disparities across dialects diminish as models scale?"**. Prior work has shown that there are few domains which exhibit [inverse scaling](https://arxiv.org/abs/2306.09479), so we do not expect to see this for any of the dialects. Instead, we are interested in how performance improves across dialects *relative* to other dialects in English. From a fairness perspective, this is connected to the core idea of allocational harms e.g. are the benefits of Large Language Models likely to increase disparity across groups due to uneven allocation of benefits across groups?

    As such, we are interesed in fitting a "Relative" Scaling Law. For Dialect $s$ and Dialect $o$, the relative scaling law is simply

    \[
    \text{Relative Error} = \frac{\text{Error}_s - \text{Error}_o}{\text{Error}_s} \\
     = \frac{\frac{\alpha_s}{F^{\beta_s}} - \frac{\alpha_o}{F^{\beta_o}}}{\frac{\alpha_s}{F^{\beta_s}}} \\
     = 1 - \frac{\alpha_o}{\alpha_s} \cdot F^{\beta_s-\beta_o}
    \]

    For dialects, we have a reasonable hypothesis that US English will be the best represented for Language Models since the US has [the largest population of English speakers in the world](https://en.wikipedia.org/wiki/List_of_countries_by_English-speaking_population), had [access to the internet before the rest of the world](https://cs.stanford.edu/people/eroberts/courses/soco/projects/distributed-computing/html/history.html), and has [generally been the largest producer of NLP research by volume](https://arxiv.org/abs/2311.08391). As such, we treat $\alpha_s$ and $\beta_o$ as implicit constants and simply fit 

    \[
    \text{Relative Error} = 1 - \alpha_o \cdot F^{-\beta_o}
    \]

    This formula has a clear interpretation! $\alpha_o$ describes the baseline disparity between US English and the nation $o$'s variety of English, while $\beta_o$ describes the rate at which this disparity grows or shrinks (or stays the same). Most importantly assuming that $\alpha_o$ is positive, if $\beta_o > 0$, then scaling will reduce the relative disparity between $o$ and US English. On the other hand, if $\beta_o < 0$, then scaling will increase the relative disparity.
    """
    )
    return


@app.cell
def _():
    import statsmodels.api as sm
    import plotly.graph_objects as go
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from typing import NamedTuple


    diverging_colors = [
        "#850550", "#BF0F76", "#FC7BCE", "#FFADE4", "#FFD9F2",
        "#F0F2F5", "#CBF9D7", "#A3E6B5", "#45BD62", "#24A142", "#1D632E",
    ]

    categorical_colors = [
        "#1877F2", "#F0701A", "#5A24C7", "#E42C97", "#00487C",
        "#0EAC96", "#AB76FF", "#B50550", "#0099E6", "#22085F", "#783301"
    ]
    return NamedTuple, categorical_colors, go, jax, jnp, minimize


@app.cell
def _(NamedTuple, jnp, np, pd):
    def explode_metrics(
        final_df, options, ratios, domain_folder="lm_eval/", baseline="USA"
    ):
        plot_df2 = final_df.dropna(subset=["total_flops"]).copy()
        plot_df2["total_flops"] = pd.to_numeric(plot_df2["total_flops"])

        stacked_data = []
        print(options)
        for kaplan_metric in options:
            if (domain_folder not in kaplan_metric
                or baseline in kaplan_metric
            ):
                continue

            tmp_df = plot_df2.copy()
            tmp_df[kaplan_metric] = pd.to_numeric(
                tmp_df[kaplan_metric], errors="coerce"
            )
            tmp_df = tmp_df.dropna(subset=[kaplan_metric])

            tmp_df["relative_bpb"] = tmp_df[kaplan_metric]
            tmp_df["log_flops"] = np.log10(tmp_df["total_flops"])
            tmp_df["category"] = kaplan_metric.split("/")[-2]
            tmp_df["metric"] = kaplan_metric
            tmp_df["gzip_bpb"] = 0

            stacked_data.append(
                tmp_df[
                    [
                        "log_flops",
                        "relative_bpb",
                        "category",
                        "metric",
                        "total_flops",
                        "num_tokens",
                        "num_parameters",
                        "gzip_bpb",
                    ]
                ]
            )

        combined_df = pd.concat(stacked_data, ignore_index=True)
        combined_df["gzip_bpb"] = (
            combined_df["gzip_bpb"] - combined_df["gzip_bpb"].mean()
        )
        return combined_df


    class FeatureBatch(NamedTuple):
        # Every field is a JAX array, so the whole object is automatically a PyTree.
        cat: jnp.ndarray  # one-hot categories, shape (N, n_cats)
        gzip_bpb: jnp.ndarray  # shape (N,)
        total_flops: jnp.ndarray  # shape (N,)
        log_flops: jnp.ndarray  # shape (N,)
        num_tokens: jnp.ndarray  # shape (N,)
        num_parameters: jnp.ndarray  # shape (N,)
        n_cats: int  # metadata – OK as a leaf, or make it static (see below)

        def filter_by_category(self, cat_idx: int) -> "FeatureBatch":
            # Create a mask where the category is equal to cat_idx
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


    def create_feature_batch(df: pd.DataFrame) -> FeatureBatch:
        # One-hot-encode *once* and keep it as a single 2-D array
        cat = jnp.asarray(
            pd.get_dummies(df["category"], dtype=float).to_numpy()
        )  # (N, n_cats)

        return FeatureBatch(
            cat=cat,
            gzip_bpb=jnp.asarray(df["gzip_bpb"].to_numpy()),
            total_flops=jnp.asarray(df["total_flops"].to_numpy()),
            log_flops=jnp.asarray(df["log_flops"].to_numpy()),
            num_tokens=jnp.asarray(df["num_tokens"].to_numpy()),
            num_parameters=jnp.asarray(df["num_parameters"].to_numpy()),
            n_cats=cat.shape[-1],  # simple scalar leaf
        )
    return FeatureBatch, create_feature_batch, explode_metrics


@app.cell
def _(FeatureBatch, jax, jnp, np):
    def mse_factory():
        @jax.jit
        def mse(preds, truth, reduction=jnp.sum):
            return reduction((truth - preds) ** 2)

        return mse

    def huber_factory():
        @jax.jit
        def huber(preds, truth, reduction=jnp.sum, delta=1e-3):
            """Standard Huber loss - JAX version"""
            residuals = truth - preds
            cond = jnp.abs(residuals) <= delta
            loss = jnp.where(
                cond, 0.5 * residuals**2, delta * (jnp.abs(residuals) - 0.5 * delta)
            )
            return reduction(loss)

        return huber


    def power_law_compute_factory(
        n_cats: int,
        axis_of_interest: str = "log_flops",
    ):
        try:
            field_idx = FeatureBatch._fields.index(axis_of_interest)
        except ValueError as e:
            raise ValueError(
                f"{axis_of_interest!r} is not a field of FeatureBatch"
            ) from e

        @jax.jit
        def predict(theta, batch):
            a = theta[0:n_cats]  # (K,)
            b = theta[n_cats : 2 * n_cats]

            x = batch[field_idx]
            a_cat = batch.cat @ a
            b_cat = batch.cat @ b
            return (
                a_cat * (x**b_cat)
            )  # This comes from difference of power laws

        def init():
            # a = 1, b = −0.5  for every cat
            return np.concatenate(
                [np.zeros(n_cats, dtype=float), np.zeros(n_cats, dtype=float), 
                    np.zeros(1, dtype=float),]
            ).tolist()

        return predict, init


    def gzip_adjusted_factory(
        n_cats: int,
        axis_of_interest: str = "log_flops",
    ):
        try:
            field_idx = FeatureBatch._fields.index(axis_of_interest)
        except ValueError as e:
            raise ValueError(
                f"{axis_of_interest!r} is not a field of FeatureBatch"
            ) from e

        @jax.jit
        def predict(theta, batch):
            a = theta[0:n_cats]  # (K,)
            b = theta[n_cats : 2 * n_cats]
            gzip_slope = theta[-1]

            x = batch[field_idx]
            a_cat = batch.cat @ a
            b_cat = batch.cat @ b
            return 1 - (a_cat * (x**(b_cat + (gzip_slope * batch.gzip_bpb) )))

        def init():
            # a = 1, b = −0.5  for every cat
            return np.concatenate(
                [
                    np.zeros(n_cats, dtype=float),
                    np.zeros(n_cats, dtype=float),
                    np.zeros(1, dtype=float),
                ]
            ).tolist()

        return predict, init

    def chinchilla_compute_factory(n_cats: int):
        @jax.jit
        def predict(theta, batch):
            k_D = theta[0 : 1 * n_cats]  # (K,)
            k_P = theta[1 * n_cats : 2 * n_cats]
            e_D = theta[2 * n_cats : 3 * n_cats]
            e_P = theta[3 * n_cats : 4 * n_cats]

            kD_cat = batch.cat @ k_D
            kP_cat = batch.cat @ k_P
            eD_cat = batch.cat @ e_D
            eP_cat = batch.cat @ e_P

            D_opt = kD_cat * (jnp.log10(batch.num_tokens)**(eD_cat))
            P_opt = kP_cat * (jnp.log10(batch.num_parameters)**eP_cat)
            return 1 - (D_opt + P_opt)

        def init():
            # k_N = k_T = 1,  e_N = 0.49,  e_T = 0.51  for every cat
            return np.concatenate(
                [
                    np.zeros(n_cats, dtype=float),  # k_N
                    np.zeros(n_cats, dtype=float),  # k_T
                    np.zeros(n_cats, dtype=float),  # e_N
                    np.zeros(n_cats, dtype=float),  # e_T
                ]
            ).tolist()

        return predict, init

    def chinchilla_gzip_compute_factory(n_cats: int):
        @jax.jit
        def predict(theta, batch):
            k_D = theta[0 : 1 * n_cats]  # (K,)
            k_P = theta[1 * n_cats : 2 * n_cats]
            e_D = theta[2 * n_cats : 3 * n_cats]
            e_P = theta[3 * n_cats : 4 * n_cats]
            gzip_slope = theta[-1]

            kD_cat = batch.cat @ k_D
            kP_cat = batch.cat @ k_P
            eD_cat = batch.cat @ e_D
            eP_cat = batch.cat @ e_P

            D_opt = kD_cat * (batch.num_tokens**(eD_cat + (gzip_slope * batch.gzip_bpb) ))
            P_opt = kP_cat * (batch.num_parameters**eP_cat)
            return 1- (D_opt + P_opt)

        def init():
            # k_N = k_T = 1,  e_N = 0.49,  e_T = 0.51  for every cat
            return np.concatenate(
                [
                    np.zeros(n_cats, dtype=float),  # k_N
                    np.zeros(n_cats, dtype=float),  # k_T
                    np.zeros(n_cats, dtype=float),  # e_N
                    np.zeros(n_cats, dtype=float),  # e_T
                    np.zeros(1, dtype=float),
                ]
            ).tolist()

        return predict, init
    return (
        chinchilla_gzip_compute_factory,
        gzip_adjusted_factory,
        huber_factory,
        mse_factory,
        power_law_compute_factory,
    )


@app.cell
def _(
    create_feature_batch,
    explode_metrics,
    jax,
    jnp,
    minimize,
    mo,
    mse_factory,
    pd,
    power_law_compute_factory,
):
    @mo.persistent_cache
    def fit_scaling_law(
        final_df,
        options,
        ratios,
        loss_factory=mse_factory,
        predict_factory=power_law_compute_factory,
        domain_folder="lm_eval/", 
        baseline="USA"
    ):
        combined_df = explode_metrics(final_df, options, ratios, domain_folder=domain_folder, baseline=baseline)
        loss_fn = loss_factory()

        feature_batch = create_feature_batch(combined_df)
        n_cats = feature_batch.n_cats
        y = jnp.array(combined_df["relative_bpb"].astype(float))

        predict, init = predict_factory(n_cats=n_cats)
        opt_fn = lambda theta: loss_fn(predict(theta, feature_batch), y)
        grad_fn = jax.grad(opt_fn)

        theta0 = init()

        res = minimize(
            opt_fn,
            theta0,
            jac=grad_fn,
            method="L-BFGS-B",
            options=dict(
                maxiter=500_000,  # iterations (default 15000)
                maxfun=5000_000,  # allowed objective evals
                maxcor=1000,  # history size (default 10)
                ftol=1e-12,  # function-value tolerance
                gtol=1e-12,  # gradient-norm tolerance
                maxls=200,  # line-search steps per iter (default 20)
                disp=True,  # live progress in stdout
            ),
        )
        if not res.success:
            raise RuntimeError(f"Fit did not converge: {res.message}")

        return res

    def pretty_print_coefficients(theta_opt, combined_df):
        """
        Nicely display the per-category a & b coefficients and the
        single global gzip_slope coming from `gzip_adjusted_factory`.

        Parameters
        ----------
        theta_opt   : array-like, shape (2 * n_cats + 1,)
            Optimized parameter vector returned by `minimize`.
        combined_df : pd.DataFrame
            The same dataframe you passed into `explode_metrics`; only
            used here to recover the category names and count.
        """
        cats   = sorted(combined_df["category"].unique())
        n_cats = len(cats)

        lets = ["a", "b", "c", "d", "e"]
        # Slice theta into its logical pieces
        cat_coefs = {}
        if n_cats > 1:
            range_seq = range(len(theta_opt) // n_cats)
        else:
            range_seq = range(len(theta_opt) - 1)
        for i in range_seq:
            let = lets[i]
            vals = theta_opt[i*n_cats: (i+1)*n_cats]
            cat_coefs[let] = vals

        gzip_slope   = theta_opt[-1]

        coef_df = pd.DataFrame({
            "category": cats,
            **cat_coefs
        })

        # Consistent, readable floats
        with pd.option_context("display.float_format", "{:,.6g}".format):
            print("\nPer-category coefficients:")
            print(coef_df.to_string(index=False))

        print(f"\nGlobal gzip_slope: {gzip_slope:.6g}")
        return coef_df
    return fit_scaling_law, pretty_print_coefficients


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Optimization Methodology 

    ### Basic Method

    We follow the method of [the EpochAI Chinchilla reproduction](https://arxiv.org/html/2404.10102v2) using Scipy's implementation of L-BFGS-B, summed Huber loss ($\delta=0.001$), and an extremely small tolerance to prevent early stopping. Our loss function and prediction method are implemented in Jax so that we can provide SciPy an analytical gradient for improved convergence speed. 

    We further want control for a few factors which would confound the interpretation of scaling law slopes in non-meaningful ways. Firstly, is the fact that some dialects of English are simply less common and therefore should be (for unconditional LM generation) assigned lower probabilities. To account for this, we mask out the errors from the head (first 30 tokens) of each document and only measure the scaling law over BPB for this (conditional) distribution of continuations. A second possible confound is that some dialects could have inherently higher entropy, we account foor this by adding an additional factor in the regression based on [GZip compression ratio](https://arxiv.org/html/2405.16684v1). 

    ### Confidence Intervals 

    Most importantly, we are interested in using our fitted scaling laws to forecast what future disparities between dialects will look like! As such, the confidence interval over much larger scale models is as, if not more, important than our exact estimate. Following [Hoffman et al.](https://arxiv.org/abs/2203.15556), we utilize Bootstrapping to get an estimate on our parameters. However, since our Gemstone runs utilize WSD many of the points are correlated due to their initialization which breaks normal $\text{I.I.D}$ assumptions for bootstrapping. As such, we perform [Block Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Block_bootstrap) on the model confidgurations, which resamples at the run level rather than at the individual point level so that each resample follows the data generation process.
    """
    )
    return


@app.cell
def _(List, explode_metrics, fit_scaling_law, jnp, mo, np, pd):
    from functools import partial


    @mo.persistent_cache
    def bootstrap_scaling_law(
        final_df: pd.DataFrame,
        options,
        ratios,
        *,
        loss_factory,
        predict_factory,
        n_boot: int = 1_000,
        seed: int | None = None,
        block: bool = True,
        domain_folder="lm_eval/",
        baseline="USA",
    ):
        """
        Bootstrap re-fit of `fit_scaling_law` with a tidy output.

        Each row in the returned `boot_df` has columns
            loss, gzip_slope, a[cat₁], …, a[catₙ], b[cat₁], …, b[catₙ]

        so it can be inspected or summarised with ordinary pandas tools.

        Parameters
        ----------
        final_df, options, ratios, loss_fn, predict_factory
            Passed straight through to `fit_scaling_law`.
        n_boot : int, default 1000
            Number of bootstrap draws.
        seed : int or None
            RNG seed for reproducibility.

        Returns
        -------
        boot_df : pd.DataFrame
            Tidy per-draw coefficients + loss.
        summary : pd.DataFrame
            Descriptive stats across draws.
        """
        # ----- static information we only need once ----------------------------
        base_combined = explode_metrics(
            final_df,
            options,
            ratios,
            domain_folder=domain_folder,
            baseline=baseline,
        )
        cats = sorted(base_combined["category"].unique())
        n_cats = len(cats)

        rng = np.random.default_rng(seed)
        rows = []

        # -----------------------------------------------------------------------
        for i in mo.status.progress_bar(range(n_boot)):
            if not block:
                sample = final_df.sample(
                    frac=1.0,
                    replace=True,
                    random_state=int(rng.integers(0, 2**32)),
                )
            else:
                grouped = list(final_df.groupby(["width", "depth"], dropna=False))
                blocks: List[pd.DataFrame] = [g for _, g in grouped]

                n_blocks = len(blocks)
                if n_blocks == 0:
                    raise ValueError(
                        "No blocks found: ensure 'width' and 'depth' columns exist and are not all NaN."
                    )

                sampled_idx = rng.integers(0, n_blocks, size=n_blocks)

                resampled = pd.concat(
                    [blocks[i] for i in sampled_idx], ignore_index=True
                )

                sample = resampled.reset_index(drop=True)

            res = fit_scaling_law(
                sample,
                options,
                ratios,
                loss_factory=loss_factory,
                predict_factory=predict_factory,
                domain_folder=domain_folder,
                baseline=baseline,
            )

            theta = res.x
            vals = []

            if n_cats > 1:
                range_seq = range(len(theta) // n_cats)
            else:
                range_seq = range(len(theta) - 1)
            for i in range_seq:
                vals.append(theta[i*n_cats:(i+1)*n_cats])

            for i in range(1, (len(theta) % n_cats)+1):
                vals.append(theta[-i])
            lets = ["a", "b", "c", "d", "e"]
            row = {}

            if n_cats > 1:
                range_seq = range(len(theta) // n_cats)
            else:
                range_seq = range(len(theta) - 1)

            for i in range_seq:
                let = lets[i]
                for c, v in zip(cats, vals[i]):
                    row[f"{let}[{c}]"] = v

            if n_cats > 1:
                range_seq = range(1, (len(theta) % n_cats)+1)
            else:
                range_seq = range(1,2)

            for i in range_seq:
                let = lets[i]
                row[f"{let}"] = vals[-i]
            print(row)
            rows.append(row)

        boot_df = pd.DataFrame(rows)
        # A quick numeric summary—feel free to ignore or extend
        summary = boot_df.describe(percentiles=[0.025, 0.5, 0.975])
        return boot_df, summary


    def fit_and_bootstrap(
        final_df,
        options,
        ratios,
        loss_factory,
        factory_fn,
        n_boot,
        seed,
        domain_folder="lm_eval/",
        baseline="USA",
    ):
        final_df = final_df[
            ~final_df["Run Name"].str.contains("OLMo", na=False)
        ].copy()
        # ----------------------------------- 1. Fit once on original data
        res = fit_scaling_law(
            final_df,
            options,
            ratios,
            loss_factory=loss_factory,
            predict_factory=factory_fn,
            domain_folder=domain_folder,
            baseline=baseline,
        )
        if not res.success:
            raise RuntimeError(res.message)
        theta_opt = jnp.asarray(res.x)

        # ----------------------------------- 2. Bootstrap
        boot_df, _ = bootstrap_scaling_law(
            final_df,
            options,
            ratios,
            loss_factory=loss_factory,
            predict_factory=factory_fn,
            n_boot=n_boot,
            seed=seed,
            domain_folder=domain_folder,
            baseline=baseline,
        )
        return theta_opt, boot_df
    return (fit_and_bootstrap,)


@app.cell
def _(
    FeatureBatch,
    categorical_colors,
    create_feature_batch,
    explode_metrics,
    final_df,
    fit_and_bootstrap,
    go,
    gzip_adjusted_factory,
    huber_factory,
    jnp,
    mo,
    np,
    options,
    power_law_compute_factory,
    pretty_print_coefficients,
    ratios,
):
    # ------------------------------------------------------------
    #  Palette helpers (uses the 11-color categorical scheme you
    #  keep in `categorical_colors`)
    # ------------------------------------------------------------
    def rgba(hex_color, alpha=0.25):
        """Convert '#RRGGBB' → 'rgba(r,g,b,alpha)'."""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # ------------------------------------------------------------
    #  Main function
    # ------------------------------------------------------------
    def wrapper_combined_categorical_jax(
        *,
        n_boot: int = 10,
        seed: int = 42,
        loss_factory=huber_factory,
        factory_fn=gzip_adjusted_factory,
        options=options
    ):
        domain_folder="lm_eval/"
        baseline="USA"
        theta_opt, boot_df = fit_and_bootstrap(
            final_df,
            options,
            ratios,
            loss_factory,
            factory_fn,
            n_boot,
            seed,
            domain_folder=domain_folder,
            baseline=baseline,
        )
        # ----------------------------------- 3. Set-up helpers for prediction
        combined_df = explode_metrics(final_df, options, ratios, domain_folder=domain_folder, baseline=baseline)
        feature_batch = create_feature_batch(combined_df)
        cats = sorted(combined_df["category"].unique())
        n_cats = feature_batch.n_cats
        predict, _ = factory_fn(n_cats=n_cats)

        # Grid of FLOPs we’ll evaluate on
        flops = np.logspace(18, 30, num=500, dtype=float)  # (500,)
        log_F = np.log10(flops)  # (500,)

        # ----------------------------------- 4. Plotly figure
        fig_all = go.Figure(
            layout=dict(
                template="plotly_white",
                xaxis=dict(
                    title="Total FLOPs",
                    type="log",
                    exponentformat="power",
                    showgrid=True,
                ),
                yaxis=dict(
                    title="Difference Aware Fairness Accuracy", zeroline=True, showgrid=True
                ),
                legend_title="Category",
                margin=dict(l=60, r=20, t=40, b=60),
            )
        )

        # ----------------------------------- 5. One loop per category
        for i, cat in enumerate(cats):
            color = categorical_colors[i % len(categorical_colors)]
            shade_color = rgba(color, 0.25)  # translucent for CI ribbon

            # --- median gzip_bpb for this category
            gzip_val = combined_df.loc[
                combined_df["category"] == cat, "gzip_bpb"
            ].unique()[0]

            # --- FeatureBatch template for this curve
            one_hot = np.zeros((flops.size, n_cats), dtype=float)
            one_hot[:, i] = 1.0
            base_batch = FeatureBatch(
                cat=jnp.asarray(one_hot),
                gzip_bpb=jnp.asarray(np.full(flops.size, gzip_val)),
                total_flops=jnp.asarray(flops),
                log_flops=jnp.asarray(log_F),
                num_tokens=jnp.zeros(flops.size),
                num_parameters=jnp.zeros(flops.size),
                n_cats=n_cats,
            )

            # --- Point estimates (median line)
            y_median = np.asarray(predict(theta_opt, base_batch))

            # --- Raw data points
            sub = combined_df[combined_df["category"] == cat].copy()
            sub.loc[combined_df["num_tokens"] > 1000e9, "relative_bpb"] += 0.025
            sub["baseline"] = combined_df["num_tokens"] > 1000e9

            y_pred = np.asarray(predict(theta_opt, feature_batch.filter_by_category(i)))
            resid_std = np.std(sub["relative_bpb"]-y_pred)  # e.g., from training: np.std(y_train - y_pred_train)

            # --- Bootstrap: reconstruct θ for every draw → predictions
            a_key = f"a[{cat}]"
            b_key = f"b[{cat}]"
            a_boot = boot_df[a_key].to_numpy()      # (n_boot,)
            b_boot = boot_df[b_key].to_numpy()      # (n_boot,)
            boot_df = boot_df.rename(columns={"b": "gzip_slope"})
            g_boot = boot_df["gzip_slope"].to_numpy()  # (n_boot,)

            # Vectorised θ construction:   θ = [a_vec, b_vec, gzip_slope]
            theta_boot = np.zeros((n_boot, 2 * n_cats + 1), dtype=float)
            theta_boot[:, i] = a_boot
            theta_boot[:, n_cats + i] = b_boot
            theta_boot[:, -1] = g_boot

            # Broadcast-predict over all draws
            y_boot = np.empty((n_boot, flops.size), dtype=float)
            for k in range(n_boot):
                y_pred = np.asarray(predict(theta_boot[k], base_batch))  # shape: (n_obs,)
                y_boot[k] = y_pred

            # 99.8% prediction interval (wider than CI for same quantiles)
            y_lower = np.quantile(y_boot, 0.001, axis=0)- (1.96*resid_std)
            y_upper = np.quantile(y_boot, 0.999, axis=0)+ (1.96*resid_std)

            # --- Ribbon (add before the line so it sits underneath)
            fig_all.add_trace(
                go.Scatter(
                    x=np.concatenate([flops, flops[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill="toself",
                    fillcolor=shade_color,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    legendgroup=str(cat),
                    name=str(cat),
                )
            )

            # --- Median prediction line
            fig_all.add_trace(
                go.Scatter(
                    x=flops,
                    y=y_median,
                    mode="lines",
                    legendgroup=str(cat),
                    name=str(cat),
                    line=dict(color=color, width=2),
                )
            )

            fig_all.add_trace(
                go.Scatter(
                    x=sub["total_flops"],
                    y=sub["relative_bpb"],
                    marker_symbol=sub["baseline"],
                    mode="markers",
                    legendgroup=str(cat),
                    name=str(cat),
                    marker=dict(color=color, size=12, opacity=0.35),
                    showlegend=True,
                )
            )

        # Update layout to ensure proper rendering
        fig_all.update_layout(yaxis=dict(tickformat=".0%"), showlegend=True)
        baselines = [(2.326305e23, "OLMo 2 7B"), (5.632975e23, "OLMo 2 13B")]
        for flops, name in baselines:
            # Add the vertical line
            fig_all.add_vline(
                x=flops,
                line_dash="dash",
                line_color="#7F7776",
                line_width=1,
            )

            # Try this simplified annotation approach
            fig_all.add_annotation(
                x=np.log10(flops) + 0.06,
                y=1 - (0.017 * len(name)),  # Top of the plot area
                text=name,
                showarrow=False,
                textangle=90,  # Keep text horizontal for now
                xanchor="center",
                yanchor="bottom",
                font=dict(size=12, color="#7F7776"),
                yref="paper",
                borderwidth=1,
            )
        fig_all.update_yaxes(range = (-.15, .15))
        # ----------------------------------- 6.  Pretty coefficients + return
        coef_df = pretty_print_coefficients(theta_opt, combined_df)
        return [mo.ui.plotly(fig_all)], coef_df, boot_df


    # Run the categorical regression
    plotly1, coef_df, boot_df = wrapper_combined_categorical_jax(factory_fn=power_law_compute_factory, options=["lm_eval/discrim_eval_implicit/asian_bias","lm_eval/discrim_eval_explicit/asian_bias"])
    #plotly2, coef_df, boot_df = wrapper_combined_categorical_jax(factory_fn=power_law_compute_factory, options=[])
    mo.vstack(plotly1)
    return (rgba,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Preliminary Results

    Our initial results match our expectations! There is no sub-corpus of ICE which is projected to outperform us English, even if we project to 5 order of magnitudes beyond Llama 3 405B. In general, we see that scale exacerbates relative error rate compared with US English. Notably, this is despite ICE's focus on sampling data from highly educated English speakers in these nations. 

    As a preliminary explanatory study, we can see
    """
    )
    return


@app.cell
def _(
    FeatureBatch,
    categorical_colors,
    chinchilla_gzip_compute_factory,
    create_feature_batch,
    explode_metrics,
    final_df,
    fit_and_bootstrap,
    go,
    huber_factory,
    jnp,
    mo,
    np,
    options,
    pretty_print_coefficients,
    ratios,
    rgba,
):
    # ------------------------------------------------------------
    #  Main function
    # ------------------------------------------------------------
    def wrapper_chinchilla(
        *,
        n_boot: int = 10,
        seed: int = 42,
        loss_factory=huber_factory,
        factory_fn=chinchilla_gzip_compute_factory,
    ):
        domain_folder="lm_eval/"
        baseline="USA"
        theta_opt, boot_df = fit_and_bootstrap(
            final_df,
            options,
            ratios,
            loss_factory,
            factory_fn,
            n_boot,
            seed,
            domain_folder=domain_folder,
            baseline=baseline,
        )
        # ----------------------------------- 3. Set-up helpers for prediction
        combined_df = explode_metrics(final_df, options, ratios, domain_folder=domain_folder, baseline=baseline)
        feature_batch = create_feature_batch(combined_df)
        cats = sorted(combined_df["category"].unique())
        n_cats = feature_batch.n_cats
        predict, _ = factory_fn(n_cats=n_cats)

        # Grid of FLOPs we’ll evaluate on
        flops = np.logspace(18, 30, num=500, dtype=float)  # (500,)
        final_df.loc[final_df["model_id"].str.contains("7B"), "num_parameters"] = 7.3e9
        final_df.loc[final_df["model_id"].str.contains("13B"), "num_parameters"] = 13.7e9
        vals = final_df[["num_parameters", "num_tokens", "total_flops"]].sort_values(by="total_flops")

        flops = vals["total_flops"]
        tokens = vals["num_tokens"]
        params = vals["num_parameters"]
        print(vals)

        #flops_per_tokens = 
        log_F = np.log10(flops)  # (500,)

        # ----------------------------------- 4. Plotly figure
        fig_all = go.Figure(
            layout=dict(
                template="plotly_white",
                xaxis=dict(
                    title="Total Tokens",
                    type="log",
                    exponentformat="power",
                    showgrid=True,
                ),
                yaxis=dict(
                    title="Predicted Relative BPB", zeroline=True, showgrid=True
                ),
                legend_title="Category",
                margin=dict(l=60, r=20, t=40, b=60),
            )
        )

        # ----------------------------------- 5. One loop per category
        for i, cat in enumerate(cats):
            color = categorical_colors[i % len(categorical_colors)]
            shade_color = rgba(color, 0.25)  # translucent for CI ribbon

            # --- median gzip_bpb for this category
            gzip_val = combined_df.loc[
                combined_df["category"] == cat, "gzip_bpb"
            ].unique()[0]

            # --- FeatureBatch template for this curve
            one_hot = np.zeros((flops.size, n_cats), dtype=float)
            one_hot[:, i] = 1.0

            base_batch = FeatureBatch(
                cat=jnp.asarray(one_hot),
                gzip_bpb=jnp.asarray(np.full(flops.size, gzip_val)),
                total_flops=jnp.asarray(flops),
                log_flops=jnp.asarray(log_F),
                num_tokens=jnp.asarray(tokens),
                num_parameters=jnp.asarray(params),
                n_cats=n_cats,
            )

            # --- Point estimates (median line)
            y_median = np.asarray(predict(theta_opt, base_batch))

            # --- Raw data points
            sub = combined_df[combined_df["category"] == cat].copy()
            sub.loc[combined_df["num_tokens"] > 1000e9, "relative_bpb"] += 0.025
            sub["baseline"] = combined_df["num_tokens"] > 1000e9

            y_pred = np.asarray(predict(theta_opt, feature_batch.filter_by_category(i)))
            resid_std = np.std(sub["relative_bpb"]-y_pred)  # e.g., from training: np.std(y_train - y_pred_train)

            # --- Bootstrap: reconstruct θ for every draw → predictions
            a_key = f"a[{cat}]"
            b_key = f"b[{cat}]"
            c_key = f"c[{cat}]"
            d_key = f"d[{cat}]"
            a_boot = boot_df[a_key].to_numpy()      # (n_boot,)
            b_boot = boot_df[b_key].to_numpy()      # (n_boot,)
            c_boot = boot_df[c_key].to_numpy()      # (n_boot,)
            d_boot = boot_df[d_key].to_numpy()      # (n_boot,)
            boot_df = boot_df.rename(columns={"b": "gzip_slope"})
            g_boot = boot_df["gzip_slope"].to_numpy()  # (n_boot,)

            # Vectorised θ construction:   θ = [a_vec, b_vec, gzip_slope]
            theta_boot = np.zeros((n_boot, len(boot_df.columns)), dtype=float)
            theta_boot[:, i] = a_boot
            theta_boot[:, n_cats + i] = b_boot
            theta_boot[:, (2*n_cats) + i] = c_boot
            theta_boot[:, (3*n_cats) + i] = d_boot
            theta_boot[:, -1] = g_boot

            # Broadcast-predict over all draws
            y_boot = np.empty((n_boot, flops.size), dtype=float)
            for k in range(n_boot):
                y_pred = np.asarray(predict(theta_boot[k], base_batch))  # shape: (n_obs,)
                y_boot[k] = y_pred

            # 99.8% prediction interval (wider than CI for same quantiles)
            y_lower = np.quantile(y_boot, 0.001, axis=0)#- (1.96*resid_std)
            y_upper = np.quantile(y_boot, 0.999, axis=0)#+ (1.96*resid_std)

            # --- Ribbon (add before the line so it sits underneath)
            # fig_all.add_trace(
            #     go.Scatter(
            #         x=np.concatenate([flops, flops[::-1]]),
            #         y=np.concatenate([y_upper, y_lower[::-1]]),
            #         fill="toself",
            #         fillcolor=shade_color,
            #         line=dict(color="rgba(255,255,255,0)"),
            #         hoverinfo="skip",
            #         legendgroup=str(cat),
            #         name=str(cat),
            #     )
            # )

            # --- Median prediction line
            # fig_all.add_trace(
            #     go.Scatter(
            #         x=tokens,
            #         y=y_median,
            #         mode="markers",
            #         legendgroup=str(cat),
            #         name=str(cat),
            #         line=dict(color=color, width=2),
            #     )
            # )

            fig_all.add_trace(
                go.Scatter(
                    x=sub["num_tokens"],
                    y=sub["relative_bpb"],
                    marker_symbol=sub["baseline"],
                    #hoverinfo=["num_parameters"],
                    mode="markers",
                    legendgroup=str(cat),
                    name=str(cat),
                    marker=dict(color=color, size=12, opacity=0.35),
                    showlegend=True,
                )
            )

        # Update layout to ensure proper rendering
        fig_all.update_layout(yaxis=dict(tickformat=".0%"), showlegend=True)
        # baselines = [(2.326305e23, "OLMo 2 7B"), (5.632975e23, "OLMo 2 13B")]
        # for flops, name in baselines:
        #     # Add the vertical linex
        #     fig_all.add_vline(
        #         x=flops,
        #         line_dash="dash",
        #         line_color="#7F7776",
        #         line_width=1,
        #     )

        #     # Try this simplified annotation approach
        #     fig_all.add_annotation(
        #         x=np.log10(flops) + 0.06,
        #         y=1 - (0.017 * len(name)),  # Top of the plot area
        #         text=name,
        #         showarrow=False,
        #         textangle=90,  # Keep text horizontal for now
        #         xanchor="center",
        #         yanchor="bottom",
        #         font=dict(size=12, color="#7F7776"),
        #         yref="paper",
        #         borderwidth=1,
        #     )

        # ----------------------------------- 6.  Pretty coefficients + return
        coef_df = pretty_print_coefficients(theta_opt, combined_df)
        return [mo.ui.plotly(fig_all)], coef_df, boot_df

    # Run the categorical regression
    _plotly, _coef_df, _boot_df = wrapper_chinchilla()
    mo.vstack(_plotly)
    return


if __name__ == "__main__":
    app.run()
