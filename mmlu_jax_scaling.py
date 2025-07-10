import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full", app_title="Dialect Scaling Laws")


@app.cell
def _(mo):
    def _marimo_theme(default: str = "light") -> str:
        """
        Returns 'dark' or 'light' depending on the active Marimo theme.
        Falls back to *default* if executed outside Marimo.
        """
        try:
            return mo.app_meta().theme           # "dark" | "light" 
        except Exception:
            return default

    import plotly.io as pio
    TEMPLATE_MAP = {"dark": "plotly_dark", "light": "plotly_white"}

    def apply_plotly_theme():
        tmpl = TEMPLATE_MAP[_marimo_theme()]
        pio.templates.default = tmpl            # global override
        return tmpl
    return (apply_plotly_theme,)


@app.cell
def _():
    import wandb

    _ = wandb.login()

    api = wandb.Api()

    runs = api.runs(
        "marin-community/marin", filters={"displayName": {"$regex": "marin-us-central2.*mmlu_sl.*"}, "state": "finished"}
    )

    print(list(runs))
    print(runs[0].summary.keys())
    options = [key for key in runs[0].summary.keys() if "lm_eval/" in key and ("choice_logprob" in key or "acc" in key) and "averages" not in key]
    print(options)
    return options, runs


@app.cell
def _():
    import marimo as mo

    VOCAB_OURS = 50304
    SEQ_LEN = 2048
    WORLD_BATCH_SIZE = 2048.0
    HEAD_SIZE = 128
    EXPAND_FACTOR = 4.0


    def flops_per_token_gqa(
        width,
        depth,
        vocab_size=VOCAB_OURS,
        queries_per_group=2,
        seq_len=SEQ_LEN,
    ):
        """
        Some details (negligible even for extremely wide models) omitted, including:
        * numerically stable softmax
        * softmax addition only being over rows
        * dot products being only n-1 additions (fused multiply-add exists anyway)
        """
        num_qheads = width / HEAD_SIZE
        num_kvheads = 2 * num_qheads / queries_per_group

        embeddings = 0  # 0 if sparse lookup, backward FLOPs negligible
        attention = 2.0 * seq_len * (num_qheads + num_kvheads) * width * HEAD_SIZE
        attention += (
            3.5 * seq_len * (num_qheads + num_kvheads / 2) * HEAD_SIZE
        )  # RoPE, as implemented here/GPT-NeoX

        # score FLOPs are halved because causal => triangular mask => usable sparsity
        kq_logits = 1.0 * seq_len * seq_len * HEAD_SIZE * num_qheads
        softmax = 3.0 * seq_len * seq_len * num_qheads
        softmax_q_red = 2.0 * seq_len * seq_len * HEAD_SIZE * num_qheads
        final_linear = 2.0 * seq_len * width * HEAD_SIZE * num_qheads

        attn_bwd = (
            2.0 * attention
            + 2.5 * (kq_logits + softmax + softmax_q_red)
            + 2.0 * final_linear
        ) * depth

        attention += kq_logits + softmax + softmax_q_red + final_linear

        ffw_size = EXPAND_FACTOR * width
        dense_block = (
            6.0 * seq_len * width * ffw_size
        )  # three matmuls instead of usual two because of GEGLU
        dense_block += (
            10 * seq_len * ffw_size
        )  # 7 for other ops: 3 for cubic, two additions, two scalar mults
        dense_block += 2.0 * width * seq_len  # both/sandwich residual additions

        rmsnorm = 2 * 7.0 * width * seq_len
        final_rms_norm = 7.0 * width * seq_len  # one last RMSNorm
        final_logits = 2.0 * seq_len * width * vocab_size

        nonattn_bwd = 2.0 * (
            embeddings
            + depth * (dense_block + rmsnorm)
            + final_rms_norm
            + final_logits
        )

        forward_pass = (
            embeddings
            + depth * (attention + dense_block + rmsnorm)
            + final_rms_norm
            + final_logits
        )

        backward_pass = attn_bwd + nonattn_bwd  # flash attention
        return (forward_pass + backward_pass) / seq_len
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
def _(final_df, np):
    # ──────────────────────────────────────────────────────────────────────────────
    # Two side-by-side Plotly figures
    #   • Panel 1:  NLL  vs.  total FLOPs   (log-linear fit)
    #   • Panel 2:  NLL  vs.  accuracy      (sigmoid fit)
    #
    # Assumes a DataFrame called `final_df` that already contains the columns:
    #   - log_total_flops
    #   - lm_eval/arc_challenge/choice_logprob
    #   - lm_eval/arc_challenge/acc
    #
    # Color & theme use the preferences you gave me earlier.
    # ──────────────────────────────────────────────────────────────────────────────
    from scipy.optimize import curve_fit
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── 1.  Clean & pull the relevant columns ────────────────────────────────────
    _df = final_df.copy()
    # _df = _df[(_df["log_total_flops"] > 20)]
    # _df = _df[((_df["width"] / _df["depth"]) > 100) & (_df["width"] / _df["depth"]) < 200]
    print((_df["width"] / _df["depth"]).mean())
    # print(_df["log_total_flops"])


    x_flops = _df["log_total_flops"].astype(float).values
    task = "mmlu_sl_.*5_shot"
    # ⬇️ columns that end with “…/choice_logprob_norm”
    y_nll = (
        _df.filter(regex=rf"^lm_eval/{task}/choice_logprob$")
        .mean(axis=1)  # row-wise mean
        .values  # NumPy array
    )

    # ⬇️ columns that end with “…/acc”
    x_acc = _df.filter(regex=rf"^lm_eval/{task}/acc$").mean(axis=1).values
    print(x_acc)

    # ── 2.  Fit models ───────────────────────────────────────────────────────────
    # 2-a. Log-linear:  y = m · log₁₀(x) + b
    m, b = np.polyfit(np.log10(x_flops), y_nll, 1)
    x1_fit = np.linspace(x_flops.min(), x_flops.max(), 200)
    y1_fit = m * np.log10(x1_fit) + b


    # 2-b. Sigmoid:  y = L / (1 + exp(-k·(x-x₀))) + c
    def sigmoid(x, L, k, x0, c):
        return L / (1 + np.exp(-k * (x - x0))) + c


    # sensible initial guess
    p0 = [y_nll.max() - y_nll.min(), -10, x_acc.mean(), y_nll.min()]
    # params, _ = curve_fit(sigmoid, x_acc, y_nll, p0=p0, maxfev=10_000)
    # x2_fit = np.linspace(x_acc.min(), x_acc.max(), 200)
    # y2_fit = sigmoid(x2_fit, *params)

    # ── 3.  Plot ─────────────────────────────────────────────────────────────────
    BLUE = "#1877F2"  # Categorical Blue
    ORANGE = "#F0701A"  # Categorical Orange

    _fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("NLL vs. FLOPs (log-linear fit)", "NLL vs. Accuracy"),
    )

    # Panel 1  ─ scatter & fit
    _fig.add_trace(
        go.Scatter(
            x=x_flops,
            y=y_nll,
            mode="markers",
            marker=dict(color=BLUE),
            name="Data",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=x1_fit,
            y=y1_fit,
            mode="lines",
            line=dict(color=BLUE),
            name=f"Fit: y = {m:.3f}·log₁₀(x) + {b:.3f}",
        ),
        row=1,
        col=1,
    )

    # Panel 2  ─ scatter & fit
    _fig.add_trace(
        go.Scatter(
            x=y_nll,
            y=x_acc,
            mode="markers",
            marker=dict(color=ORANGE),
            name="Data",
        ),
        row=1,
        col=2,
    )
    # _fig.add_trace(
    #     go.Scatter(x=y2_fit, y=x2_fit, mode="lines",
    #                line=dict(color=ORANGE),
    #                name="Sigmoid fit"),
    #     row=1, col=2
    # )

    _fig.update_layout(
        title=task,
        template="plotly_white",
        showlegend=False,
        height=500,
        width=1000,
    )

    _fig.update_xaxes(title_text="Total FLOPs (log scale)", row=1, col=1)
    _fig.update_yaxes(title_text="NLL Per Char", row=1, col=1)
    _fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    _fig.update_xaxes(title_text="NLL Per Char", row=1, col=2)

    _fig
    return BLUE, ORANGE, b, go, m, make_subplots, x_flops


@app.cell
def _(BLUE, ORANGE, b, final_df, go, m, make_subplots, np, x_flops):
    _df = final_df.copy()
    # _df = _df[(_df["log_total_flops"] > 20)]
    # _df = _df[((_df["width"] / _df["depth"]) > 100) & (_df["width"] / _df["depth"]) < 200]
    _task = "mmlu_.*5shot"
    # ⬇️ columns that end with “…/choice_logprob_norm”
    _y_nll = (
        _df.filter(regex=rf"^lm_eval/{_task}/choice_logprob$")
        .mean(axis=1)  # row-wise mean
        .values  # NumPy array
    )

    # ⬇️ columns that end with “…/acc”
    _x_acc = _df.filter(regex=rf"^lm_eval/{_task}/acc$").mean(axis=1).values

    # ── 2.  Fit models ───────────────────────────────────────────────────────────
    # 2-a. Log-linear:  y = m · log₁₀(x) + b
    _m, _b = np.polyfit(np.log10(x_flops), _y_nll, 1)
    _x1_fit = np.linspace(x_flops.min(), x_flops.max(), 200)
    _y1_fit = _m * np.log10(_x1_fit) + _b


    # sensible initial guess
    _p0 = [_y_nll.max() - _y_nll.min(), -10, _x_acc.mean(), _y_nll.min()]
    # params, _ = curve_fit(sigmoid, x_acc, y_nll, p0=p0, maxfev=10_000)
    # x2_fit = np.linspace(x_acc.min(), x_acc.max(), 200)
    # y2_fit = sigmoid(x2_fit, *params)

    _fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("NLL vs. FLOPs (log-linear fit)", "NLL vs. Accuracy"),
    )

    # Panel 1  ─ scatter & fit
    _fig.add_trace(
        go.Scatter(
            x=x_flops,
            y=_y_nll,
            mode="markers",
            marker=dict(color=BLUE),
            name="Data",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_x1_fit,
            y=_y1_fit,
            mode="lines",
            line=dict(color=BLUE),
            name=f"Fit: y = {m:.3f}·log₁₀(x) + {b:.3f}",
        ),
        row=1,
        col=1,
    )

    # Panel 2  ─ scatter & fit
    _fig.add_trace(
        go.Scatter(
            x=_y_nll,
            y=_x_acc,
            mode="markers",
            marker=dict(color=ORANGE),
            name="Data",
        ),
        row=1,
        col=2,
    )
    # _fig.add_trace(
    #     go.Scatter(x=y2_fit, y=x2_fit, mode="lines",
    #                line=dict(color=ORANGE),
    #                name="Sigmoid fit"),
    #     row=1, col=2
    # )

    _fig.update_layout(
        title=_task,
        template="plotly_white",
        showlegend=False,
        height=500,
        width=1000,
    )

    _fig.update_xaxes(title_text="Total FLOPs (log scale)", row=1, col=1)
    _fig.update_yaxes(title_text="NLL Per Char", row=1, col=1)
    _fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    _fig.update_xaxes(title_text="NLL Per Char", row=1, col=2)

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
