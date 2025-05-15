import marimo

__generated_with = "0.13.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import sklearn as sk
    import matplotlib.pyplot as plt
    from src.data_processing.vae_data_transform import inverse_transform
    import marimo as mo
    return inverse_transform, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r'''
    # Data Pipeline Evaluation
        This Notebook's Purpose is to assess the Data Piepline for the processing of the VAE data that was provided by Jack O'Brian. The config file holds the Latent and Parameter Values of the Variational Auto Encoder. Mapped to it, we have the VAE simulations for which the model was trained on. Further information on how this data was produced can be found in the Dalek paper 
    ''')
    return


@app.cell
def _(mo):
    mo.md(r"""## Loading in the Data Files from our Pipeline""")
    return


@app.cell
def _(pd):
    X_data = pd.read_feather(r"data\processed\predictor\x_data.feather")
    y_data = pd.read_feather(r"data\processed\target\y_data.feather")
    return X_data, y_data


@app.cell
def _(X_data):
    X_data.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inverse Transform

    The inverse transformation for \( V \) and \( W \) is defined as:

    $$
    \begin{aligned}
    V &= \exp \left( (V_{\text{std}} + \mu_v) \cdot \sigma \right) \\
    W &= \exp \left( (W_{\text{std}} + \mu_w) \cdot \sigma \right)
    \end{aligned}
    $$

    where:

    - \( V_{\text{std}} \) and \( W_{\text{std}} \) are the standardized values,

    - \( \mu_v \) and \( \mu_w \) are the mean values,

    - \( \sigma \) represents the scaling factor.
    """)
    return


@app.cell
def _(inverse_transform, y_data):
    v_origingal, w_original = inverse_transform(
        v_scaled=y_data.iloc[:,20:], 
        w_scaled=y_data.iloc[:,:20], 
        scaler_path="scalers.pkl"
    )
    return v_origingal, w_original


@app.cell
def _(mo, v_origingal):
    slider = mo.ui.slider(0, v_origingal.shape[0], step=1)
    return (slider,)


@app.cell
def _(mo, slider):
    mo.md(f"Data at index {slider.value}   {slider}")
    return


@app.cell
def _(np, plt, slider, v_origingal, w_original, y_data):
    x_array = np.arange(0,20, 1)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,5))
    ax[0][0].plot(v_origingal[slider.value,:])
    ax[0][0].set_xlabel("V_inner")
    ax[0][0].set_title("V_inner Unscaled")
    ax[0][1].plot(w_original[slider.value,:])
    ax[0][1].set_title("Dilution Factor Unscaled")
    ax[0][1].set_xlabel("Dilution Factor")
    ax[1][0].plot(x_array, y_data.iloc[slider.value,20:])
    ax[1][0].set_xlabel("V_inner")
    ax[1][0].set_title("V_inner Scaled")

    ax[1][1].plot(x_array, y_data.iloc[slider.value,:20])
    ax[1][1].set_title("Dilution Factor Scaled")
    ax[1][1].set_xlabel("Dilution Factor")

    fig.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
