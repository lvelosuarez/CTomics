import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.utils import resample
    from sklearn.base import clone
    from sklearn.metrics import (
        r2_score,
        mean_squared_error,
        mean_absolute_error,
    )

    import matplotlib.pyplot as plt

    return (
        SVR,
        clone,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pl,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CTomics – Predict % microbial reads from qPCR Ct values
    ## Upload  csv with columns `ct_16S`, `ct_ACTB`, optional `delta`, and `pct_microbial` to train simple ML models."
    """)
    return


@app.cell
def _(pl):
    DATA_PATH = "../data/data.csv"  # <-- change to your file
    # Load with polars
    df = pl.read_csv(DATA_PATH)
    df = df.with_columns(
        (pl.col("pct_microbial") + 0.01).log().alias("log_pct")
    )
    return (df,)


@app.cell
def _(df, plt):
    plt.scatter(df["delta"], df["log_pct"])
    plt.xlabel("delta (ACTB - 16S)")
    plt.ylabel("log(% microbial + 0.01)")
    plt.show()
    return


@app.cell
def _(df):
    feature_cols = ["ct_16S", "ct_ACTB", "delta"]  
    target_col = "pct_microbial"
    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()
    return X, y


@app.cell
def _(X, train_test_split, y):
    RANDOM_STATE = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,        # 20% test
        random_state=RANDOM_STATE,
    )

    X_train.shape, X_test.shape
    return X_test, X_train, y_test, y_train


@app.cell
def _(
    SVR,
    X_test,
    X_train,
    mean_absolute_error,
    mean_squared_error,
    np,
    r2_score,
    y_test,
    y_train,
):
    # A decent starting point; you can tune C, epsilon, gamma later
    svr = SVR(
        kernel="rbf",
        C=10.0,
        epsilon=0.1,
    )

    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("=== SVR (RBF) ===")
    print(f"R² (test)  : {r2:.3f}")
    print(f"RMSE (test): {rmse:.3f}")
    print(f"MAE  (test): {mae:.3f}")
    return (y_pred,)


@app.cell
def _(plt, y_pred, y_test):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Observed % microbial reads")
    plt.ylabel("Predicted % microbial reads (SVR)")
    plt.title("SVR: Predicted vs observed % microbial reads")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    SVR,
    X_test,
    X_train,
    clone,
    mean_absolute_error,
    mean_squared_error,
    np,
    r2_score,
    y_test,
    y_train,
):
    def bootstrap_svr_metrics(
        base_model,
        X_train,
        y_train,
        X_test,
        y_test,
        n_boot=500,
        random_state=123,
    ):
        rng = np.random.default_rng(random_state)
        n_train = X_train.shape[0]

        r2_vals = []
        rmse_vals = []
        mae_vals = []

        for b in range(n_boot):
            idx = rng.integers(0, n_train, size=n_train)
            Xb = X_train[idx]
            yb = y_train[idx]

            model = clone(base_model)
            model.fit(Xb, yb)
            y_pred_b = model.predict(X_test)

            r2_vals.append(r2_score(y_test, y_pred_b))
            rmse_vals.append(np.sqrt(mean_squared_error(y_test, y_pred_b)))
            mae_vals.append(mean_absolute_error(y_test, y_pred_b))

        r2_vals = np.array(r2_vals)
        rmse_vals = np.array(rmse_vals)
        mae_vals = np.array(mae_vals)

        def ci95(x):
            return np.quantile(x, [0.025, 0.975])

        return {
            "r2": r2_vals,
            "rmse": rmse_vals,
            "mae": mae_vals,
            "r2_ci95": ci95(r2_vals),
            "rmse_ci95": ci95(rmse_vals),
            "mae_ci95": ci95(mae_vals),
        }

    # use the same hyper-params as svr above
    base_svr = SVR(kernel="rbf", C=10.0, epsilon=0.1)

    boot = bootstrap_svr_metrics(
        base_model=base_svr,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_boot=500,
        random_state=123,
    )

    print("SVR bootstrap R²  95% CI:", boot["r2_ci95"])
    print("SVR bootstrap RMSE 95% CI:", boot["rmse_ci95"])
    print("SVR bootstrap MAE  95% CI:", boot["mae_ci95"])
    return (boot,)


@app.cell
def _(boot, plt):
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.hist(boot["r2"], bins=30, density=True)
    plt.title("SVR bootstrap R²")

    plt.subplot(1, 3, 2)
    plt.hist(boot["rmse"], bins=30, density=True)
    plt.title("SVR bootstrap RMSE")

    plt.subplot(1, 3, 3)
    plt.hist(boot["mae"], bins=30, density=True)
    plt.title("SVR bootstrap MAE")

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
