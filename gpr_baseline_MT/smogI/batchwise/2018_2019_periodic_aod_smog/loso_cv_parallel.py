"""
Parallel LOSO CV with Batchwise SGD for Seasonal Interaction Kernel.

Supports two batching strategies:
  --batch_size N    Random sampling of N observations per epoch
  --n_days N        Daily batching: sample N complete days per epoch

Following Patel et al. (2022, AAAI), each epoch sees a fresh batch,
providing implicit regularization via stochastic optimization.

Usage:
  python loso_cv_parallel.py --n_days 1 --n_epochs 1000 --patience 30
  python loso_cv_parallel.py --batch_size 4000 --n_epochs 200 --patience 20
"""
import sys
sys.path.insert(0, '../../..')
import os
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp

np.random.seed(42)
torch.manual_seed(42)


class SeasonalInteractionGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx,
                 period_init=None, product_mode=False):
        super().__init__(train_x, train_y, likelihood)
        self.product_mode = product_mode
        self.mean_module = gpytorch.means.ConstantMean()

        self.base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=len(base_indices),
                active_dims=torch.tensor(base_indices)))

        self.aot_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([aot_idx]))
        self.aot_periodic = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([doy_idx]))
        self.aot_seasonal_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([doy_idx]))
        if period_init is not None:
            self.aot_periodic.initialize(period_length=period_init)
        self.summer_kernel = gpytorch.kernels.ScaleKernel(
            self.aot_rbf * self.aot_periodic * self.aot_seasonal_rbf)

        self.smog_rbf = gpytorch.kernels.RBFKernel(
            ard_num_dims=2, active_dims=torch.tensor([smogI_idx, smogP_idx]))
        self.smog_periodic = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([doy_idx]))
        self.smog_seasonal_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([doy_idx]))
        if period_init is not None:
            self.smog_periodic.initialize(period_length=period_init)
        self.winter_kernel = gpytorch.kernels.ScaleKernel(
            self.smog_rbf * self.smog_periodic * self.smog_seasonal_rbf)

        self.residual_periodic = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([doy_idx]))
        self.residual_seasonal_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([doy_idx]))
        if period_init is not None:
            self.residual_periodic.initialize(period_length=period_init)
        self.seasonal_kernel = gpytorch.kernels.ScaleKernel(
            self.residual_periodic * self.residual_seasonal_rbf)

    def forward(self, x):
        mean_x = self.mean_module(x)
        if self.product_mode:
            covar_x = (self.base_kernel(x) * self.summer_kernel(x)
                       * self.winter_kernel(x) * self.seasonal_kernel(x))
        else:
            covar_x = (self.base_kernel(x) + self.summer_kernel(x)
                       + self.winter_kernel(x) + self.seasonal_kernel(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def run_fold(args):
    """Run a single LOSO fold with batchwise SGD on a specific GPU."""
    (fold_idx, held_out_site, X_train, y_train_raw, X_test, y_test_raw,
     train_date_ordinals,
     base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx,
     batch_size, n_days, n_epochs, gpu_id, patience,
     inference_size, inference_days, product_kernel) = args

    device = torch.device(f'cuda:{gpu_id}')
    torch.manual_seed(42 + fold_idx)
    np.random.seed(42 + fold_idx)

    y_train = np.log(y_train_raw + 1)
    y_test = np.log(y_test_raw + 1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    period_init = 365.25 / scaler.scale_[doy_idx]

    full_train_x = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    full_train_y = torch.tensor(y_train, dtype=torch.float32).to(device)
    test_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    n_train = len(full_train_x)

    # Daily batching setup
    use_daily = n_days is not None and n_days > 0
    if use_daily:
        unique_dates = np.unique(train_date_ordinals)
        date_to_indices = {d: np.where(train_date_ordinals == d)[0] for d in unique_dates}
        n_days_per_batch = min(n_days, len(unique_dates))

    # Get initial batch
    if use_daily:
        init_dates = np.random.choice(unique_dates, n_days_per_batch, replace=False)
        init_idx = np.concatenate([date_to_indices[d] for d in init_dates])
    else:
        actual_batch = min(batch_size, n_train)
        init_idx = torch.randperm(n_train)[:actual_batch].numpy()

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SeasonalInteractionGP(
        full_train_x[init_idx], full_train_y[init_idx], likelihood,
        base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx, period_init,
        product_mode=product_kernel
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    batch_sizes_log = []
    best_smoothed = float('inf')
    best_state = None
    best_lik_state = None
    patience_counter = 0
    smoothed_loss = None
    stopped_epoch = n_epochs

    train_start = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        likelihood.train()

        # Sample batch
        if use_daily:
            sampled_dates = np.random.choice(unique_dates, n_days_per_batch, replace=False)
            batch_idx = np.concatenate([date_to_indices[d] for d in sampled_dates])
        else:
            batch_idx = torch.randperm(n_train)[:actual_batch].numpy()

        batch_x = full_train_x[batch_idx]
        batch_y = full_train_y[batch_idx]
        model.set_train_data(batch_x, batch_y, strict=False)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = -mll(output, batch_y)
        loss.backward()
        optimizer.step()

        current = loss.item()
        losses.append(current)
        batch_sizes_log.append(len(batch_idx))

        # EMA for early stopping
        smoothed_loss = current if smoothed_loss is None else 0.9 * smoothed_loss + 0.1 * current

        if smoothed_loss < best_smoothed:
            best_smoothed = smoothed_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_lik_state = {k: v.clone() for k, v in likelihood.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience > 0 and patience_counter >= patience:
            stopped_epoch = epoch + 1
            break

    train_time = time.perf_counter() - train_start

    # Restore best model
    model.load_state_dict(best_state)
    likelihood.load_state_dict(best_lik_state)

    # Set conditioning data for inference
    if use_daily and inference_days is not None and inference_days > 0:
        n_infer_days = min(inference_days, len(unique_dates))
        infer_dates = np.random.choice(unique_dates, n_infer_days, replace=False)
        infer_idx = np.concatenate([date_to_indices[d] for d in infer_dates])
    else:
        actual_infer = min(inference_size, n_train)
        infer_idx = torch.randperm(n_train)[:actual_infer].numpy()

    model.set_train_data(full_train_x[infer_idx], full_train_y[infer_idx], strict=False)

    # Predict
    infer_start = time.perf_counter()
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        pred_mean = pred.mean.cpu().numpy()
        pred_var = pred.variance.cpu().numpy()
    infer_time = time.perf_counter() - infer_start

    params = {
        'fold': fold_idx,
        'site': held_out_site,
        'base_scale': model.base_kernel.outputscale.item(),
        'summer_scale': model.summer_kernel.outputscale.item(),
        'winter_scale': model.winter_kernel.outputscale.item(),
        'seasonal_scale': model.seasonal_kernel.outputscale.item(),
        'aot_period_days': model.aot_periodic.period_length.item() * scaler.scale_[doy_idx],
        'smog_period_days': model.smog_periodic.period_length.item() * scaler.scale_[doy_idx],
        'residual_period_days': model.residual_periodic.period_length.item() * scaler.scale_[doy_idx],
        'noise': likelihood.noise.item(),
    }

    site_rmse = np.sqrt(np.mean((pred_mean - y_test)**2))
    site_mae = np.mean(np.abs(pred_mean - y_test))
    ss_res = np.sum((y_test - pred_mean)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    site_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')

    del model, likelihood, full_train_x, full_train_y, test_x
    torch.cuda.empty_cache()

    return {
        'fold_idx': fold_idx,
        'site': held_out_site,
        'pred_mean': pred_mean.tolist(),
        'pred_var': pred_var.tolist(),
        'y_test': y_test.tolist(),
        'losses': losses,
        'params': params,
        'metrics': {
            'site': held_out_site,
            'n_obs': len(y_test),
            'rmse_log': site_rmse,
            'mae_log': site_mae,
            'r2_log': site_r2,
        },
        'timing': {
            'fold': fold_idx,
            'site': held_out_site,
            'n_train': n_train,
            'n_test': len(y_test),
            'avg_batch_size': float(np.mean(batch_sizes_log)),
            'train_time': train_time,
            'infer_time': infer_time,
            'fold_time': train_time + infer_time,
            'stopped_epoch': stopped_epoch,
            'n_infer_points': len(infer_idx),
        },
        'gpu_id': gpu_id,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=8)

    # Batching strategy (mutually exclusive in practice)
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Random batch size per epoch (0 = use --n_days instead)')
    parser.add_argument('--n_days', type=int, default=0,
                        help='Number of complete days per training batch (0 = use --batch_size)')

    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience on EMA loss (0 to disable)')

    # Inference conditioning
    parser.add_argument('--inference_size', type=int, default=4000,
                        help='Random conditioning points for inference (used with --batch_size)')
    parser.add_argument('--inference_days', type=int, default=200,
                        help='Complete days for inference conditioning (used with --n_days)')

    # Output prefix
    parser.add_argument('--prefix', type=str, default='parallel',
                        help='Output file prefix')
    parser.add_argument('--product_kernel', action='store_true',
                        help='Use product (multiplicative) kernel instead of additive')
    args = parser.parse_args()

    # Validate: need at least one batching strategy
    if args.batch_size == 0 and args.n_days == 0:
        args.n_days = 1  # default to daily batching

    use_daily = args.n_days > 0
    batch_desc = (f"daily ({args.n_days} day(s)/epoch)" if use_daily
                  else f"random ({args.batch_size} obs/epoch)")

    kernel_type = "product" if args.product_kernel else "additive"
    print(f"Configuration: n_gpus={args.n_gpus}, n_epochs={args.n_epochs}, patience={args.patience}")
    print(f"Kernel: {kernel_type}")
    print(f"Batching: {batch_desc}")
    if use_daily:
        print(f"Inference: {args.inference_days} days")
    else:
        print(f"Inference: {args.inference_size} random points")
    print(f"Available CUDA devices: {torch.cuda.device_count()}")

    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    print(f"Using {n_gpus} GPUs")

    # Load data
    pm_all = pd.read_csv("../../../../data/pm25_data_complete_2003_2021_smogI_031026.csv",
                         low_memory=False)
    pm_fixed = pd.read_csv('../../../../eda/pm25_locs_with_states.csv')

    mt_sites = pm_fixed[pm_fixed['state'] == 'MT'].copy()
    mt_ll_ids = set(mt_sites['ll_id'].values)

    pm_all['date'] = pd.to_datetime(pm_all['date'], format='%Y%m%d')
    pm_all['year'] = pm_all['date'].dt.year
    pm_mt = pm_all[(pm_all['ll_id'].isin(mt_ll_ids)) & (pm_all['year'].isin([2018, 2019]))].copy()

    time_varying_features = ['aot', 'wind', 'hgt', 'cld', 'longwave', 'rh', 'tmax', 'smogI', 'smogP']
    static_features = ['lat', 'lon', 'logpd2500g', 'minf_5000', 'sd50k',
                       'heavy_industrial_ind1', 'housing']

    available_tv = [f for f in time_varying_features if f in pm_mt.columns]
    available_static = [f for f in static_features if f in mt_sites.columns]

    pm_mt_subset = pm_mt[['ll_id', 'date', 'pm25'] + available_tv].copy()
    mt_static = mt_sites[['ll_id'] + available_static].copy()
    df = pm_mt_subset.merge(mt_static, on='ll_id', how='left')
    df['day_of_year'] = df['date'].dt.dayofyear
    df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())

    feature_cols = available_tv + available_static + ['day_of_year']
    aot_idx = feature_cols.index('aot')
    smogI_idx = feature_cols.index('smogI')
    smogP_idx = feature_cols.index('smogP')
    doy_idx = feature_cols.index('day_of_year')
    seasonal_interaction_features = {'aot', 'smogI', 'smogP', 'day_of_year'}
    base_indices = [i for i, f in enumerate(feature_cols) if f not in seasonal_interaction_features]

    df_clean = df.dropna(subset=feature_cols + ['pm25']).copy()
    sites = df_clean['ll_id'].unique()

    n_unique_dates = df_clean['date_ordinal'].nunique()
    avg_per_day = len(df_clean) / n_unique_dates
    print(f"\n{len(df_clean):,} observations, {len(sites)} sites, {n_unique_dates} unique dates")
    if use_daily:
        print(f"Avg {avg_per_day:.1f} obs/day → ~{avg_per_day * args.n_days:.0f} obs/batch")

    # Prepare fold arguments
    fold_args = []
    for i, held_out_site in enumerate(sites):
        test_mask = df_clean['ll_id'] == held_out_site
        train_df = df_clean[~test_mask]
        test_df = df_clean[test_mask]
        if len(test_df) == 0:
            continue

        X_train = train_df[feature_cols].values
        y_train_raw = train_df['pm25'].values
        X_test = test_df[feature_cols].values
        y_test_raw = test_df['pm25'].values
        train_date_ordinals = train_df['date_ordinal'].values if use_daily else None
        gpu_id = i % n_gpus

        fold_args.append((
            i, held_out_site, X_train, y_train_raw, X_test, y_test_raw,
            train_date_ordinals,
            base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx,
            args.batch_size if not use_daily else 0,
            args.n_days if use_daily else 0,
            args.n_epochs, gpu_id, args.patience,
            args.inference_size, args.inference_days if use_daily else 0,
            args.product_kernel
        ))

    # Run folds in parallel
    print(f"\nLaunching {len(fold_args)} folds across {n_gpus} GPUs...")
    cv_start = time.perf_counter()

    results = []
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(run_fold, fa): fa[1] for fa in fold_args}
        for future in as_completed(futures):
            site = futures[future]
            try:
                result = future.result()
                results.append(result)
                r2 = result['metrics']['r2_log']
                gpu = result['gpu_id']
                ep = result['timing']['stopped_epoch']
                ab = result['timing']['avg_batch_size']
                print(f"  Fold {result['fold_idx']:2d} (GPU {gpu}) site={site}: "
                      f"R²={r2:.3f}, epochs={ep}, avg_batch={ab:.0f}, "
                      f"train={result['timing']['train_time']:.1f}s")
            except Exception as e:
                print(f"  Fold for site {site} FAILED: {e}")

    cv_total_time = time.perf_counter() - cv_start
    print(f"\nTotal CV time: {cv_total_time:.1f}s ({cv_total_time/60:.1f} min)")

    results.sort(key=lambda r: r['fold_idx'])

    all_predictions = np.concatenate([np.array(r['pred_mean']) for r in results])
    all_actuals = np.concatenate([np.array(r['y_test']) for r in results])

    rmse_log = np.sqrt(np.mean((all_predictions - all_actuals)**2))
    mae_log = np.mean(np.abs(all_predictions - all_actuals))
    ss_res = np.sum((all_actuals - all_predictions)**2)
    ss_tot = np.sum((all_actuals - np.mean(all_actuals))**2)
    r2_log = 1 - (ss_res / ss_tot)

    pred_pm25 = np.exp(all_predictions) - 1
    actual_pm25 = np.exp(all_actuals) - 1
    rmse_orig = np.sqrt(np.mean((pred_pm25 - actual_pm25)**2))
    mae_orig = np.mean(np.abs(pred_pm25 - actual_pm25))
    ss_res_orig = np.sum((actual_pm25 - pred_pm25)**2)
    ss_tot_orig = np.sum((actual_pm25 - np.mean(actual_pm25))**2)
    r2_orig = 1 - (ss_res_orig / ss_tot_orig)

    print(f"\n{'='*60}")
    print(f"LOSO CV Results ({batch_desc}, {n_gpus} GPUs)")
    print(f"{'='*60}")
    print(f"Log scale:  RMSE={rmse_log:.4f}, MAE={mae_log:.4f}, R²={r2_log:.4f}")
    print(f"Orig scale: RMSE={rmse_orig:.2f}, MAE={mae_orig:.2f}, R²={r2_orig:.4f}")
    print(f"Wall time:  {cv_total_time:.1f}s ({cv_total_time/60:.1f} min)")

    # Save
    metrics_df = pd.DataFrame([r['metrics'] for r in results])
    params_df = pd.DataFrame([r['params'] for r in results])
    timing_df = pd.DataFrame([r['timing'] for r in results])
    fold_losses = {r['site']: r['losses'] for r in results}

    output = {
        'config': {
            'batch_size': args.batch_size, 'n_days': args.n_days,
            'n_epochs': args.n_epochs, 'patience': args.patience,
            'inference_size': args.inference_size,
            'inference_days': args.inference_days if use_daily else 0,
            'n_gpus': n_gpus, 'training': 'daily_batch_sgd' if use_daily else 'random_batch_sgd',
            'kernel': kernel_type,
        },
        'overall': {
            'rmse_log': rmse_log, 'mae_log': mae_log, 'r2_log': r2_log,
            'rmse_orig': rmse_orig, 'mae_orig': mae_orig, 'r2_orig': r2_orig,
            'total_time': cv_total_time, 'n_predictions': len(all_predictions),
        },
        'fold_losses': fold_losses,
    }

    prefix = args.prefix
    with open(f'{prefix}_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    metrics_df.to_csv(f'{prefix}_site_metrics.csv', index=False)
    params_df.to_csv(f'{prefix}_kernel_params.csv', index=False)
    timing_df.to_csv(f'{prefix}_fold_timings.csv', index=False)
    np.savez(f'{prefix}_predictions.npz',
             predictions=all_predictions, actuals=all_actuals,
             pred_pm25=pred_pm25, actual_pm25=actual_pm25)

    print(f"\nResults saved to {prefix}_results.json, {prefix}_site_metrics.csv, etc.")

    stopped_epochs = [r['timing']['stopped_epoch'] for r in results]
    print(f"\nEarly stopping summary (patience={args.patience}, EMA α=0.1):")
    print(f"  Stopped epochs: min={min(stopped_epochs)}, max={max(stopped_epochs)}, "
          f"mean={np.mean(stopped_epochs):.1f}")

    max_len = max(len(v) for v in fold_losses.values())
    loss_matrix = np.full((len(fold_losses), max_len), np.nan)
    for i, losses in enumerate(fold_losses.values()):
        loss_matrix[i, :len(losses)] = losses
    mean_loss = np.nanmean(loss_matrix, axis=0)
    min_epoch = np.nanargmin(mean_loss) + 1
    print(f"  Mean loss at epoch 1: {mean_loss[0]:.4f}")
    print(f"  Minimum mean loss at epoch {min_epoch}: {mean_loss[min_epoch-1]:.4f}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
