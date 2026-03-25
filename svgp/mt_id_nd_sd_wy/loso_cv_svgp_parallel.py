"""
Parallel LOSO CV with Stochastic Variational GP (SVGP) for Seasonal Interaction Kernel.
Multi-state version: MT, ID, ND, SD, WY.

Uses all training observations (no subsampling) with mini-batch optimization
of the variational ELBO. Sweeps over different numbers of inducing points
to characterize the accuracy-speed tradeoff.

Usage:
  python loso_cv_svgp_parallel.py [--n_inducing 256,512,1024] [--n_epochs 50] [--batch_size 2048]
"""
import sys
sys.path.insert(0, '../..')
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp

np.random.seed(42)
torch.manual_seed(42)

STATES = ['MT', 'ID', 'ND', 'SD', 'WY']


class SVGPSeasonalInteraction(gpytorch.models.ApproximateGP):
    """SVGP with the same seasonal interaction kernel as the exact GP baseline."""

    def __init__(self, inducing_points,
                 base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx,
                 period_init=None):
        n_inducing = inducing_points.shape[0]
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(n_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        # Base kernel: RBF with ARD over non-seasonal features
        self.base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=len(base_indices),
                active_dims=torch.tensor(base_indices)))

        # Summer kernel: AOT x Periodic(doy) x RBF(doy)
        self.aot_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([aot_idx]))
        self.aot_periodic = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([doy_idx]))
        self.aot_seasonal_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([doy_idx]))
        if period_init is not None:
            self.aot_periodic.initialize(period_length=period_init)
        self.summer_kernel = gpytorch.kernels.ScaleKernel(
            self.aot_rbf * self.aot_periodic * self.aot_seasonal_rbf)

        # Winter kernel: SmogI/P x Periodic(doy) x RBF(doy)
        self.smog_rbf = gpytorch.kernels.RBFKernel(
            ard_num_dims=2, active_dims=torch.tensor([smogI_idx, smogP_idx]))
        self.smog_periodic = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([doy_idx]))
        self.smog_seasonal_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([doy_idx]))
        if period_init is not None:
            self.smog_periodic.initialize(period_length=period_init)
        self.winter_kernel = gpytorch.kernels.ScaleKernel(
            self.smog_rbf * self.smog_periodic * self.smog_seasonal_rbf)

        # Seasonal residual: Periodic(doy) x RBF(doy)
        self.residual_periodic = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([doy_idx]))
        self.residual_seasonal_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([doy_idx]))
        if period_init is not None:
            self.residual_periodic.initialize(period_length=period_init)
        self.seasonal_kernel = gpytorch.kernels.ScaleKernel(
            self.residual_periodic * self.residual_seasonal_rbf)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = (self.base_kernel(x) + self.summer_kernel(x)
                   + self.winter_kernel(x) + self.seasonal_kernel(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def select_inducing_points(X_train, n_inducing, method='kmeans'):
    """Select inducing points via k-means clustering on training data."""
    if n_inducing >= X_train.shape[0]:
        return X_train.clone()

    if method == 'kmeans':
        X_np = X_train.cpu().numpy()
        kmeans = MiniBatchKMeans(
            n_clusters=n_inducing, random_state=42,
            batch_size=min(10000, X_np.shape[0]), n_init=3
        )
        kmeans.fit(X_np)
        return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    else:
        indices = torch.randperm(X_train.shape[0])[:n_inducing]
        return X_train[indices].clone()


def run_fold(args):
    """Run a single LOSO fold with SVGP for multiple inducing point counts."""
    (fold_idx, held_out_site, held_out_state, X_train, y_train_raw, X_test, y_test_raw,
     base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx,
     inducing_list, n_epochs, batch_size, gpu_id, patience, lr) = args

    device = torch.device(f'cuda:{gpu_id}')
    torch.manual_seed(42 + fold_idx)
    np.random.seed(42 + fold_idx)

    y_train = np.log(y_train_raw + 1)
    y_test = np.log(y_test_raw + 1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    period_init = 365.25 / scaler.scale_[doy_idx]

    train_x = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    test_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    n_train = len(train_x)

    # Results for each inducing point count
    inducing_results = []

    for n_inducing in inducing_list:
        torch.manual_seed(42 + fold_idx)
        np.random.seed(42 + fold_idx)

        # Select inducing points
        ip_start = time.perf_counter()
        inducing_points = select_inducing_points(train_x, n_inducing).to(device)
        ip_time = time.perf_counter() - ip_start

        # Build model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = SVGPSeasonalInteraction(
            inducing_points,
            base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx, period_init
        ).to(device)

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=lr)

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_train)

        # DataLoader for mini-batch training
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=0
        )

        # Training loop
        losses = []
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
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad()
                output = model(batch_x)
                loss = -mll(output, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            smoothed_loss = avg_loss if smoothed_loss is None else 0.9 * smoothed_loss + 0.1 * avg_loss

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

        # Inference
        model.eval()
        likelihood.eval()
        infer_start = time.perf_counter()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(test_x))
            pred_mean = pred.mean.cpu().numpy()
            pred_var = pred.variance.cpu().numpy()
        infer_time = time.perf_counter() - infer_start

        # Metrics (log scale)
        rmse_log = np.sqrt(np.mean((pred_mean - y_test)**2))
        mae_log = np.mean(np.abs(pred_mean - y_test))
        ss_res = np.sum((y_test - pred_mean)**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        r2_log = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')

        # Metrics (original scale)
        pred_pm25 = np.exp(pred_mean) - 1
        actual_pm25 = np.exp(y_test) - 1
        rmse_orig = np.sqrt(np.mean((pred_pm25 - actual_pm25)**2))
        mae_orig = np.mean(np.abs(pred_pm25 - actual_pm25))
        ss_res_o = np.sum((actual_pm25 - pred_pm25)**2)
        ss_tot_o = np.sum((actual_pm25 - np.mean(actual_pm25))**2)
        r2_orig = 1 - (ss_res_o / ss_tot_o) if ss_tot_o > 0 else float('nan')

        # Kernel params
        params = {
            'base_scale': model.base_kernel.outputscale.item(),
            'summer_scale': model.summer_kernel.outputscale.item(),
            'winter_scale': model.winter_kernel.outputscale.item(),
            'seasonal_scale': model.seasonal_kernel.outputscale.item(),
            'aot_period_days': model.aot_periodic.period_length.item() * scaler.scale_[doy_idx],
            'smog_period_days': model.smog_periodic.period_length.item() * scaler.scale_[doy_idx],
            'residual_period_days': model.residual_periodic.period_length.item() * scaler.scale_[doy_idx],
            'noise': likelihood.noise.item(),
        }

        inducing_results.append({
            'n_inducing': n_inducing,
            'pred_mean': pred_mean.tolist(),
            'pred_var': pred_var.tolist(),
            'losses': losses,
            'params': params,
            'metrics': {
                'rmse_log': rmse_log,
                'mae_log': mae_log,
                'r2_log': r2_log,
                'rmse_orig': rmse_orig,
                'mae_orig': mae_orig,
                'r2_orig': r2_orig,
            },
            'timing': {
                'ip_selection_time': ip_time,
                'train_time': train_time,
                'infer_time': infer_time,
                'total_fold_time': ip_time + train_time + infer_time,
                'stopped_epoch': stopped_epoch,
            },
        })

        del model, likelihood
        torch.cuda.empty_cache()

    del train_x, train_y, test_x
    torch.cuda.empty_cache()

    return {
        'fold_idx': fold_idx,
        'site': held_out_site,
        'state': held_out_state,
        'n_train': n_train,
        'n_test': len(y_test),
        'y_test': y_test.tolist(),
        'y_test_raw': y_test_raw.tolist(),
        'inducing_results': inducing_results,
        'gpu_id': gpu_id,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=8)
    parser.add_argument('--n_inducing', type=str, default='128,256,512,1024',
                        help='Comma-separated list of inducing point counts to sweep')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    inducing_list = [int(x) for x in args.n_inducing.split(',')]

    print(f"Configuration: n_gpus={args.n_gpus}, n_epochs={args.n_epochs}, "
          f"batch_size={args.batch_size}, patience={args.patience}, lr={args.lr}")
    print(f"States: {STATES}")
    print(f"Inducing point sweep: {inducing_list}")
    print(f"Available CUDA devices: {torch.cuda.device_count()}")

    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    print(f"Using {n_gpus} GPUs")

    # Load data
    pm_all = pd.read_csv("../../data/pm25_data_complete_2003_2021_smogI_031026.csv",
                         low_memory=False)
    pm_fixed = pd.read_csv('../../eda/pm25_locs_with_states.csv')

    # Filter to target states
    state_sites = pm_fixed[pm_fixed['state'].isin(STATES)].copy()
    state_ll_ids = set(state_sites['ll_id'].values)

    pm_all['date'] = pd.to_datetime(pm_all['date'], format='%Y%m%d')
    pm_all['year'] = pm_all['date'].dt.year
    pm_region = pm_all[(pm_all['ll_id'].isin(state_ll_ids)) &
                       (pm_all['year'].isin([2018, 2019]))].copy()

    time_varying_features = ['aot', 'wind', 'hgt', 'cld', 'longwave', 'rh', 'tmax', 'smogI', 'smogP']
    static_features = ['lat', 'lon', 'logpd2500g', 'minf_5000', 'sd50k',
                       'heavy_industrial_ind1', 'housing']

    available_tv = [f for f in time_varying_features if f in pm_region.columns]
    available_static = [f for f in static_features if f in state_sites.columns]

    pm_subset = pm_region[['ll_id', 'date', 'pm25'] + available_tv].copy()
    static_df = state_sites[['ll_id', 'state'] + available_static].copy()
    df = pm_subset.merge(static_df, on='ll_id', how='left')
    df['day_of_year'] = df['date'].dt.dayofyear

    feature_cols = available_tv + available_static + ['day_of_year']
    aot_idx = feature_cols.index('aot')
    smogI_idx = feature_cols.index('smogI')
    smogP_idx = feature_cols.index('smogP')
    doy_idx = feature_cols.index('day_of_year')
    seasonal_interaction_features = {'aot', 'smogI', 'smogP', 'day_of_year'}
    base_indices = [i for i, f in enumerate(feature_cols) if f not in seasonal_interaction_features]

    df_clean = df.dropna(subset=feature_cols + ['pm25']).copy()
    sites = df_clean['ll_id'].unique()

    # Build site-to-state mapping
    site_state_map = df_clean.groupby('ll_id')['state'].first().to_dict()

    print(f"\n{len(df_clean):,} observations, {len(sites)} sites")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Per-state counts
    state_counts = df_clean.groupby('state').agg(
        n_obs=('pm25', 'count'),
        n_sites=('ll_id', 'nunique')
    )
    for st, row in state_counts.iterrows():
        print(f"  {st}: {row['n_sites']} sites, {row['n_obs']:,} observations")

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
        gpu_id = i % n_gpus

        fold_args.append((
            i, held_out_site, site_state_map[held_out_site],
            X_train, y_train_raw, X_test, y_test_raw,
            base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx,
            inducing_list, args.n_epochs, args.batch_size, gpu_id,
            args.patience, args.lr
        ))

    # Run folds in parallel
    print(f"\nLaunching {len(fold_args)} folds across {n_gpus} GPUs...")
    print(f"Each fold trains {len(inducing_list)} SVGP models (inducing sweep)")
    cv_start = time.perf_counter()

    results = []
    with mp.Pool(processes=n_gpus, maxtasksperchild=1) as pool:
        async_results = []
        for fa in fold_args:
            ar = pool.apply_async(run_fold, (fa,))
            async_results.append((fa[1], ar))

        for site, ar in async_results:
            try:
                result = ar.get(timeout=1200)
                results.append(result)
                # Print summary for best inducing point count
                best = max(result['inducing_results'], key=lambda r: r['n_inducing'])
                r2 = best['metrics']['r2_log']
                gpu = result['gpu_id']
                st = result['state']
                tt = best['timing']['train_time']
                ep = best['timing']['stopped_epoch']
                print(f"  Fold {result['fold_idx']:3d} (GPU {gpu}) [{st}] site={result['site']}: "
                      f"R²={r2:.3f} (M={best['n_inducing']}), "
                      f"epochs={ep}, train={tt:.1f}s")
            except Exception as e:
                print(f"  Fold for site {site} FAILED: {e}")

    cv_total_time = time.perf_counter() - cv_start
    print(f"\nTotal CV time: {cv_total_time:.1f}s ({cv_total_time/60:.1f} min)")

    results.sort(key=lambda r: r['fold_idx'])

    # Aggregate results per inducing point count
    print(f"\n{'='*70}")
    print(f"SVGP LOSO CV Results ({'/'.join(STATES)}, {n_gpus} GPUs)")
    print(f"n_epochs={args.n_epochs}, batch_size={args.batch_size}, "
          f"patience={args.patience}, lr={args.lr}")
    print(f"{'='*70}")

    summary_rows = []
    for n_ind in inducing_list:
        all_preds = []
        all_actuals = []
        all_preds_raw = []
        all_actuals_raw = []
        all_train_times = []
        all_infer_times = []
        all_ip_times = []
        all_total_times = []
        all_epochs = []

        for r in results:
            y_test = np.array(r['y_test'])
            y_test_raw = np.array(r['y_test_raw'])
            for ir in r['inducing_results']:
                if ir['n_inducing'] == n_ind:
                    pred = np.array(ir['pred_mean'])
                    all_preds.append(pred)
                    all_actuals.append(y_test)
                    all_preds_raw.append(np.exp(pred) - 1)
                    all_actuals_raw.append(y_test_raw)
                    all_train_times.append(ir['timing']['train_time'])
                    all_infer_times.append(ir['timing']['infer_time'])
                    all_ip_times.append(ir['timing']['ip_selection_time'])
                    all_total_times.append(ir['timing']['total_fold_time'])
                    all_epochs.append(ir['timing']['stopped_epoch'])

        preds = np.concatenate(all_preds)
        actuals = np.concatenate(all_actuals)
        preds_raw = np.concatenate(all_preds_raw)
        actuals_raw = np.concatenate(all_actuals_raw)

        rmse_log = np.sqrt(np.mean((preds - actuals)**2))
        mae_log = np.mean(np.abs(preds - actuals))
        ss_res = np.sum((actuals - preds)**2)
        ss_tot = np.sum((actuals - np.mean(actuals))**2)
        r2_log = 1 - (ss_res / ss_tot)

        rmse_orig = np.sqrt(np.mean((preds_raw - actuals_raw)**2))
        mae_orig = np.mean(np.abs(preds_raw - actuals_raw))
        ss_res_o = np.sum((actuals_raw - preds_raw)**2)
        ss_tot_o = np.sum((actuals_raw - np.mean(actuals_raw))**2)
        r2_orig = 1 - (ss_res_o / ss_tot_o)

        mean_train = np.mean(all_train_times)
        mean_infer = np.mean(all_infer_times)
        mean_ip = np.mean(all_ip_times)
        mean_total = np.mean(all_total_times)
        total_wall = sum(all_total_times)
        mean_epochs = np.mean(all_epochs)

        summary_rows.append({
            'n_inducing': n_ind,
            'rmse_log': rmse_log, 'mae_log': mae_log, 'r2_log': r2_log,
            'rmse_orig': rmse_orig, 'mae_orig': mae_orig, 'r2_orig': r2_orig,
            'mean_train_time': mean_train, 'mean_infer_time': mean_infer,
            'mean_ip_time': mean_ip, 'mean_fold_time': mean_total,
            'total_wall_time': total_wall, 'mean_stopped_epoch': mean_epochs,
            'n_predictions': len(preds),
        })

        print(f"\n  M={n_ind} inducing points:")
        print(f"    Log scale:  RMSE={rmse_log:.4f}, MAE={mae_log:.4f}, R²={r2_log:.4f}")
        print(f"    Orig scale: RMSE={rmse_orig:.2f}, MAE={mae_orig:.2f}, R²={r2_orig:.4f}")
        print(f"    Timing:     train={mean_train:.2f}s, infer={mean_infer:.3f}s, "
              f"ip={mean_ip:.2f}s (mean/fold)")
        print(f"    Epochs:     mean={mean_epochs:.1f}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('svgp_summary.csv', index=False)

    # Save per-site metrics for each inducing count
    site_rows = []
    for r in results:
        for ir in r['inducing_results']:
            pred = np.array(ir['pred_mean'])
            y_test = np.array(r['y_test'])
            rmse = np.sqrt(np.mean((pred - y_test)**2))
            mae = np.mean(np.abs(pred - y_test))
            ss_res = np.sum((y_test - pred)**2)
            ss_tot = np.sum((y_test - np.mean(y_test))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
            site_rows.append({
                'site': r['site'],
                'state': r['state'],
                'n_inducing': ir['n_inducing'],
                'n_train': r['n_train'],
                'n_test': r['n_test'],
                'rmse_log': rmse,
                'mae_log': mae,
                'r2_log': r2,
                'rmse_orig': ir['metrics']['rmse_orig'],
                'mae_orig': ir['metrics']['mae_orig'],
                'r2_orig': ir['metrics']['r2_orig'],
                'train_time': ir['timing']['train_time'],
                'infer_time': ir['timing']['infer_time'],
                'ip_time': ir['timing']['ip_selection_time'],
                'total_time': ir['timing']['total_fold_time'],
                'stopped_epoch': ir['timing']['stopped_epoch'],
                **{f'param_{k}': v for k, v in ir['params'].items()},
            })
    site_df = pd.DataFrame(site_rows)
    site_df.to_csv('svgp_site_metrics.csv', index=False)

    # Save fold timings
    timing_rows = []
    for r in results:
        for ir in r['inducing_results']:
            timing_rows.append({
                'fold': r['fold_idx'],
                'site': r['site'],
                'state': r['state'],
                'n_inducing': ir['n_inducing'],
                'n_train': r['n_train'],
                'n_test': r['n_test'],
                **ir['timing'],
            })
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv('svgp_fold_timings.csv', index=False)

    # Save predictions (for largest inducing count only, to keep file size manageable)
    max_ind = max(inducing_list)
    all_preds_max = []
    all_actuals_max = []
    for r in results:
        y_test = np.array(r['y_test'])
        for ir in r['inducing_results']:
            if ir['n_inducing'] == max_ind:
                all_preds_max.append(np.array(ir['pred_mean']))
                all_actuals_max.append(y_test)
    preds_max = np.concatenate(all_preds_max)
    actuals_max = np.concatenate(all_actuals_max)
    np.savez('svgp_predictions.npz',
             predictions=preds_max, actuals=actuals_max,
             pred_pm25=np.exp(preds_max) - 1, actual_pm25=np.exp(actuals_max) - 1)

    # Save full results JSON
    output = {
        'config': {
            'inducing_list': inducing_list,
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'patience': args.patience,
            'lr': args.lr,
            'states': STATES,
            'n_gpus': n_gpus,
            'training': 'svgp_minibatch',
        },
        'summary': summary_rows,
        'total_cv_time': cv_total_time,
    }

    with open('svgp_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Save per-fold losses for each inducing count
    fold_losses = {}
    for r in results:
        for ir in r['inducing_results']:
            key = f"{r['site']}_M{ir['n_inducing']}"
            fold_losses[key] = ir['losses']

    with open('svgp_fold_losses.json', 'w') as f:
        json.dump(fold_losses, f)

    print(f"\nResults saved to svgp_*.{{json,csv,npz}}")
    print(f"Total wall time: {cv_total_time:.1f}s ({cv_total_time/60:.1f} min)")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
