# financial

Kernel-weighted trend and covariance estimation for a five-asset index-fund
portfolio, with efficient-frontier and tangency-portfolio analysis.

**Live site:** https://cabouman.github.io/financial/ — the portfolio analysis,
plus dated market notes under [Notes](https://cabouman.github.io/financial/posts.html)
(each note is a standalone HTML page linked from posts.html).

## What this is

A statistical decision aid, not investment advice. For five assets — Cash
(3-month T-bills standing in for a money-market fund), US-stock (FSKAX),
US-value (FLCOX), exUS-stock (FTIHX), and US-bond (FXNAX) — it estimates:

- the annual continuously-compounded return trend `b_H` from a weighted
  least-squares fit to log total-return wealth, using an exponential kernel
  with lookback H ∈ {5, 10, 20} years (τ = H/ln 20, so age-H data gets 5%
  of today's weight);
- the one-year log-return covariance `C_H` from overlapping detrended annual
  log returns under the same kernel;
- ordinary-return moments via the lognormal conversion, long-only tangency
  portfolios, and full five-asset efficient frontiers with cash investable;
- uncertainty on everything via a circular block bootstrap (24-month blocks,
  500 replicates, full estimator re-run per replicate).

The headline finding: the bootstrap 10–90% interval for the tangency
portfolio's US-stock weight spans 0–100% at every lookback — the optimal
*composition* is statistically unidentified from historical data — and at
recent lookbacks cash fully replaces bonds as the defensive asset.

## Reproducing the results

```bash
python3 -m venv env && source env/bin/activate
pip install yfinance pandas numpy scipy scikit-learn

python fetch_data.py        # downloads fund + T-bill data, builds spliced series in data/
python estimate.py          # b_H, C_H, tangency portfolios, bootstrap -> results/
python frontier_cash.py     # five-asset frontiers with cash investable -> results/
python export_web_data.py   # bundles results/web_data.json for the page
python -c "t=open('kernel_report.template.html').read(); d=open('results/web_data.json').read(); open('index.html','w').write(t.replace('__DATA__', d))"
```

Data sources: Yahoo Finance adjusted closes (total return, distributions
reinvested) and the FRED TB3MS series. Fund histories shorter than the
lookback are backfilled with long-lived index-fund proxies (VTSMX, VIVAX,
VGTSX, VBMFX), spliced in log-return space and labeled as such in the report.

## Disclaimer

This is an estimation exercise on public data. It is not investment advice,
and past returns do not identify expected future returns — quantifying that
non-identifiability is, in fact, the main result.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
