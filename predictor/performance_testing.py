#!/usr/bin/python3

import asyncio
import logging
import math
import pandas as pd
from datetime import datetime, timedelta
import os

import model.pricepredictor as pred
from model.priceregion import PriceRegion, PriceRegionName


START: datetime = datetime.fromisoformat("2025-09-01T00:00:00Z")
END: datetime = datetime.fromisoformat("2026-09-01T00:00:00Z")
CROSS_REGIONS = [
    PriceRegionName.DE,
    PriceRegionName.AT,
    PriceRegionName.BE,
    PriceRegionName.NL,
    PriceRegionName.SE1,
    PriceRegionName.SE2,
    PriceRegionName.SE3,
    PriceRegionName.SE4,
    PriceRegionName.DK1,
    PriceRegionName.DK2,
    PriceRegionName.ES,
    PriceRegionName.PT,
]

# Regions to evaluate (get metrics for). Usually a subset of CROSS_REGIONS for debugging.
EVAL_REGIONS = CROSS_REGIONS

LEARN_DAYS : int = 180

PARALLELIZE = True

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)


async def load_data(p : pred.PricePredictor):
    """
    preload data for whole time range to reduce individual http requests
    """
    learn_start = START - timedelta(days=LEARN_DAYS)
    await asyncio.gather(
        p.weatherstore.get_data(learn_start, END),
        p.pricestore.get_data(learn_start, END),
        p.entsoestore.get_data(learn_start, END),
        p.auxstore.get_data(learn_start, END),
        p.gasstore.get_data(learn_start, END)
    )

def mse(df1: pd.Series, df2: pd.Series):
    return (df1 - df2).pow(2).mean()

def mae(df1: pd.Series, df2: pd.Series):
    return (df1 - df2).abs().mean()


# Set to False to fall back to the original single-stage (per-region) evaluation
TWO_STAGE = True


class Metrics:
    """Collects 1d/2d/3d MAE+RMSE over a rolling backtest."""

    def __init__(self):
        self.d1_mae = []
        self.d1_mse = []
        self.d2_mae = []
        self.d2_mse = []
        self.d3_mae = []
        self.d3_mse = []

    def add(self, actual: pd.DataFrame, prediction: pd.DataFrame, d0, d1, d2, d3):
        self.d1_mae.append(mae(actual.loc[d0:d1]["price"], prediction.loc[d0:d1]["price"]))
        self.d2_mae.append(mae(actual.loc[d1:d2]["price"], prediction.loc[d1:d2]["price"]))
        self.d3_mae.append(mae(actual.loc[d2:d3]["price"], prediction.loc[d2:d3]["price"]))

        self.d1_mse.append(mse(actual.loc[d0:d1]["price"], prediction.loc[d0:d1]["price"]))
        self.d2_mse.append(mse(actual.loc[d1:d2]["price"], prediction.loc[d1:d2]["price"]))
        self.d3_mse.append(mse(actual.loc[d2:d3]["price"], prediction.loc[d2:d3]["price"]))

    def summarize(self):
        return (
            round(math.sqrt(sum(self.d1_mse) / len(self.d1_mse)), 2), round(sum(self.d1_mae) / len(self.d1_mae), 2),
            round(math.sqrt(sum(self.d2_mse) / len(self.d2_mse)), 2), round(sum(self.d2_mae) / len(self.d2_mae), 2),
            round(math.sqrt(sum(self.d3_mse) / len(self.d3_mse)), 2), round(sum(self.d3_mae) / len(self.d3_mae), 2),
        )


async def run_many(coros):
    if PARALLELIZE:
        await asyncio.gather(*coros)
    else:
        for c in coros:
            await c



async def main():
    data_dir = os.getenv("EPEXPREDICTOR_DATADIR", "./data")

    # Stage 1: baseline models for ALL cross regions (needed as features for stage-2)
    stage1 = {}
    for name in CROSS_REGIONS:
        region = name.to_region()
        stage1[name] = await pred.PricePredictor(region, data_dir).load_from_persistence()

    # Stage 2: only for eval regions, with stage-1 forecasts of ALL cross regions as features.
    # Each stage-2 predictor shares its data stores with its stage-1 counterpart.
    stage2 = {}
    if TWO_STAGE:
        for name in EVAL_REGIONS:
            region = name.to_region()
            p = pred.PricePredictor(region, data_dir, cross_predictors=[stage1[n] for n in CROSS_REGIONS])
            p.use_datastores_from(stage1[name])
            stage2[name] = p

    # Preload data for all cross regions (stage-2 shares stores with stage-1)
    for name in CROSS_REGIONS:
        await load_data(stage1[name])

    learn_start = START - timedelta(days=LEARN_DAYS)
    learn_end = START

    s1 = {name: Metrics() for name in EVAL_REGIONS}
    s2 = {name: Metrics() for name in EVAL_REGIONS}

    iterations = 0
    while learn_end < END - timedelta(days=3):

        # intervals to predict and check. Could be done nicer but w/e
        d0 = learn_end
        d1 = learn_end + timedelta(days=1)
        d2 = learn_end + timedelta(days=2)
        d3 = learn_end + timedelta(days=3)

        # Make sure training/prediction doesn't "cheat" with data that is known during
        # performance testing, but not for actual forecasts. Must cap ALL cross regions
        # since stage-2 queries their stores via cross-predictors.
        for name in CROSS_REGIONS:
            stage1[name].pricestore.horizon_cutoff = learn_end
            stage1[name].gasstore.horizon_cutoff = learn_end

        # Stage 1: train every cross region's baseline model for this window
        await run_many([stage1[name].train(learn_start, learn_end - timedelta(minutes=15)) for name in CROSS_REGIONS])

        # Stage 2: train every region's model, using the (freshly trained) stage-1 forecasts
        # of all regions as extra features. Must run after all stage-1 models are trained.
        if TWO_STAGE:
            await run_many([stage2[name].train(learn_start, learn_end - timedelta(minutes=15)) for name in EVAL_REGIONS])

        # Predict the upcoming days (cutoffs still at learn_end -> no cheating)
        preds1 = {}
        for name in EVAL_REGIONS:
            preds1[name] = await stage1[name].predict(d0, d3, False)
        preds2 = {}
        if TWO_STAGE:
            for name in EVAL_REGIONS:
                preds2[name] = await stage2[name].predict(d0, d3, False)

        # Fetch actuals (needs full data, so lift the cutoff)
        actuals = {}
        for name in EVAL_REGIONS:
            stage1[name].pricestore.horizon_cutoff = None
            stage1[name].gasstore.horizon_cutoff = None
            actuals[name] = await stage1[name].pricestore.get_data(d0, d3)
            stage1[name].pricestore.horizon_cutoff = learn_end
            stage1[name].gasstore.horizon_cutoff = learn_end

        for name in EVAL_REGIONS:
            s1[name].add(actuals[name], preds1[name], d0, d1, d2, d3)
            if TWO_STAGE:
                s2[name].add(actuals[name], preds2[name], d0, d1, d2, d3)

        learn_start += timedelta(days=1)
        learn_end += timedelta(days=1)
        iterations += 1
        print('.', end='')

    print()

    def print_table(title: str, metrics):
        print(title)
        print("| Region | 1d RMSE | 1d MAE | 2d RMSE | 2d MAE | 3d RMSE | 3d MAE |")
        print("|--------|---------|--------|---------|--------|---------|--------|")
        for name in EVAL_REGIONS:
            d1_rmse, d1_mae, d2_rmse, d2_mae, d3_rmse, d3_mae = metrics[name].summarize()
            print(f"| {name.ljust(5)}  | {str(d1_rmse).ljust(7)} | {str(d1_mae).ljust(6)} | {str(d2_rmse).ljust(7)} | {str(d2_mae).ljust(6)} | {str(d3_rmse).ljust(7)} | {str(d3_mae).ljust(6)} |")
        print()

    print_table(f"Stage 1 (baseline, per-region, iterations={iterations}):", s1)
    if TWO_STAGE:
        print_table(f"Stage 2 (2-stage, + cross-region forecasts, iterations={iterations}):", s2)


asyncio.run(main())
