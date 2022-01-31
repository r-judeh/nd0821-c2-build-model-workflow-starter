#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import tempfile

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info("Create a W&B run")

    with wandb.init(job_type="basic_cleaning") as run:
        run.config.update(args)

        # Download input artifact.
        logger.info(f"Download {args.input_artifact} ...")

        artifact_local_path = run.use_artifact(args.input_artifact).file()
        df = pd.read_csv(artifact_local_path)

        # Drop outliers
        logger.info("Drop outliers")

        idx = df['price'].between(args.min_price, args.max_price)
        df = df[idx].copy()

        # Convert the last_review column to datetime
        logger.info("Convert last_review column to datetime")

        df['last_review'] = pd.to_datetime(df['last_review'])

        # Save the results
        logger.info("Save the new dataframe to clean_sample.csv")

        df.to_csv("clean_sample.csv", index=False)

        # Create an artifact and upload it to W&B
        logger.info(f"Upload artifact {args.output_artifact} to W&B")

        artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )

        artifact.add_file("clean_sample.csv")
        run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="The name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="The type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="A description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="The minimum price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="The maximum price to be considered",
        required=True
    )

    args = parser.parse_args()

    go(args)
