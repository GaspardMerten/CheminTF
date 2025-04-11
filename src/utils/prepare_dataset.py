import os
import random

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def prepare_dataset(
    data_folder: str = "data",
    output_folder: str = "prepared_data",
    scale: bool = True,
    training_ratio: float = 0.8,
    split: int = 1000,
):
    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "test")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for file in os.listdir(data_folder):
        print(file)
        parquet_file_name = file.replace(".csv", "{index}.parquet")
        if file.endswith(".csv") and not os.path.exists(
            os.path.join(train_folder, parquet_file_name.format(index=0))
        ):
            file_path = os.path.join(data_folder, file)
            df = pl.read_csv(
                file_path,
                has_header=False,
                separator=";"
            )
            lat_column = "column_4"
            lon_column = "column_5"
            base_date_time_column = "column_1"
            mmsi_column = "column_3"

            df = df.with_columns(
                LAT=pl.col(lat_column).str.replace(',', '.').cast(pl.Float64) / 90,
                LON=pl.col(lon_column).str.replace(',', '.').cast(pl.Float64) / 180,
                timestamp=(
                    pl.col(base_date_time_column)
                    .str.strptime(pl.Datetime)
                    .dt.timestamp(time_unit="ms")
                    // 1e3  # Convert to seconds
                    % 86400  # Get seconds of the day
                    / (1 if not scale else 86400)  # Scale to [0, 1] if scale is True
                ),
                day_of_week=pl.col(base_date_time_column)
                .str.strptime(pl.Datetime)
                .dt.weekday() / 6,  # Scale to [0, 1]
            )
            # keep only the columns we need
            df = df.select(["LAT", "LON", "timestamp", "day_of_week", mmsi_column])

            # Save the processed DataFrame to parquet, index by MMSI
            output_file = os.path.join(data_folder, parquet_file_name)
            train_trajectories = []
            test_trajectories = []
            print("unique mmsi", df[mmsi_column].n_unique())
            for mmsi, group in df.group_by(mmsi_column):
                group = group.drop(mmsi_column)
                if len(group) < 10:
                    continue
                trajectory = []

                for row in group.iter_rows(named=True):
                    trajectory += [
                        row["LAT"],
                        row["LON"],
                        row["timestamp"],
                        float(row["day_of_week"]),
                    ]

                if random.random() < training_ratio:
                    train_trajectories.append(trajectory)
                else:
                    test_trajectories.append(trajectory)

            save_trajectories_to_parquet(
                train_trajectories, train_folder, parquet_file_name, split
            )
            save_trajectories_to_parquet(
                test_trajectories, test_folder, parquet_file_name, split
            )


def save_trajectories_to_parquet(trajectories, output_folder, parquet_file_name, split):
    # create files with each max split size
    for i in range(0, len(trajectories), split):
        chunk = trajectories[i : i + split]
        chunk_file_name = parquet_file_name.format(index=i // split)
        table = pa.Table.from_pydict(
            {
                "trajectory": chunk,
            },
            schema=pa.schema(
                [
                    ("trajectory", pa.list_(pa.float64())),
                ]
            ),
        )
        pq.write_table(
            table,
            os.path.join(output_folder, chunk_file_name),
            compression="snappy",
        )


if __name__ == "__main__":
    prepare_dataset(
        data_folder="data",
        output_folder="prepared_data",
        scale=True,
    )
