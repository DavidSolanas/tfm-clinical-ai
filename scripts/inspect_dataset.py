import argparse
import json

import pyarrow.ipc as ipc
import pyarrow as pa


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect the dataset stored in Arrow format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="data/processed/dataset/validation/data-00000-of-00001.arrow",
        help="Path to the Arrow file to inspect",
    )
    return p


def _read_arrow_table(path: str) -> pa.Table:
    """Load an .arrow file. Supports both stream and file formats."""
    with pa.memory_map(path, "r") as source:
        try:
            return ipc.open_stream(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return ipc.open_file(source).read_all()


def main() -> None:
    args = _build_parser().parse_args()
    dataset = args.dataset

    table = _read_arrow_table(dataset)
    print("Schema:", table.schema)

    df = table.to_pandas()
    print(df.head())
    print(df.iloc[0])
    for message in df.iloc[0]['messages']:
        print(json.dumps(message, indent=4))


if __name__ == "__main__":
    main()
