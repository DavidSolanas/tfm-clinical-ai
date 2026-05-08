import pandas as pd
from sklearn.model_selection import train_test_split

_MIN_TRANSCRIPTION_CHARS = 200


def load_mtsamples(
    path: str = "hf://datasets/harishnair04/mtsamples/mtsamples.csv",
) -> pd.DataFrame:
    """Load and clean MTSamples dataset.

    Loads the MTSamples CSV, normalizes text columns (strips whitespace, handles
    missing values), and filters out records with short transcriptions.

    Args:
        path: Dataset path or Hugging Face Hub identifier. Defaults to the
            public MTSamples dataset on HF Hub.

    Returns:
        DataFrame with cleaned records where each transcription has at least
        _MIN_TRANSCRIPTION_CHARS characters. Index is reset.
    """
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "record_id"})
    for col in (
        "transcription",
        "description",
        "medical_specialty",
        "sample_name",
        "keywords",
    ):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["transcription"].str.len() >= _MIN_TRANSCRIPTION_CHARS].reset_index(drop=True)
    return df


def split_by_medical_specialty(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/validation/test stratified by medical specialty.

    Uses stratified random sampling to ensure each medical specialty is
    represented proportionally in all three splits.

    Args:
        df: Input DataFrame with 'medical_specialty' column.
        train_ratio: Fraction of data for training (default 0.80).
        val_ratio: Fraction of data for validation (default 0.10).
            Remaining fraction (1 - train_ratio - val_ratio) is reserved for test.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df), each with reset index.
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=df["medical_specialty"],
    )

    relative_val = val_ratio / (1.0 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - relative_val,
        random_state=seed,
        stratify=temp_df["medical_specialty"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
