import pandas as pd

from sklearn.model_selection import train_test_split

def split_data_label_dist(df : pd.DataFrame, label_cols : list[str], val_frac=0.1, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Split Data accounting for label distribution (Not all rows have the same non Nan Label)    
    Algo:
    - Assign a group to each nan/label pattern
    - Split each group into train / val
    '''
    df = df.copy()
    
    # Create a tuple mask for each row indicating which labels are present
    df["_nan_pattern"] = df[label_cols].notna().apply(lambda row: tuple(row.astype(int)), axis=1)

    train_parts = []
    val_parts = []

    for pattern, group in df.groupby("_nan_pattern"):
        if len(group) < 2:
            # If group too small to split, assign to train
            train_parts.append(group)
            continue
        
        train, val = train_test_split(
            group,
            test_size=val_frac,
            random_state=random_state,
            shuffle=True
        )
        train_parts.append(train)
        val_parts.append(val)

    # Combine all pieces
    train_df = pd.concat(train_parts).drop(columns=["_nan_pattern"]).reset_index(drop=True)
    val_df = pd.concat(val_parts).drop(columns=["_nan_pattern"]).reset_index(drop=True)

    return train_df, val_df