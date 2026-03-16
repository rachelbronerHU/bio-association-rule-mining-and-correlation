import os
import pandas as pd

def add_organ_column(df, mibi_gut_dir_path):
    """
    Standardized helper to add an 'Organ' column to a dataframe.
    Looks up the Localization from biopsy_metadata.csv based on Biopsy_ID or Patient,
    and falls back to parsing the Cohort string for Colon/Duodenum.
    
    Args:
        df (pd.DataFrame): DataFrame containing at least 'Biopsy_ID' or 'Patient' and optionally 'Cohort'.
        mibi_gut_dir_path (str): Path to the MIBIGutCsv directory containing biopsy_metadata.csv.
        
    Returns:
        pd.DataFrame: A copy of the DataFrame with an 'Organ' column populated.
    """
    df = df.copy()
    
    # Try to load biopsy metadata
    biopsy_meta_path = os.path.join(mibi_gut_dir_path, "biopsy_metadata.csv")
    if os.path.exists(biopsy_meta_path):
        try:
            biopsy_df = pd.read_csv(biopsy_meta_path)
            if "Localization" in biopsy_df.columns:
                # Some scripts use 'Biopsy_ID', others use 'Patient' to map to Biopsy_ID in metadata
                merge_key = 'Biopsy_ID' if 'Biopsy_ID' in df.columns else ('Patient' if 'Patient' in df.columns else None)
                
                if merge_key:
                    # Create a mapping dictionary to be safe against duplicates
                    mapping = biopsy_df.set_index('Biopsy_ID')['Localization'].to_dict()
                    df['Organ'] = df[merge_key].map(mapping)
        except Exception as e:
            print(f"Warning: Could not process {biopsy_meta_path} for Organ mapping: {e}")

    # Fallback to parsing the Cohort column if Organ is still missing or has NaNs
    if 'Organ' not in df.columns:
        df['Organ'] = None
        
    if 'Cohort' in df.columns:
        df['Organ'] = df['Organ'].fillna(
            df['Cohort'].apply(lambda x: 'Colon' if 'Colon' in str(x) else ('Duodenum' if 'Duodenum' in str(x) else 'Unknown'))
        )
    
    # Final fallback if both metadata and Cohort failed/were missing
    df['Organ'] = df['Organ'].fillna('Unknown')
    
    return df
