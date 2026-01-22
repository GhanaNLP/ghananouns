import google.generativeai as genai
import pandas as pd
import time
from pathlib import Path

# Initialize the Gemini client
genai.configure(api_key="YOUR-GEMINI-API-KEY-HERE")  # Replace with your actual API key

def translate_batch_to_twi(nouns):
    """
    Translate a batch of nouns to Twi using Gemini API
    """
    try:
        # Create the model
        model = genai.GenerativeModel('gemini-2.5-pro') #gemini-2.5-pro #gemini-3-flash-preview
        
        # Create prompt with numbered words
        words_text = "\n".join([f"{i+1}. {noun}" for i, noun in enumerate(nouns)])
        
        prompt = f"""Translate these list of nouns extracted from Ghanaian news articles to Twi. Return ONLY the translations in the same numbered format, one per line. Do not add any explanations or extra text.

{words_text}"""
        
        # Generate translation
        response = model.generate_content(prompt)
        translation_text = response.text.strip()
        
        # Parse the response back into a list
        translations = []
        lines = translation_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove numbering if present (e.g., "1. ", "1) ", "1- ")
            import re
            cleaned = re.sub(r'^\d+[\.\)\-]\s*', '', line)
            translations.append(cleaned)
        
        # Ensure we have the same number of translations as inputs
        if len(translations) != len(nouns):
            print(f"Warning: Expected {len(nouns)} translations but got {len(translations)}")
            # Pad with None if we got fewer translations
            while len(translations) < len(nouns):
                translations.append(None)
        
        return translations
    
    except Exception as e:
        print(f"Error translating batch: {e}")
        return [None] * len(nouns)

def main():
    # File paths
    input_file = "nouns_all_en.csv"
    output_file = "nouns_with_twi_translations.csv"
    
    # Batch size
    BATCH_SIZE = 200
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} nouns from {input_file}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Assuming the noun column is the first column
    noun_column = df.columns[0]
    
    # Check if output file exists for resume capability
    start_idx = 0
    if Path(output_file).exists():
        print(f"\nFound existing output file: {output_file}")
        existing_df = pd.read_csv(output_file)
        
        # Check if twi_translation column exists
        if 'twi_translation' in existing_df.columns:
            # Find the last non-null translation
            non_null_mask = existing_df['twi_translation'].notna()
            if non_null_mask.any():
                start_idx = non_null_mask.sum()
                print(f"Resuming from index {start_idx} (already translated {start_idx} nouns)")
                df = existing_df
            else:
                print("No translations found in existing file, starting from beginning")
                df['twi_translation'] = None
        else:
            print("No translation column found, starting from beginning")
            df['twi_translation'] = None
    else:
        print("No existing output file found, starting fresh")
        df['twi_translation'] = None
    
    # Process nouns in batches
    total_batches = (len(df) - start_idx + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        batch_start = start_idx + (batch_num * BATCH_SIZE)
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        
        # Get batch of nouns
        batch_nouns = df.loc[batch_start:batch_end-1, noun_column].tolist()
        
        print(f"\nProcessing batch {batch_num + 1}/{total_batches} (indices {batch_start} to {batch_end-1})")
        print(f"Translating {len(batch_nouns)} words...")
        
        # Translate the batch
        translations = translate_batch_to_twi(batch_nouns)
        
        # Assign translations back to dataframe
        for i, translation in enumerate(translations):
            df.loc[batch_start + i, 'twi_translation'] = translation
        
        # Save progress after each batch
        df.to_csv(output_file, index=False)
        print(f"✓ Batch {batch_num + 1} completed and saved")
        
        # Add a delay between batches to avoid rate limiting
        if batch_num < total_batches - 1:  # Don't sleep after the last batch
            time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"Translation complete! Results saved to {output_file}")
    successful = df['twi_translation'].notna().sum()
    print(f"Successfully translated: {successful}/{len(df)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
