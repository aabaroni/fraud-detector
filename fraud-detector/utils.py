import pandas as pd

def load_data_local(input_path: str, input_type: str):

    if input_type == "csv":
        try:
            data = pd.read_csv(input_path)
        except:
            raise ValueError("File not found")
    else:
        raise ValueError("Wrong file type specified. Currently supported: 'csv'.")

    return data

