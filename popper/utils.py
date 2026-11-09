import pandas as pd
import numpy as np
import os
import json
import time
import openai
from glob import glob
import numpy as np
import pandas as pd

from popper.llm.custom_model import CustomChatModel
from popper.llm.compatible_model import CompatibleChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.utils.interactive_env import is_interactive_env

def get_llm(model = 'claude-3-5-sonnet-20240620', temperature=0.7, port=30000, api_key = "EMPTY", base_url=None, **kwargs):
    """An explicit base_url selects an OpenAI-compatible service for any model name."""
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty URL or None")
        endpoint_key = api_key if api_key not in (None, "", "EMPTY") else os.environ.get("OPENAI_API_KEY") or "EMPTY"
        return CompatibleChatOpenAI(model=model, base_url=base_url, api_key=endpoint_key,
                          temperature=temperature, **kwargs)

    source = "Local"
    if model[:7] == 'claude-':
        source = 'Anthropic'
    elif model[:4] == 'gpt-' or model.startswith("o1"):
        source = 'OpenAI'
    # elif model.startswith('llama'):
    #     source = "Llama"
    # if source not in ['OpenAI', 'Anthropic']:
    #     raise ValueError('Invalid source')
    if source == 'OpenAI':
        if model.startswith("o1"):
            return ChatOpenAI(model = model, temperature = -1, **kwargs)
        return ChatOpenAI(model = model, temperature = temperature, **kwargs)
    elif source == 'Anthropic':
        return ChatAnthropic(model = model, 
                            temperature = temperature,
                            max_tokens = 4096,
                            **kwargs)
    else:
        # assuming a locally-served model
        assert port is not None, f"Model {model} is not supported, please provide a local port if it is a locally-served model."
        llm = CustomChatModel(model = model, model_type=source, temperature = temperature)
        llm.client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key=api_key).chat.completions
        return llm

class ExperimentalDataLoader:
    def __init__(self, data_path, table_dict_selection='default', data_sampling=-1):
        self.data_path = os.path.join(data_path, 'bio_database')
        self.available_datasets = [
            "gtex_tissue_gene_tpm",
            "gwas_catalog",
            "gene_info",
            "genetic_interaction",
            "genebass_synonymous_filtered",
            "genebass_missense_LC_filtered",
            "genebass_pLoF_filtered",
            "affinity_capture_ms",
            "two_hybrid",
            "synthetic_growth_defect",
            "affinity_capture_rna",
            "co_fractionation",
            "synthetic_lethality",
            "dosage_growth_defect",
            "proximity_label_ms",
            "synthetic_rescue",
            "reconstituted_complex",
            "eqtl_ukbb",
            "pqtl_ukbb",
            "sqtl_ukbb",
            "variant_table",
            "trait"
        ]

        self.permute_columns = {
            'gtex_tissue_gene_tpm': ['Gene'],
            'gwas_catalog': ['REPORTED GENE(S)', 'MAPPED_GENE', 'UPSTREAM_GENE_ID', 'DOWNSTREAM_GENE_ID', 'SNP_GENE_IDS'],
        }

        # Load datasets based on user input (default or all_bio)
        if table_dict_selection == 'default':
            self.datasets_to_load = [
                "gtex_tissue_gene_tpm",
                "gwas_catalog",
                "gene_info"
            ]
        elif table_dict_selection == 'all_bio':
            self.datasets_to_load = self.available_datasets  # Load all datasets

        if data_sampling != -1:
            all_datasets = self.available_datasets
            np.random.seed(42)
            np.random.shuffle(all_datasets)
            self.datasets_to_load = all_datasets[:data_sampling]
            print(f"Sampled datasets: {self.datasets_to_load}")

        # Load the selected datasets into the table_dict
        self.table_dict_selection = table_dict_selection
        self.table_dict = self._load_selected_datasets()
        self.data_desc = self._generate_data_description()
        
    def _load_selected_datasets(self):
        """Loads only the selected datasets and returns a dictionary."""
        table_dict = {}
        for dataset in self.datasets_to_load:
            df_name = f"df_{dataset}"
            table_dict[df_name] = self._load_data(f"{dataset}.pkl")
        return table_dict

    def _load_data(self, file_name):
        """Helper method to load data from a pickle file."""
        try:
            return pd.read_pickle(os.path.join(self.data_path, file_name))
        except FileNotFoundError:
            print(f"File {file_name} not found in path {self.data_path}")
            return None

    def _generate_data_description(self):
        """Generates a description of each dataset's columns and the first row of data."""
        desc = ""
        for name, df in self.table_dict.items():
            if df is not None:
                desc += f"{name}:\n{dict(zip(df.columns.values, df.iloc[0].values))}\n\n"
        return desc

    def get_data(self, table_name):
        """Returns the requested DataFrame."""
        return self.table_dict.get(table_name, None)
    
    def load_into_globals(self):
        """Loads each dataset into the global namespace."""
        for name, df in self.table_dict.items():
            if df is not None:
                globals()[name] = df

    def display_data_description(self):
        """Prints the data description."""
        print(self.data_desc)

    def permute_selected_columns(self, random_seed = 42):

        if self.table_dict_selection == 'default':
            self.random_seed = random_seed
            """Permutes the specified columns together for each dataset in permute_columns."""
            for dataset_name, columns_to_permute in self.permute_columns.items():
                df_name = f"df_{dataset_name}"
                df = self.table_dict.get(df_name, None)
                
                if df is not None and all(col in df.columns for col in columns_to_permute):
                    # Set the random seed for reproducibility
                    if self.random_seed is not None:
                        np.random.seed(self.random_seed)

                    # Permute rows of the selected columns together
                    # Shuffle the DataFrame rows and then reassign the columns
                    permuted_df = df[columns_to_permute].sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
                    df[columns_to_permute] = permuted_df
                    
                    # Update the table_dict with the permuted DataFrame
                    self.table_dict[df_name] = df
                    print(f"Permuted columns {columns_to_permute} in dataset {df_name}")
                else:
                    print(f"Columns {columns_to_permute} not found in dataset {df_name} or dataset does not exist.")
        elif self.table_dict_selection == 'all_bio':
            for df_name in self.table_dict:
                df = self.table_dict[df_name]
                permuted_df = df.copy()
                for col in df.columns:
                    permuted_df[col] = df[col].sample(frac=1, random_state=random_seed).reset_index(drop=True)
                    random_seed = random_seed * 2 % (2 ** 31)
                self.table_dict[df_name] = permuted_df


def load_file_dynamic(filepath):
    # Read the first few bytes to infer the delimiter
    with open(filepath, 'r') as file:
        first_line = file.readline()
    
    # Check for delimiter
    if '\t' in first_line:
        delimiter = '\t'
    elif ',' in first_line:
        delimiter = ','
    else:
        raise ValueError("Unknown delimiter. File is neither CSV nor TSV.")
    
    # Load the DataFrame using the detected delimiter
    df = pd.read_csv(filepath, delimiter=delimiter)
    return df


class DiscoveryBenchDataLoader:
    def __init__(self, data_path, metadata):
        self.data_path = data_path
        self.metadata = metadata    # dictionary/json object
        self.available_datasets = metadata["datasets"]
        self.table_dict = self._load_datasets()
        self.data_desc = self._generate_data_description()
    
    def _load_datasets(self):
        table_dict = {}
        for entry in self.available_datasets:
            table_path = os.path.join(self.data_path, entry["name"])
            table_dict[f'df_{entry["name"].split(".")[0]}'] = load_file_dynamic(table_path)
        return table_dict
    
    def _generate_data_description(self):
        desc = {}
        for entry in self.available_datasets:
            # print(entry["name"])
            table_name = f'df_{entry["name"].split(".")[0]}'
            df = self.table_dict[table_name]
            
            columns = entry["columns"]["raw"]
            for column in columns:
                value = df[column['name']].iloc[0]
                if isinstance(value, np.generic):
                    value = value.item()
                column["example_value"] = value
            desc_entry = {
                "description": entry["description"],
                "columns": json.dumps(columns),
            }
            desc[table_name] = desc_entry
        
        return json.dumps(desc, indent=4)
    
    def load_into_globals(self):
        """Loads each dataset into the global namespace."""
        for name, df in self.table_dict.items():
            if df is not None:
                globals()[name] = df
    
    def display_data_description(self):
        """Prints the data description."""
        print(self.data_desc)
    
    def permute_selected_columns(self, columns="all", random_seed = 42):
        # permute all columns
        np.random.seed(random_seed)
        for df_name in self.table_dict:
            df = self.table_dict[df_name]
            self.table_dict[df_name] = df.apply(np.random.permutation)
        
        # for df_name in self.table_dict:
        #     df = self.table_dict[df_name]
        #     permuted_df = df.copy()
        #     for col in df.columns:
        #         permuted_df[col] = df[col].sample(frac=1, random_state=random_seed).reset_index(drop=True)
        #         random_seed = random_seed * 2 % (2 ** 31)
        #     self.table_dict[df_name] = permuted_df

def pretty_print(message, printout = True):
    if isinstance(message, tuple):
        title = message
    else:
        if isinstance(message.content, list):
            title = get_msg_title_repr(message.type.title().upper() + " Message", bold=is_interactive_env())
            if message.name is not None:
                title += f"\nName: {message.name}"

            for i in message.content:
                if i['type'] == 'text':
                    title += f"\n{i['text']}\n"
                elif i['type'] == 'tool_use':
                    title += f"\nTool: {i['name']}"
                    title += f"\nInput: {i['input']}"
            if printout:
                print(f"{title}")
        else:
            title = get_msg_title_repr(message.type.title() + " Message", bold=is_interactive_env())
            if message.name is not None:
                title += f"\nName: {message.name}"
            title += f"\n\n{message.content}"
            if printout:
                print(f"{title}")
    return title

class CustomDataLoader:
    def __init__(self, data_folder, random_seed=42):
        """Initialize data loader with path to data folder.
        
        Args:
            data_folder (str): Path to folder containing pickle files
            random_seed (int): Random seed for permutations
        """
        self.data_path = data_folder
        self.random_seed = random_seed
        self.table_dict = {}
        self._load_all_datasets()
        self.data_desc = self._generate_data_description()

    def _load_all_datasets(self):
        """Automatically loads all pickle and CSV files found in data_path."""
        pickle_files = glob(os.path.join(self.data_path, "*.pkl"))
        csv_files = glob(os.path.join(self.data_path, "*.csv"))
        
        if not pickle_files and not csv_files:
            raise ValueError(f"No pickle or CSV files found in {self.data_path}")
            
        for file_path in pickle_files + csv_files:
            file_name = os.path.basename(file_path)
            dataset_name = os.path.splitext(file_name)[0]
            df_name = f"df_{dataset_name}"
            if file_path.endswith('.pkl'):
                self.table_dict[df_name] = self._load_data(file_name)
            elif file_path.endswith('.csv'):
                self.table_dict[df_name] = pd.read_csv(file_path)

    def _load_data(self, file_name):
        """Helper method to load data from a pickle file."""
        try:
            df = pd.read_pickle(os.path.join(self.data_path, file_name))
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"File {file_name} does not contain a pandas DataFrame")
            return df
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
            return None

    def _generate_data_description(self):
        """Generates a description of each dataset's columns and the first row."""
        desc = ""
        for name, df in self.table_dict.items():
            if df is not None:
                desc += f"{name}:\nColumns: {df.columns.tolist()}\n"
                desc += f"Sample row: {dict(zip(df.columns, df.iloc[0]))}\n\n"
        return desc

    def get_data(self, table_name):
        """Returns the requested DataFrame."""
        return self.table_dict.get(table_name, None)

    def load_into_globals(self):
        """Loads each dataset into the global namespace."""
        for name, df in self.table_dict.items():
            if df is not None:
                globals()[name] = df

    def display_data_description(self):
        """Prints the data description."""
        print(self.data_desc)

    def permute_columns(self, dataset_name, columns_to_permute):
        """Permutes specified columns in a dataset.
        
        Args:
            dataset_name (str): Name of dataset (without 'df_' prefix)
            columns_to_permute (list): List of column names to permute
        """
        df_name = f"df_{dataset_name}"
        df = self.table_dict.get(df_name, None)
        
        if df is None:
            raise ValueError(f"Dataset {df_name} not found")
            
        if not all(col in df.columns for col in columns_to_permute):
            raise ValueError(f"Not all columns {columns_to_permute} found in {df_name}")
            
        np.random.seed(self.random_seed)
        permuted_df = df[columns_to_permute].sample(frac=1).reset_index(drop=True)
        df[columns_to_permute] = permuted_df
        self.table_dict[df_name] = df
