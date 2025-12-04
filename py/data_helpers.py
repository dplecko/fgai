import pandas as pd


# data loading utilities
def load_sfm(dataset: str):
    if dataset == "nsduh":
        sfm = {
            "X": ["race"],
            "Z": ["age", "sex"],
            "W": ["edu", "income"],
            "Y": ["mj_monthly"],
        }
    elif dataset == "brfss":
        sfm = {
            "X": ["race"],
            "Z": ["age_group", "sex", "state"],
            "W": ["education", "income", "bmi", "exercise_monthly"],
            "Y": ["diabetes"],
        }
    elif dataset == "census_income":
        sfm = {
            "X": ["sex"],
            "Z": ["age", "race", "economic_region"],
            "W": [
                "education",
                "",
            ],
        }
    elif dataset == "census_doctor":
        sfm = {
            "X": ["sex"],
            "Z": ["age", "race", "economic_region"],
            "W": ["marital_status", "children", "family_size", "hours_worked"],
            "Y": ["doctor"],
        }
    elif dataset == "census_surgeon":
        sfm = {
            "X": ["sex"],
            "Z": ["age", "race", "economic_region"],
            "W": [
                "marital_status",
                "children",
                "family_size",
                "hours_worked",
            ],
            "Y": ["surgeon"],
        }

    return sfm


def load_data(dataset: str):

    if "census" in dataset:
        df = pd.read_parquet(f"data/raw/census.parquet")

        if "doctor" in dataset:
            df = df[df["healthcare"] == 1]
        elif "surgeon" in dataset:
            df = df[df["doctor"] == 1]
    else:
        df = pd.read_parquet(f"data/raw/{dataset}.parquet")

    var_names = get_var_names(dataset)
    sfm = load_sfm(dataset)
    var_ord = {
        var: isinstance(df[var].dtype, pd.CategoricalDtype) and df[var].dtype.ordered
        for var in df.columns
    }
    var_dict = {}
    for var in var_names.keys():
        # extract the variable categories
        if isinstance(df[var].dtype, pd.CategoricalDtype) and df[var].cat.ordered:
            var_dict[var] = list(df[var].cat.categories)
        else:
            var_dict[var] = sorted(df[var].unique())

    return df, var_dict, var_names, var_ord, sfm


def get_var_names(dataset):
    if dataset == "nsduh":
        return {
            "age": "age",
            "sex": "sex",
            "race": "race",
            "edu": "education",
            "income": "income",
            "alc_monthly": "alcohol last month use",
            "cig_monthly": "cigarette last month use",
            "mj_monthly": "marijuana last month use",
            "coc_ever": "cocaine ever use",
        }
    elif dataset == "brfss":
        return {
            "age": "age",
            "state": "state",
            "sex": "sex",
            "race": "race",
            "education": "education",
            "income": "income",
            "exercise_monthly": "exercise monthly",
            "bmi": "body mass index (BMI)",
            "diabetes": "diabetes",
            "high_bp": "high blood pressure",
        }
    elif "census" in dataset:
        return {
            "age": "age",
            "sex": "sex",
            "race": "race",
            "economic_region": "economic region",
            "education": "education",
            "income": "income",
            "hours_per_week": "hours worked per week",
            "occupation": "occupation",
            "marital_status": "marital status",
            "children": "number of children",
            "family_size": "family size",
            "salary_group": "salary",
            "doctor": "is a physician",
            "surgeon": "is a surgeon",
        }
