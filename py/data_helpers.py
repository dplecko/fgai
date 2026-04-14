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
            "Z": ["age_group", "sex"],
            "W": ["education", "income", "bmi", "exercise_monthly"],
            "Y": ["diabetes"],
        }
    elif dataset == "census_income":
        sfm = {
            "X": ["sex"],
            "Z": ["age_group", "race", "economic_region"],
            "W": [
                "education",
                "hours_worked",
                "employer"
            ],
            "Y": ["salary_group"],
        }
    elif dataset == "census_doctor":
        sfm = {
            "X": ["sex"],
            "Z": ["age_group", "race", "economic_region"],
            "W": ["marital", "children", "family_size", "hours_worked"],
            "Y": ["doctor"],
        }
    elif dataset == "census_surgeon":
        sfm = {
            "X": ["sex"],
            "Z": ["age_group", "race", "economic_region"],
            "W": [
                "marital",
                "children",
                "family_size",
                "hours_worked",
            ],
            "Y": ["surgeon"],
        }

    return sfm


def load_context(dataset: str):
    CONTEXTS = {
        "nsduh": "We are considering adults (18+ years old) living in the United States in 2023.",
        "brfss": "We are considering adults (18+ years old) living in the United States in 2023.",
        "census_income": "We are considering adults (18+ years old) who are employed in the United States in 2023.",
    }
    return CONTEXTS.get(dataset, "")


def load_data(dataset: str):

    if "census" in dataset:
        df = pd.read_parquet(f"data/raw/census.parquet")
        
        # categorize education
        education_levels = [
            "No schooling completed",
            "Nursery school, preschool",
            "Kindergarten",
            "Grade 1",
            "Grade 2",
            "Grade 3",
            "Grade 4",
            "Grade 5",
            "Grade 6",
            "Grade 7",
            "Grade 8",
            "Grade 9",
            "Grade 10",
            "Grade 11",
            "12th grade - no diploma",
            "Regular high school diploma",
            "GED or alternative credential",
            "Some college, but less than 1 year",
            "1 or more years of college credit, no degree",
            "Associate's degree",
            "Bachelor's degree",
            "Master's degree",
            "Professional degree beyond a bachelor's degree",
            "Doctorate degree",
        ]
        df["education"] = pd.Categorical(
            df["education"], categories=education_levels, ordered=True
        )
        
        # bin age into groups
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 24, 34, 44, 54, 64, 74, 79, float("inf")],
            labels=["18-24 years", "25-34 years", "35-44 years", "45-54 years", "55-64 years", "65-74 years", "75-79 years", "80+ years"]
        )
        
        # bin hours worked into <10, 10-20, 20-30, 30-40, 40-50, 50+ hours
        df["hours_worked"] = pd.cut(
            df["hours_worked"],
            bins=[0, 10, 20, 30, 40, 50, float("inf")],
            labels=["<10", "10-20", "20-30", "30-40", "40-50", "50+"]
        )
        
        # bin children into 0, 1, 2, 3, 4, 5+ children
        df["children"] = pd.cut(
            df["children"],
            bins=[-1, 0, 1, 2, 3, 4, float("inf")],
            labels=["0", "1", "2", "3", "4", "5+"]
        )
        
        # bin family size into 1, 2, 3, 4, 5+ family members
        df["family_size"] = pd.cut(
            df["family_size"],
            bins=[0, 1, 2, 3, 4, float("inf")],
            labels=["1", "2", "3", "4", "5+"]
        )
        
        # code doctor and surgeon as binary variables no/yes based on 0/1 values
        df["doctor"] = df["doctor"].apply(lambda x: "yes" if x == 1 else "no")
        df["surgeon"] = df["surgeon"].apply(lambda x: "yes" if x == 1 else "no")

        if "doctor" in dataset:
            df = df[df["healthcare"] == 1]
        elif "surgeon" in dataset:
            df = df[df["doctor"] == 1]
        elif "income" in dataset:  # census income case
            # filter age >= 18 and <= 65
            df = df[(df["age"] >= 18) & (df["age"] <= 65)]
            # filter such that employment status is employed
            df = df[df["employment_status"] == "employed"]
    elif dataset == "brfss":
        df = pd.read_parquet(f"data/raw/brfss.parquet")
        
        # fix education and income ordering
        df["education"] = pd.Categorical(
            df["education"],
            categories=["no high school", "high school", "some college", "college graduate"],
            ordered=True,
        )
        df["income"] = pd.Categorical(
            df["income"],
            categories=["<$15k", "$15–25k", "$25–35k", "$35–50k", "$50–100k", "$100k-200k", ">$200k"],
            ordered=True,
        )
        
        # discretize BMI into bins according to WHO
        df["bmi"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 35, 40, float("inf")],
            labels=["underweight", "normal", "overweight", 
                    "obese class 1", "obese class 2", "obese class 3"],
        )
    else: 
        df = pd.read_parquet(f"data/raw/{dataset}.parquet")

    var_names = get_var_names(dataset)
    sfm = load_sfm(dataset)
    
    # subset var_names and df to only include variables in sfm
    sfm_vars = sfm["X"] + sfm["Z"] + sfm["W"] + sfm["Y"]
    var_names = {var: name for var, name in var_names.items() if var in sfm_vars}
    df = df[sfm_vars + ["weight"]]
    
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
            
    context = load_context(dataset)

    return df, var_dict, var_names, var_ord, sfm, context


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
            "age_group": "age",
            # "state": "state", --- IGNORE for now (>26 levels) ---
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
            "age_group": "age",
            "sex": "sex",
            "race": "race",
            "economic_region": "economic region",
            "education": "education",
            "hours_worked": "hours worked per week",
            "occp_code": "occupation",
            "marital": "marital status",
            "children": "number of children",
            "family_size": "family size",
            "employer": "employer",
            "salary_group": "salary",
            "doctor": "is a physician",
            "surgeon": "is a surgeon",
        }
