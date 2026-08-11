"""
Hand-written few-shot demonstrations for the annotation prompt (see
prep_ann_prompt in generation.py). One entry per (dataset, variable) pair
actually annotated by the pipeline (sfm X+Z+W+Y for nsduh/brfss/census_income),
each carrying 4 hand-written stories illustrating the annotation rules:

  first_person  - multiple people in the story -> answer only about the first
  contradiction - the same fact is stated inconsistently -> NA
  unclear       - only a vague, non-committal hint is given -> NA
  clear         - the fact is stated plainly -> answer it directly

Levels are pulled from the actual dataset source (py/datasets/*.py,
py/data_helpers.py), not re-derived at runtime, so they may drift from the
live var_dict if those files change -- worth re-checking periodically.

NOTE on census_income/salary_group: the bucketing function (discrete_col) that
produces these labels isn't in the repo (only referenced from a stale raw-data
script). The 8 buckets below are reconstructed from the known break points
[12000, 25000, 50000, 75000, 100000, 150000, 200000] plus one confirmed real
label ("50001-75000 $", see r/helpers.R:86) -- not verified against the actual
pipeline output. Double check this one specifically.
"""

FEW_SHOT_EXAMPLES = {

    # ============================== NSDUH ==============================

    ("nsduh", "race"): {
        "var_name": "race",
        "levels": ["White", "Black", "Native American", "Pacific Islander", "Asian", "Multiple", "Hispanic"],
        "examples": {
            "first_person": {
                "story": (
                    "Elena, a bank teller from Phoenix, is Hispanic. Her coworker Daniel, "
                    "who transferred from the Chicago branch last year, is White."
                ),
                "answer": "Hispanic",
            },
            "contradiction": {
                "story": (
                    "The passage describes Marcus as Black in one sentence, then a few "
                    "lines later refers to him as Asian without explanation."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "The story mentions that Wendy's family has roots going back "
                    "generations in the region, without ever stating her race directly."
                ),
                "answer": "NA",
            },
            "clear": {
                "story": "Sofia is Asian and recently moved to Seattle for a new job in tech.",
                "answer": "Asian",
            },
        },
    },

    ("nsduh", "age"): {
        "var_name": "age",
        "levels": ["18–20 years", "21–23 years", "24–25 years", "26–29 years",
                   "30–34 years", "35–49 years", "50-64 years", "65+ years"],
        "examples": {
            "first_person": {
                "story": (
                    "Ruth, 68, spent Saturday afternoon watching her granddaughter's soccer "
                    "game. Her granddaughter Mia, 22 and a recent college graduate, scored "
                    "the winning goal."
                ),
                "answer": "65+ years",
            },
            "contradiction": {
                "story": (
                    "The passage first says Daniel is 30, but later in the same paragraph "
                    "refers to him celebrating his 45th birthday."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "The story mentions that Priya has been in the workforce for what feels "
                    "like forever, without ever giving a specific age."
                ),
                "answer": "NA",
            },
            "clear": {
                "story": "Jason is 24 years old and just started his first full-time job.",
                "answer": "24–25 years",
            },
        },
    },

    ("nsduh", "sex"): {
        "var_name": "sex",
        "levels": ["female", "male"],
        "examples": {
            "first_person": {
                "story": (
                    "Michael works the night shift at a downtown hospital. His sister "
                    "Amanda, who lives two states away, is a schoolteacher."
                ),
                "answer": "male",
            },
            "contradiction": {
                "story": (
                    "The passage refers to Jordan as 'he' early on, then later describes "
                    "Jordan using 'she' without explanation."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "The story only refers to the person as 'the individual' throughout, "
                    "without using any gendered language or pronouns."
                ),
                "answer": "NA",
            },
            "clear": {
                "story": (
                    "Angela grew up in rural Vermont before moving to Boston for college, "
                    "where she studies biology."
                ),
                "answer": "female",
            },
        },
    },

    ("nsduh", "edu"): {
        "var_name": "education",
        "levels": ["≤ 8th grade", "Some high school", "High school graduate",
                   "Some college, no degree", "Associate degree", "Bachelor’s or higher"],
        "examples": {
            "first_person": {
                "story": (
                    "Carlos completed his bachelor's degree in engineering five years ago. "
                    "His younger cousin Luis dropped out after eighth grade to help support "
                    "the family."
                ),
                "answer": "Bachelor’s or higher",
            },
            "contradiction": {
                "story": (
                    "The story states that Denise finished high school, but a few sentences "
                    "later says she never made it past eighth grade."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "The passage mentions that Farah has 'been in and out of school over "
                    "the years' without saying how far she got."
                ),
                "answer": "NA",
            },
            "clear": {
                "story": "Anthony earned his associate degree from a community college in 2019.",
                "answer": "Associate degree",
            },
        },
    },

    ("nsduh", "income"): {
        "var_name": "income",
        "levels": ["< $10,000", "$10,000 - $19,999", "$20,000 - $29,999", "$30,000 - $39,999",
                   "$40,000 - $49,999", "$50,000 - $74,999", "> $75,000"],
        "examples": {
            "first_person": {
                "story": (
                    "Nathan brings home just over $75,000 a year working as a project "
                    "manager. His roommate Devon, a part-time barista, earns less than "
                    "$10,000 annually."
                ),
                "answer": "> $75,000",
            },
            "contradiction": {
                "story": (
                    "The passage says Lily earns around $30,000 a year, but later mentions "
                    "her salary is closer to $60,000."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The story mentions that Omar 'gets by okay' financially without ever stating a figure.",
                "answer": "NA",
            },
            "clear": {
                "story": "Grace earns between $40,000 and $49,999 a year as a nurse.",
                "answer": "$40,000 - $49,999",
            },
        },
    },

    ("nsduh", "mj_monthly"): {
        "var_name": "marijuana use last month",
        "levels": ["no", "yes"],
        "examples": {
            "first_person": {
                "story": (
                    "Ben mentions he smoked marijuana at a friend's party just last "
                    "weekend. His brother Kyle, by contrast, says he hasn't touched it in "
                    "years."
                ),
                "answer": "yes",
            },
            "contradiction": {
                "story": (
                    "The story first says Tara hasn't used marijuana in the past month, "
                    "but a later sentence describes her smoking with friends 'last week'."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "The passage vaguely mentions that Chris has 'been to a few parties "
                    "lately' without saying anything about drug use."
                ),
                "answer": "NA",
            },
            "clear": {
                "story": "Rachel says she has not used marijuana at all in the past month.",
                "answer": "no",
            },
        },
    },

    # ============================== BRFSS ==============================

    ("brfss", "race"): {
        "var_name": "race",
        "levels": ["White", "Black", "Asian", "AIAN", "Hispanic", "Other"],
        "examples": {
            "first_person": {
                "story": (
                    "Angela identifies as Black and works as a paralegal downtown. Her "
                    "coworker Diane, who sits in the next cubicle, is White."
                ),
                "answer": "Black",
            },
            "contradiction": {
                "story": (
                    "The passage describes Hector as Hispanic in the opening line, but a "
                    "few sentences later calls him Asian."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The story mentions that Wendy grew up 'all over the place' without ever describing her race.",
                "answer": "NA",
            },
            "clear": {
                "story": "Tom is Asian and has lived in the same house in Sacramento for over a decade.",
                "answer": "Asian",
            },
        },
    },

    ("brfss", "age_group"): {
        "var_name": "age",
        "levels": ["18-24 years", "25-34 years", "35-44 years", "45-54 years",
                   "55-64 years", "65-74 years", "75-79 years", "80+ years"],
        "examples": {
            "first_person": {
                "story": (
                    "Diane just turned 52 and recently switched careers to teaching. Her "
                    "nephew Alex, fresh out of college at 23, is still figuring out his "
                    "path."
                ),
                "answer": "45-54 years",
            },
            "contradiction": {
                "story": (
                    "The passage says Frank is 40, but later refers to him celebrating his "
                    "70th birthday."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "The story mentions that Yusuf has 'been around long enough to "
                    "remember when things were different,' without giving an actual age."
                ),
                "answer": "NA",
            },
            "clear": {
                "story": "Bianca is 29 years old and works as a graphic designer.",
                "answer": "25-34 years",
            },
        },
    },

    ("brfss", "sex"): {
        "var_name": "sex",
        "levels": ["Male", "Female"],
        "examples": {
            "first_person": {
                "story": (
                    "Kevin coaches little league on weekends. His wife Natalie runs a "
                    "small bakery downtown."
                ),
                "answer": "Male",
            },
            "contradiction": {
                "story": "The story refers to Sam as 'he' at the start, then switches to 'she' later without explanation.",
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage only ever refers to the subject as 'the patient,' with no pronouns or gendered details given.",
                "answer": "NA",
            },
            "clear": {
                "story": "Melissa spends her weekends hiking with her dog.",
                "answer": "Female",
            },
        },
    },

    ("brfss", "education"): {
        "var_name": "education",
        "levels": ["no high school", "high school", "some college", "college graduate"],
        "examples": {
            "first_person": {
                "story": (
                    "Patricia graduated college with a degree in nursing. Her brother Sean "
                    "never finished high school."
                ),
                "answer": "college graduate",
            },
            "contradiction": {
                "story": "The story says Doug graduated high school, but a later line describes him as never having finished.",
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that Aisha 'learned everything the hard way' without describing her schooling.",
                "answer": "NA",
            },
            "clear": {
                "story": "Brian has some college credits but never completed his degree.",
                "answer": "some college",
            },
        },
    },

    ("brfss", "income"): {
        "var_name": "income",
        "levels": ["<$15k", "$15–25k", "$25–35k", "$35–50k",
                   "$50–100k", "$100k-200k", ">$200k"],
        "examples": {
            "first_person": {
                "story": (
                    "Vanessa runs her own consulting firm and earns well over $200,000 a "
                    "year. Her assistant Miguel makes closer to $30,000."
                ),
                "answer": ">$200k",
            },
            "contradiction": {
                "story": (
                    "The story says Renee's household income is around $20,000 a year, but "
                    "a later sentence puts it closer to $80,000."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that Leo 'does alright for himself' without stating an income.",
                "answer": "NA",
            },
            "clear": {
                "story": "Carmen's household income falls between $35,000 and $50,000 a year.",
                "answer": "$35–50k",
            },
        },
    },

    ("brfss", "exercise_monthly"): {
        "var_name": "exercise monthly",
        "levels": ["Yes", "No"],
        "examples": {
            "first_person": {
                "story": (
                    "James goes for a run most mornings before work. His father, Walter, "
                    "hasn't exercised in years due to a bad knee."
                ),
                "answer": "Yes",
            },
            "contradiction": {
                "story": (
                    "The story says Nadia hasn't exercised at all this month, but a few "
                    "lines later mentions her regular gym routine."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that Oscar has 'been staying active lately' without giving any real detail.",
                "answer": "NA",
            },
            "clear": {
                "story": "Ellen has not exercised at all in the past month due to a shoulder injury.",
                "answer": "No",
            },
        },
    },

    ("brfss", "bmi"): {
        "var_name": "body mass index (BMI)",
        "levels": ["underweight", "normal", "overweight", "obese class 1", "obese class 2", "obese class 3"],
        "examples": {
            "first_person": {
                "story": (
                    "Julia's doctor classified her BMI as obese class 1 at her last "
                    "checkup. Her sister Paige, who runs marathons, has a BMI in the "
                    "underweight range."
                ),
                "answer": "obese class 1",
            },
            "contradiction": {
                "story": (
                    "The story describes Marco's BMI as normal early on, but later says "
                    "his doctor flagged him as obese class 2."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that Grace has 'put on a bit of weight recently' without giving any specifics.",
                "answer": "NA",
            },
            "clear": {
                "story": "Henry's BMI falls in the overweight range according to his last physical.",
                "answer": "overweight",
            },
        },
    },

    ("brfss", "diabetes"): {
        "var_name": "diabetes",
        "levels": ["Yes", "No"],
        "examples": {
            "first_person": {
                "story": (
                    "Dorothy was diagnosed with diabetes five years ago. Her husband "
                    "Walter has never had any blood sugar issues."
                ),
                "answer": "Yes",
            },
            "contradiction": {
                "story": "The story says Felix does not have diabetes, but later mentions he takes insulin daily.",
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that Camille has 'been more careful about sugar lately' without saying why.",
                "answer": "NA",
            },
            "clear": {
                "story": "Roberto does not have diabetes and has never been diagnosed with any blood sugar condition.",
                "answer": "No",
            },
        },
    },

    # =========================== CENSUS_INCOME ===========================

    ("census_income", "sex"): {
        "var_name": "sex",
        "levels": ["female", "male"],
        "examples": {
            "first_person": {
                "story": (
                    "Derek manages a warehouse team in Ohio. His colleague Priya oversees "
                    "logistics in the same facility."
                ),
                "answer": "male",
            },
            "contradiction": {
                "story": "The passage calls the subject 'he' in the first paragraph, then 'she' in the next without explanation.",
                "answer": "NA",
            },
            "unclear": {
                "story": "The story refers to the person only as 'the employee' throughout, with no gendered details.",
                "answer": "NA",
            },
            "clear": {
                "story": "Monica has worked in accounting for the same firm for eight years.",
                "answer": "female",
            },
        },
    },

    ("census_income", "age_group"): {
        "var_name": "age",
        "levels": ["18-24 years", "25-34 years", "35-44 years", "45-54 years",
                   "55-64 years", "65-74 years", "75-79 years", "80+ years"],
        "examples": {
            "first_person": {
                "story": (
                    "Walter is 61 and plans to retire in a few years. His junior colleague "
                    "Ben, fresh out of college at 24, just started."
                ),
                "answer": "55-64 years",
            },
            "contradiction": {
                "story": (
                    "The story says the subject is 33, but a later sentence mentions he's "
                    "been working since he turned 50."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions the person has 'been in the workforce a long time' without stating an age.",
                "answer": "NA",
            },
            "clear": {
                "story": "Isabella is 27 years old and works in marketing.",
                "answer": "25-34 years",
            },
        },
    },

    ("census_income", "race"): {
        "var_name": "race",
        "levels": ["white", "black", "asian", "AIAN", "NHOPI", "other", "mix"],
        "examples": {
            "first_person": {
                "story": (
                    "Andre identifies as Black and manages a retail store in Atlanta. His "
                    "assistant manager Wei is Asian."
                ),
                "answer": "black",
            },
            "contradiction": {
                "story": (
                    "The story describes the subject as white in the opening line, but "
                    "later refers to him as mixed race."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that the subject's family moved around a lot growing up, without describing race at all.",
                "answer": "NA",
            },
            "clear": {
                "story": "Layla is Asian and recently relocated to San Jose for work.",
                "answer": "asian",
            },
        },
    },

    ("census_income", "economic_region"): {
        "var_name": "economic region",
        "levels": ["New England", "Mideast", "Great Lakes", "Plains", "Southeast",
                   "Southwest", "Rocky Mountain", "Far West", "Abroad"],
        "examples": {
            "first_person": {
                "story": (
                    "Greg works out of an office in Boston, part of the New England "
                    "region. His counterpart Rosa works remotely from a Phoenix office in "
                    "the Southwest."
                ),
                "answer": "New England",
            },
            "contradiction": {
                "story": (
                    "The story places the subject's workplace in the Great Lakes region "
                    "early on, then later refers to their office in the Southeast."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that the subject 'travels a lot for work' without saying where their main office is.",
                "answer": "NA",
            },
            "clear": {
                "story": "Natalie's office is located in Seattle, part of the Far West region.",
                "answer": "Far West",
            },
        },
    },

    ("census_income", "education"): {
        "var_name": "education",
        "levels": [
            "No schooling completed", "Nursery school, preschool", "Kindergarten", "Grade 1",
            "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7", "Grade 8",
            "Grade 9", "Grade 10", "Grade 11", "12th grade - no diploma",
            "Regular high school diploma", "GED or alternative credential",
            "Some college, but less than 1 year",
            "1 or more years of college credit, no degree", "Associate's degree",
            "Bachelor's degree", "Master's degree",
            "Professional degree beyond a bachelor's degree", "Doctorate degree",
        ],
        "examples": {
            "first_person": {
                "story": (
                    "Samuel holds a bachelor's degree in economics. His younger brother "
                    "Eli only completed grade 9 before leaving school."
                ),
                "answer": "Bachelor's degree",
            },
            "contradiction": {
                "story": (
                    "The story says the subject has a master's degree, but a later "
                    "sentence mentions they never finished high school."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that the subject 'picked up a lot of skills on the job' without describing their formal education.",
                "answer": "NA",
            },
            "clear": {
                "story": "Hannah has a regular high school diploma and works as an administrative assistant.",
                "answer": "Regular high school diploma",
            },
        },
    },

    ("census_income", "hours_worked"): {
        "var_name": "hours worked per week",
        "levels": ["<10", "10-20", "20-30", "30-40", "40-50", "50+"],
        "examples": {
            "first_person": {
                "story": (
                    "Victor works over 50 hours a week running his own business. His "
                    "employee Dana works closer to 20 hours a week part-time."
                ),
                "answer": "50+",
            },
            "contradiction": {
                "story": (
                    "The story says the subject works about 15 hours a week, but later "
                    "mentions their full 45-hour work schedule."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that the subject 'keeps busy with work' without stating any hours.",
                "answer": "NA",
            },
            "clear": {
                "story": "Olivia works a standard 40-hour week at a marketing firm.",
                "answer": "30-40",
            },
        },
    },

    ("census_income", "employer"): {
        "var_name": "employer",
        "levels": ["for-profit company", "non-profit company", "government", "self-employed"],
        "examples": {
            "first_person": {
                "story": (
                    "Marcus works for a large for-profit tech company in Austin. His "
                    "neighbor Teresa works for a nonprofit focused on housing."
                ),
                "answer": "for-profit company",
            },
            "contradiction": {
                "story": (
                    "The story says the subject works for the government, but later "
                    "describes them running their own self-employed consulting practice."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that the subject 'has a good job downtown' without saying what kind of organization employs them.",
                "answer": "NA",
            },
            "clear": {
                "story": "Nina is self-employed and runs a small photography business.",
                "answer": "self-employed",
            },
        },
    },

    # NOTE: levels reconstructed, not verified -- see module docstring.
    ("census_income", "salary_group"): {
        "var_name": "salary",
        "levels": ["≤$12,000", "$12,001–$25,000", "$25,001–$50,000",
                   "$50,001–$75,000", "$75,001–$100,000", "$100,001–$150,000",
                   "$150,001–$200,000", ">$200,000"],
        "examples": {
            "first_person": {
                "story": (
                    "Patrick earns just over $150,000 a year as a senior engineer. His "
                    "junior colleague Chen earns closer to $50,000."
                ),
                "answer": "$150,001–$200,000",
            },
            "contradiction": {
                "story": (
                    "The story says the subject earns around $60,000 a year, but later "
                    "references a salary closer to $120,000."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The passage mentions that the subject is 'financially comfortable' without giving a number.",
                "answer": "NA",
            },
            "clear": {
                "story": "Grace earns $90,000 a year as a project manager.",
                "answer": "$75,001–$100,000",
            },
        },
    },
}
