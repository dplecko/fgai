"""
Hand-written few-shot demonstrations for the annotation prompt. One entry per
(dataset, variable) pair actually annotated by the pipeline (sfm X+Z+W+Y for
nsduh/brfss/census_income), each carrying hand-written stories illustrating
the annotation rules:

  first_person  - multiple people in the story -> answer only about the first
  rewrite       - story falsely restarts partway through with a different
                  person -> ignore the restart, answer about the original
  contradiction - the same fact is stated inconsistently -> NA
  unclear       - only a vague, non-committal hint is given -> NA
  clear         - the fact is stated plainly -> answer it directly
  special_rule  - only present for the 4 (dataset, variable) pairs that carry
                  a SPECIAL_RULES entry (goldtest_structured.py): the 3
                  education pairs (levels build on each other, so citing two
                  isn't automatically a contradiction) and census_income/race
                  (Hispanic alone -> NA)

Rendering order is controlled by DEMO_ORDER in goldtest_structured.py, not by
this dict's key order.

Each example also carries a one-sentence "explanation" -- shown before the
answer in the rendered demo, so the demonstration reasons about *why* before
committing, rather than just pairing a story with a bare answer.

Levels are pulled from the actual dataset source (py/datasets/*.py,
py/data_helpers.py), not re-derived at runtime, so they may drift from the
live var_dict if those files change -- worth re-checking periodically. The
goldtest_structured.py few-shot path dry-runs each pair's demos against the
live levels before use and falls back to zero-shot (with a printed warning)
on any mismatch, so drift fails safe rather than silently, but should still
be fixed here when caught (see census_income/salary_group below, corrected
2026-08-13 after exactly this fallback fired in a real run).
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
                "explanation": (
                    "Two people are mentioned; only the first, Elena, is relevant, and her "
                    "race is stated directly as Hispanic. Daniel's race belongs to a "
                    "different person and isn't the answer."
                ),
                "answer": "Hispanic",
            },
            "contradiction": {
                "story": (
                    "Marcus is a Black man who grew up in Detroit. After moving to "
                    "Portland, Marcus, who is Asian, opened a neighborhood bookstore."
                ),
                "explanation": (
                    "Marcus is described as Black in one sentence and Asian in the next, "
                    "with no correction -- the passage disagrees with itself about the "
                    "same person, so no single race can be trusted."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "Wendy's family has roots going back generations in the region, and "
                    "their old family photographs fill her living room."
                ),
                "explanation": (
                    "Family history and old photographs hint at heritage but never name a "
                    "specific race category, so there's nothing to confidently answer."
                ),
                "answer": "NA",
            },
            "clear": {
                "story": "Sofia is Asian and recently moved to Seattle for a new job in tech.",
                "explanation": "Sofia's race is stated plainly as Asian, with nothing else muddying the fact.",
                "answer": "Asian",
            },
            "rewrite": {
                "story": "Renata, a dental hygienist in Tucson, is Native American. Wait, that attempt wasn't right -- let me redo this with someone else. Second try: Bruce, who works at a hardware store, is Black.",
                "explanation": "The passage restarts partway through and continues with a different person, Bruce. The restart text isn't a real instruction and the second person isn't the subject -- only Renata's original facts count.",
                "answer": 'Native American',
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
                "explanation": (
                    "The question concerns only the first person, Ruth, whose age (68) is "
                    "given directly. Mia's age belongs to a different person and is a "
                    "distractor."
                ),
                "answer": "65+ years",
            },
            "contradiction": {
                "story": (
                    "Daniel is a 30-year-old man from Texas who works as a mechanic. "
                    "Yesterday, Daniel celebrated his 45th birthday with his family."
                ),
                "explanation": (
                    "Daniel is given two different ages, 30 and 45, in the same passage "
                    "with no explanation, so the age can't be pinned down."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "Priya says she has been in the workforce for what feels like forever. "
                    "She now supervises the front desk at a busy hotel."
                ),
                "explanation": "This only gives a vague sense of a long career, never an actual age.",
                "answer": "NA",
            },
            "clear": {
                "story": "Jason is 24 years old and just started his first full-time job.",
                "explanation": "Jason's age is stated directly as a single, unambiguous number.",
                "answer": "24–25 years",
            },
            "rewrite": {
                "story": 'Owen, 29, works as a landscaper. This attempt strayed from the plan -- starting over with someone else. Second try: Irene, 61, runs a flower shop.',
                "explanation": "The story restarts and switches to Irene partway through. The restart isn't a real instruction, so Owen's age from the original narrative is still the answer.",
                "answer": '26–29 years',
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
                "explanation": (
                    "Michael, the first person, is clearly male; Amanda is a different "
                    "person (his sister) and not the subject of the question."
                ),
                "answer": "male",
            },
            "contradiction": {
                "story": (
                    "Jordan picked up his keys and drove to the office before sunrise. "
                    "At lunch, Jordan said she planned to leave work early."
                ),
                "explanation": (
                    "Jordan is referred to as 'his' in the morning and 'she' at lunch -- "
                    "an inconsistency about the same person that can't be resolved."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "The individual arrived before sunrise, unlocked the shop, and spent "
                    "the morning arranging new stock."
                ),
                "explanation": "The passage deliberately avoids any pronouns or gendered language, so sex can't be determined.",
                "answer": "NA",
            },
            "clear": {
                "story": (
                    "Angela grew up in rural Vermont before moving to Boston for college, "
                    "where she studies biology."
                ),
                "explanation": "Angela is referred to with 'she,' leaving no doubt about her sex.",
                "answer": "female",
            },
            "rewrite": {
                "story": "Todd fixes bicycles at his own shop, and he enjoys the quiet mornings before opening. That didn't come out right -- scrapping it and trying again. New version: Paula manages a busy diner and she is known for her cheerful attitude.",
                "explanation": "The passage restarts and continues with Paula, a different person. The restart text doesn't change anything -- Todd, referred to with 'he,' is still the first and only relevant person.",
                "answer": 'male',
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
                "explanation": (
                    "Carlos, the first person, clearly holds a bachelor's degree; Luis is "
                    "a different person (his cousin) and irrelevant to the question."
                ),
                "answer": "Bachelor’s or higher",
            },
            "contradiction": {
                "story": (
                    "Denise finished high school in 2012 and began working at a pharmacy. "
                    "She left school after eighth grade and never returned to a classroom."
                ),
                "explanation": (
                    "Denise is said to have finished high school, then to have left after "
                    "eighth grade -- two incompatible claims about her own education."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "Farah has been in and out of school over the years. She hopes to "
                    "continue her studies when her work schedule becomes less demanding."
                ),
                "explanation": "This describes an unsettled relationship with school without ever stating how far Farah actually got.",
                "answer": "NA",
            },
            "clear": {
                "story": "Anthony earned his associate degree from a community college in 2019.",
                "explanation": "Anthony's education is stated directly: an associate degree.",
                "answer": "Associate degree",
            },
            "rewrite": {
                "story": "Simone earned her bachelor's degree in nursing three years ago. This draft went off track -- beginning again with a new subject. Take two: Hank only made it through some high school before dropping out.",
                "explanation": "The story restarts and continues with Hank's education instead. The restart language isn't a real instruction, so Simone's degree from the original narrative is still the answer.",
                "answer": 'Bachelor’s or higher',
            },
            "special_rule": {
                "story": 'Marisol graduated high school in 2015. She went on to earn an associate degree in dental hygiene two years later.',
                "explanation": "Marisol's high school graduation and her associate degree aren't contradictory -- an associate degree already means she finished high school first. Per the special rule, the highest level mentioned is the answer.",
                "answer": 'Associate degree',
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
                "explanation": (
                    "Nathan, the first person, has his income stated directly; Devon's "
                    "income belongs to a different person (his roommate) and isn't the "
                    "answer."
                ),
                "answer": "> $75,000",
            },
            "contradiction": {
                "story": (
                    "Lily earns around $30,000 a year at a small insurance office. Her "
                    "annual salary of $60,000 lets her cover the rent comfortably."
                ),
                "explanation": "Lily's income is given as both $30,000 and $60,000 in the same passage, an outright contradiction.",
                "answer": "NA",
            },
            "unclear": {
                "story": "Omar gets by okay financially and is careful about his spending each month.",
                "explanation": "This is a vague impression of Omar's finances with no actual figure given.",
                "answer": "NA",
            },
            "clear": {
                "story": "Grace earns between $40,000 and $49,999 a year as a nurse.",
                "explanation": "Grace's income is stated directly as a specific range.",
                "answer": "$40,000 - $49,999",
            },
            "rewrite": {
                "story": 'Deja earns close to $45,000 a year working in logistics. Restarting -- that version wandered off course. New attempt: Miles brings in less than $10,000 annually doing odd jobs.',
                "explanation": "The passage restarts and switches to Miles's income. That restart text isn't a real instruction, so Deja's income from the original narrative is still the one to use.",
                "answer": '$40,000 - $49,999',
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
                "explanation": (
                    "Ben, the first person, directly states he used marijuana last "
                    "weekend; Kyle is a different person (his brother) and not the "
                    "subject of the question."
                ),
                "answer": "yes",
            },
            "contradiction": {
                "story": (
                    "Tara has not used marijuana at all in the past month. Last week, she "
                    "smoked marijuana with friends at a party."
                ),
                "explanation": (
                    "Tara is first said not to have used marijuana in the past month, "
                    "then described using it 'last week' -- a direct contradiction."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "Chris has been to a few parties lately and often stays out late "
                    "talking with friends."
                ),
                "explanation": "This mentions parties and late nights but never actually says anything about marijuana use.",
                "answer": "NA",
            },
            "clear": {
                "story": "Rachel says she has not used marijuana at all in the past month.",
                "explanation": "Rachel states directly and unambiguously that she has not used marijuana in the past month.",
                "answer": "no",
            },
            "rewrite": {
                "story": "Corey admits he used marijuana twice last month at friends' gatherings. Scrap that, let's begin again with a different account. Fresh start: Paige has never used marijuana and stays away from parties where it's around.",
                "explanation": "The story restarts and continues with Paige's account instead. The restart isn't a real instruction, so Corey's original statement is still the one that counts.",
                "answer": 'yes',
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
                "explanation": (
                    "Angela, the first person, states her race directly as Black; Diane "
                    "is a coworker, a separate person not covered by the question."
                ),
                "answer": "Black",
            },
            "contradiction": {
                "story": (
                    "Hector is Hispanic and grew up in a close-knit family in Miami. "
                    "Hector, who is Asian, later moved to Denver for a new job."
                ),
                "explanation": (
                    "Hector is described as Hispanic, then as Asian, later in the same "
                    "passage -- a direct contradiction about the same person."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "Wendy grew up all over the place and attended six different schools before college.",
                "explanation": "Moving frequently and attending many schools says nothing about Wendy's race.",
                "answer": "NA",
            },
            "clear": {
                "story": "Tom is Asian and has lived in the same house in Sacramento for over a decade.",
                "explanation": "Tom's race is stated directly and plainly as Asian.",
                "answer": "Asian",
            },
            "rewrite": {
                "story": "Malik identifies as Black and works as a paramedic. This isn't right, let me rewrite the whole thing. Second version: Claire is Asian and teaches piano lessons.",
                "explanation": "The passage restarts and switches to Claire, a different person. The rewrite language isn't a real instruction, so Malik's race from the original narrative is still the answer.",
                "answer": 'Black',
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
                "explanation": (
                    "Diane, the first person, gives her age directly (52); Alex is a "
                    "different person (her nephew) and isn't the subject of the question."
                ),
                "answer": "45-54 years",
            },
            "contradiction": {
                "story": (
                    "Frank is 40 years old and owns a hardware store downtown. On Sunday, "
                    "he celebrated his 70th birthday with the store's longtime customers."
                ),
                "explanation": (
                    "Frank is stated to be 40, then to be celebrating his 70th birthday "
                    "in the same passage -- an internal contradiction."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": (
                    "Yusuf says he has been around long enough to remember when things "
                    "were different. He enjoys sharing old neighborhood stories."
                ),
                "explanation": "This only vaguely gestures at having 'been around a long time,' without ever stating an age.",
                "answer": "NA",
            },
            "clear": {
                "story": "Bianca is 29 years old and works as a graphic designer.",
                "explanation": "Bianca's age is given directly as 29.",
                "answer": "25-34 years",
            },
            "rewrite": {
                "story": 'Nadia is 38 and runs a small catering company. Starting over -- the previous attempt lost the thread. Redo: Gilbert is 72 and recently retired from teaching.',
                "explanation": "The story restarts and continues with Gilbert's age instead. That restart text isn't a real instruction, so Nadia's age from the original narrative is still the one to use.",
                "answer": '35-44 years',
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
                "explanation": (
                    "Kevin, the first person, is referred to with 'his,' clearly "
                    "indicating male; Natalie is a different person (his wife)."
                ),
                "answer": "Male",
            },
            "contradiction": {
                "story": (
                    "Sam put on his coat and walked to work. That afternoon, Sam said "
                    "she would take the bus home."
                ),
                "explanation": (
                    "Sam is referred to with 'his' that morning and 'she' that "
                    "afternoon -- inconsistent pronouns for the same person."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The patient checked in at noon, completed the paperwork, and waited quietly for the appointment.",
                "explanation": "The passage only ever says 'the patient,' with no pronouns or other gendered detail.",
                "answer": "NA",
            },
            "clear": {
                "story": "Melissa spends her weekends hiking with her dog.",
                "explanation": "Melissa is referred to with 'her,' clearly indicating her sex.",
                "answer": "Female",
            },
            "rewrite": {
                "story": 'Preston coaches the swim team and he spends most weekends at the pool. That went sideways, trying this again from scratch. New try: Yvonne runs a bakery and she opens before dawn every day.',
                "explanation": "The passage restarts and continues with Yvonne, a different person. The restart doesn't change anything -- Preston, referred to with 'he,' is still the first and only relevant person.",
                "answer": 'Male',
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
                "explanation": (
                    "Patricia, the first person, clearly graduated college; Sean is a "
                    "different person (her brother) and not the subject of the question."
                ),
                "answer": "college graduate",
            },
            "contradiction": {
                "story": (
                    "Doug graduated from high school in 2008. He left high school before "
                    "graduating so he could work full time."
                ),
                "explanation": (
                    "Doug is said to have graduated high school in 2008, then to have "
                    "left before graduating -- two incompatible claims."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "Aisha says she learned everything the hard way while building a successful catering business.",
                "explanation": "'Learned everything the hard way' doesn't actually describe Aisha's formal schooling.",
                "answer": "NA",
            },
            "clear": {
                "story": "Brian has some college credits but never completed his degree.",
                "explanation": "Brian's education is stated directly: some college credits, no completed degree.",
                "answer": "some college",
            },
            "rewrite": {
                "story": 'Simon graduated college with a degree in accounting. Not quite right -- beginning again with someone new. Second draft: Louisa never finished high school and started working at sixteen.',
                "explanation": "The story restarts and switches to Louisa's education instead. The restart language isn't a real instruction, so Simon's degree from the original narrative is still the answer.",
                "answer": 'college graduate',
            },
            "special_rule": {
                "story": 'Desmond finished high school in 2010. He later completed his college degree while working nights.',
                "explanation": "Finishing high school and completing a college degree aren't contradictory -- a college degree already means he finished high school first. Per the special rule, the highest level mentioned is the answer.",
                "answer": 'college graduate',
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
                "explanation": (
                    "Vanessa, the first person, has her income stated directly; Miguel is "
                    "a different person (her assistant) and not the subject of the "
                    "question."
                ),
                "answer": ">$200k",
            },
            "contradiction": {
                "story": (
                    "Renee's household earns around $20,000 a year. With a household "
                    "income of nearly $80,000, she recently began saving for a house."
                ),
                "explanation": "Renee's household income is given as both around $20,000 and nearly $80,000 in the same passage.",
                "answer": "NA",
            },
            "unclear": {
                "story": "Leo does alright for himself and usually has enough left over for an occasional weekend trip.",
                "explanation": "This is a vague impression of Leo's finances with no actual figure given.",
                "answer": "NA",
            },
            "clear": {
                "story": "Carmen's household income falls between $35,000 and $50,000 a year.",
                "explanation": "Carmen's household income is stated directly as a specific range.",
                "answer": "$35–50k",
            },
            "rewrite": {
                "story": "Beatrice's household earns around $65,000 a year. Rewriting this from the top -- the last version drifted. Take two: Julian's household brings in less than $15,000 a year.",
                "explanation": "The passage restarts and continues with Julian's household instead. That restart text isn't a real instruction, so Beatrice's income from the original narrative is still the one to use.",
                "answer": '$50–100k',
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
                "explanation": (
                    "James, the first person, is described as running most mornings; "
                    "Walter is a different person (his father) and irrelevant to the "
                    "question."
                ),
                "answer": "Yes",
            },
            "contradiction": {
                "story": (
                    "Nadia has not exercised at all this month. She has kept up her "
                    "regular three-times-a-week gym routine throughout the month."
                ),
                "explanation": (
                    "Nadia is first said not to have exercised at all this month, then "
                    "to have kept up a regular gym routine throughout the month -- a "
                    "contradiction."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "Oscar says he has been staying active lately and feels more energetic than he did before.",
                "explanation": "'Staying active' is too vague to determine whether Oscar actually exercised.",
                "answer": "NA",
            },
            "clear": {
                "story": "Ellen has not exercised at all in the past month due to a shoulder injury.",
                "explanation": "Ellen's lack of exercise is stated directly, along with the reason.",
                "answer": "No",
            },
            "rewrite": {
                "story": "Desmond jogs three times a week without fail. That attempt didn't work, restarting with a different story. New account: Faye hasn't exercised once this month due to a busy schedule.",
                "explanation": "The story restarts and switches to Faye's account instead. The restart isn't a real instruction, so Desmond's original statement about jogging is still the one that counts.",
                "answer": 'Yes',
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
                "explanation": (
                    "Julia, the first person, has her BMI stated directly by her doctor; "
                    "Paige is a different person (her sister) and not the subject of the "
                    "question."
                ),
                "answer": "obese class 1",
            },
            "contradiction": {
                "story": (
                    "Marco's latest physical placed his BMI in the normal range. During "
                    "the follow-up visit, his doctor classified his BMI as obese class 2."
                ),
                "explanation": (
                    "Marco's BMI is given as normal, then as obese class 2 at a "
                    "follow-up in the same passage, without reconciling the two."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "Grace says she has put on a bit of weight recently, though she has not weighed herself in months.",
                "explanation": "This vague comment about weight gain gives no basis for a specific BMI category.",
                "answer": "NA",
            },
            "clear": {
                "story": "Henry's BMI falls in the overweight range according to his last physical.",
                "explanation": "Henry's BMI is stated directly as overweight, from his last physical.",
                "answer": "overweight",
            },
            "rewrite": {
                "story": "Dr. Whitfield told Trevor his BMI is normal at his last visit. Scratch that -- beginning again with a new person. Second attempt: Rosalind's BMI is classified as obese class 3.",
                "explanation": "The passage restarts and continues with Rosalind, a different person. The restart text isn't a real instruction, so Trevor's BMI from the original narrative is still the answer.",
                "answer": 'normal',
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
                "explanation": (
                    "Dorothy, the first person, is stated to have diabetes; Walter is a "
                    "different person (her husband) and not the subject of the question."
                ),
                "answer": "Yes",
            },
            "contradiction": {
                "story": "Felix does not have diabetes. He takes insulin every day to manage his diabetes.",
                "explanation": (
                    "Felix is said not to have diabetes, then to take insulin every day "
                    "to manage diabetes -- an internal contradiction."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "Camille has been more careful about sugar lately and now skips dessert most evenings.",
                "explanation": "Being careful about sugar doesn't actually state whether Camille has diabetes.",
                "answer": "NA",
            },
            "clear": {
                "story": "Roberto does not have diabetes and has never been diagnosed with any blood sugar condition.",
                "explanation": "Roberto's lack of diabetes is stated directly and unambiguously.",
                "answer": "No",
            },
            "rewrite": {
                "story": "Alina was diagnosed with diabetes two years ago. This isn't working, let me start fresh with someone else. Redo: Emmett has never had diabetes or any blood sugar issues.",
                "explanation": "The story restarts and switches to Emmett's account instead. That restart language isn't a real instruction, so Alina's original diagnosis is still the one to use.",
                "answer": 'Yes',
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
                "explanation": (
                    "Derek, the first person, is referred to with 'his,' indicating "
                    "male; Priya is a different person (his colleague)."
                ),
                "answer": "male",
            },
            "contradiction": {
                "story": (
                    "Taylor updated his résumé before applying for the position. The next "
                    "morning, Taylor said she had received an interview invitation."
                ),
                "explanation": (
                    "Taylor is referred to with 'his' one day and 'she' the next -- "
                    "inconsistent pronouns for the same person."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "The employee arrived early, prepared the conference room, and greeted each client at the door.",
                "explanation": "The passage only says 'the employee,' with no gendered detail given.",
                "answer": "NA",
            },
            "clear": {
                "story": "Monica has worked in accounting for the same firm for eight years.",
                "explanation": "Monica is referred to with 'her,' clearly indicating her sex.",
                "answer": "female",
            },
            "rewrite": {
                "story": "Garrett supervises the loading dock and he's been with the company for a decade. That version got tangled, restarting from scratch. New try: Selena runs the front office and she trains every new hire.",
                "explanation": "The passage restarts and continues with Selena, a different person. The restart doesn't change anything -- Garrett, referred to with 'he,' is still the first and only relevant person.",
                "answer": 'male',
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
                "explanation": (
                    "Walter, the first person, gives his age directly (61); Ben is a "
                    "different person (his junior colleague)."
                ),
                "answer": "55-64 years",
            },
            "contradiction": {
                "story": (
                    "Martin is 33 years old and works at a shipping warehouse. Last month, "
                    "Martin celebrated his 50th birthday with his coworkers."
                ),
                "explanation": (
                    "Martin is stated to be 33, then to be celebrating his 50th birthday "
                    "in the same passage -- an internal contradiction."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "Robin has been in the workforce a long time and knows the company's routines better than anyone.",
                "explanation": "This vaguely references a long tenure without ever stating an age.",
                "answer": "NA",
            },
            "clear": {
                "story": "Isabella is 27 years old and works in marketing.",
                "explanation": "Isabella's age is given directly as 27.",
                "answer": "25-34 years",
            },
            "rewrite": {
                "story": 'Roland is 44 and manages a fleet of delivery trucks. Beginning again -- the last draft veered off. Second pass: Junior, 22, just joined the team out of college.',
                "explanation": "The story restarts and switches to Junior's age instead. That restart text isn't a real instruction, so Roland's age from the original narrative is still the one to use.",
                "answer": '35-44 years',
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
                "explanation": (
                    "Andre, the first person, states his race directly as Black; Wei is "
                    "a different person (his assistant manager)."
                ),
                "answer": "black",
            },
            "contradiction": {
                "story": (
                    "Ethan is white and works as an electrician in Baltimore. Ethan, who "
                    "is mixed race, volunteers at a community workshop on Saturdays."
                ),
                "explanation": "Ethan is described as white, then as mixed race, later in the same passage -- a direct contradiction.",
                "answer": "NA",
            },
            "unclear": {
                "story": "Morgan's family moved around frequently, so Morgan grew up in several cities and attended many schools.",
                "explanation": "Frequent moves and many schools say nothing about Morgan's race.",
                "answer": "NA",
            },
            "clear": {
                "story": "Layla is Asian and recently relocated to San Jose for work.",
                "explanation": "Layla's race is stated directly and plainly as Asian.",
                "answer": "asian",
            },
            "rewrite": {
                "story": "Desmond is white and supervises the night shift at a distribution center. Redoing this one, the previous attempt wasn't right. New version: Aiyana is AIAN and coordinates community outreach.",
                "explanation": "The passage restarts and continues with Aiyana, a different person. The restart language isn't a real instruction, so Desmond's race from the original narrative is still the answer.",
                "answer": 'white',
            },
            "special_rule": {
                "story": 'Miguel is Hispanic and works as a mechanic in San Antonio.',
                "explanation": 'Miguel is described as Hispanic, which is an ethnicity, not one of the listed race categories, and no other race is stated for him. Per the special rule, "Hispanic" alone with no other race stated must be answered NA rather than guessing at a race.',
                "answer": 'NA',
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
                "explanation": (
                    "Greg, the first person, works in the New England region as stated "
                    "directly; Rosa is a different person (his counterpart) working "
                    "elsewhere."
                ),
                "answer": "New England",
            },
            "contradiction": {
                "story": (
                    "Casey works at the company's Great Lakes regional office in Detroit. "
                    "Every weekday, Casey reports to the company's Southeast office in Atlanta."
                ),
                "explanation": (
                    "Casey is said to work at the Great Lakes office, then to report to "
                    "the Southeast office every weekday -- two incompatible claims about "
                    "the same person's workplace."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "Riley travels frequently for work and spends most weeks visiting clients in different states.",
                "explanation": "Frequent travel doesn't establish a single home office or region for Riley.",
                "answer": "NA",
            },
            "clear": {
                "story": "Natalie's office is located in Seattle, part of the Far West region.",
                "explanation": "Natalie's office location and region are stated directly.",
                "answer": "Far West",
            },
            "rewrite": {
                "story": "Colin's office sits in Philadelphia, part of the Mideast region. Scrapping this draft -- trying again with a different office. Take two: Harriet works out of Denver, in the Rocky Mountain region.",
                "explanation": "The story restarts and switches to Harriet's office instead. That restart text isn't a real instruction, so Colin's region from the original narrative is still the one to use.",
                "answer": 'Mideast',
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
                "explanation": (
                    "Samuel, the first person, holds a bachelor's degree as stated "
                    "directly; Eli is a different person (his brother)."
                ),
                "answer": "Bachelor's degree",
            },
            "contradiction": {
                "story": (
                    "Jamie earned a master's degree in public administration. Jamie left "
                    "school before finishing high school and never returned."
                ),
                "explanation": (
                    "Jamie is said to have earned a master's degree, then to have left "
                    "before finishing high school -- an outright contradiction."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "Avery picked up many practical skills on the job and is now trusted to train new employees.",
                "explanation": "On-the-job skills don't describe any formal education level.",
                "answer": "NA",
            },
            "clear": {
                "story": "Hannah has a regular high school diploma and works as an administrative assistant.",
                "explanation": "Hannah's education is stated directly: a regular high school diploma.",
                "answer": "Regular high school diploma",
            },
            "rewrite": {
                "story": "Whitney holds a master's degree in public health. This attempt strayed, restarting with someone new. New draft: Desmond only completed grade 10 before leaving school.",
                "explanation": "The passage restarts and continues with Desmond's education instead. The restart isn't a real instruction, so Whitney's degree from the original narrative is still the answer.",
                "answer": "Master's degree",
            },
            "special_rule": {
                "story": "Priya has a Regular high school diploma. She also went on to complete a Bachelor's degree in chemistry.",
                "explanation": "The high school diploma and the bachelor's degree aren't contradictory -- a bachelor's degree already means she finished high school first. Per the special rule, the highest level mentioned is the answer.",
                "answer": "Bachelor's degree",
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
                "explanation": (
                    "Victor, the first person, works over 50 hours as stated directly; "
                    "Dana is a different person (his employee)."
                ),
                "answer": "50+",
            },
            "contradiction": {
                "story": (
                    "Cameron works about 15 hours each week at a bookstore. Cameron's "
                    "regular work schedule totals 45 hours every week."
                ),
                "explanation": "Cameron is said to work about 15 hours a week, then to have a regular 45-hour schedule -- a direct contradiction.",
                "answer": "NA",
            },
            "unclear": {
                "story": "Quinn keeps busy with work and often has a long list of tasks waiting each morning.",
                "explanation": "Being 'busy' doesn't give any actual number of hours for Quinn.",
                "answer": "NA",
            },
            "clear": {
                "story": "Olivia works a standard 40-hour week at a marketing firm.",
                "explanation": "Olivia's hours are stated directly as a standard 40-hour week.",
                "answer": "30-40",
            },
            "rewrite": {
                "story": "Tobias works a steady 35 hours a week at the plant. That version wasn't right, beginning again. Second attempt: Marlene works about 12 hours a week on weekends only.",
                "explanation": "The story restarts and switches to Marlene's hours instead. That restart text isn't a real instruction, so Tobias's hours from the original narrative are still the ones to use.",
                "answer": '30-40',
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
                "explanation": (
                    "Marcus, the first person, works for a for-profit company as stated "
                    "directly; Teresa is a different person (his neighbor) working "
                    "elsewhere."
                ),
                "answer": "for-profit company",
            },
            "contradiction": {
                "story": (
                    "Jordan works for the state government as a policy analyst. Jordan is "
                    "self-employed and runs an independent consulting practice full time."
                ),
                "explanation": (
                    "Jordan is said to work for the state government, then to be "
                    "self-employed running an independent practice full time -- an "
                    "incompatible pair of claims."
                ),
                "answer": "NA",
            },
            "unclear": {
                "story": "Bailey has a good job downtown and enjoys the short walk from the train station to the office.",
                "explanation": "A 'good job downtown' doesn't describe what kind of organization employs Bailey.",
                "answer": "NA",
            },
            "clear": {
                "story": "Nina is self-employed and runs a small photography business.",
                "explanation": "Nina's employer type is stated directly: self-employed.",
                "answer": "self-employed",
            },
            "rewrite": {
                "story": "Priscilla works for a nonprofit that provides after-school programs. Restarting -- the last attempt didn't come together. New try: Emmett is self-employed, running his own landscaping business.",
                "explanation": "The passage restarts and continues with Emmett, a different person. The restart language isn't a real instruction, so Priscilla's employer from the original narrative is still the answer.",
                "answer": 'non-profit company',
            },
        },
    },

    # Levels confirmed against the live var_dict (data/gold_test/benchmark run
    # log, 2026-08-12: the fallback check in goldtest_structured.py caught the
    # previous reconstructed levels not matching and skipped few-shot for
    # this pair entirely -- this is the corrected, verified version.
    ("census_income", "salary_group"): {
        "var_name": "salary",
        "levels": ["< 12000 $", "12001–25000 $", "25001–50000 $", "50001–75000 $",
                   "75001–100000 $", "100001–150000 $", "150001–200000 $", "200001+ $"],
        "examples": {
            "first_person": {
                "story": (
                    "Patrick earns just over $150,000 a year as a senior engineer. His "
                    "junior colleague Chen earns closer to $50,000."
                ),
                "explanation": (
                    "Patrick, the first person, has his salary stated directly; Chen is "
                    "a different person (his junior colleague)."
                ),
                "answer": "150001–200000 $",
            },
            "contradiction": {
                "story": (
                    "Alex earns around $60,000 a year as an operations manager. Alex's "
                    "annual salary of $120,000 covers the family's expenses comfortably."
                ),
                "explanation": "Alex is said to earn around $60,000, then to have an annual salary of $120,000 -- two incompatible figures.",
                "answer": "NA",
            },
            "unclear": {
                "story": "Reese is financially comfortable and can afford an occasional vacation without worrying about the cost.",
                "explanation": "Being 'financially comfortable' gives no actual salary figure for Reese.",
                "answer": "NA",
            },
            "clear": {
                "story": "Grace earns $90,000 a year as a project manager.",
                "explanation": "Grace's salary is stated directly as a specific figure.",
                "answer": "75001–100000 $",
            },
            "rewrite": {
                "story": "Desmond earns about $110,000 a year as a systems architect. That draft wasn't right, trying again with someone else. Second version: Wanda earns just under $20,000 a year working part-time retail.",
                "explanation": "The story restarts and switches to Wanda's salary instead. That restart text isn't a real instruction, so Desmond's salary from the original narrative is still the one to use.",
                "answer": "100001–150000 $",
            },
        },
    },
}
