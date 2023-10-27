# Python format of the table from the paper headings are ["Task", "Tested Concepts", "Supercategory"],

import os

tasks = [
    ["Abstract Algebra", "Groups, rings, fields, vector spaces, ...", "STEM"],
    ["Anatomy", "Central nervous system, circulatory system, ...", "STEM"],
    ["Astronomy", "Solar system, galaxies, asteroids, ...", "STEM"],
    [
        "Business Ethics",
        "Corporate responsibility, stakeholders, regulation, ...",
        "Other",
    ],
    [
        "Clinical Knowledge",
        "Spot diagnosis, joints, abdominal examination, ...",
        "Other",
    ],
    ["College Biology", "Cellular structure, molecular biology, ecology, ...", "STEM"],
    ["College Chemistry", "Analytical, organic, inorganic, physical, ...", "STEM"],
    ["College Computer Science", "Algorithms, systems, graphs, recursion, ...", "STEM"],
    [
        "College Mathematics",
        "Differential equations, real analysis, combinatorics, ...",
        "STEM",
    ],
    [
        "College Medicine",
        "Introductory biochemistry, sociology, reasoning, ...",
        "Other",
    ],
    [
        "College Physics",
        "Electromagnetism, thermodynamics, special relativity, ...",
        "STEM",
    ],
    ["Computer Security", "Cryptography, malware, side channels, fuzzing, ...", "STEM"],
    [
        "Conceptual Physics",
        "Newton's laws, rotational motion, gravity, sound, ...",
        "STEM",
    ],
    [
        "Econometrics",
        "Volatility, long-run relationships, forecasting, ...",
        "Social Sciences",
    ],
    [
        "Electrical Engineering",
        "Circuits, power systems, electrical drives, ...",
        "STEM",
    ],
    [
        "Elementary Mathematics",
        "Word problems, multiplication, remainders, rounding, ...",
        "STEM",
    ],
    [
        "Formal Logic",
        "Propositions, predicate logic, first-order logic, ...",
        "Humanities",
    ],
    ["Global Facts", "Extreme poverty, literacy rates, life expectancy, ...", "Other"],
    [
        "High School Biology",
        "Natural selection, heredity, cell cycle, Krebs cycle, ...",
        "STEM",
    ],
    ["High School Chemistry", "Chemical reactions, ions, acids and bases, ...", "STEM"],
    [
        "High School Computer Science",
        "Arrays, conditionals, iteration, inheritance, ...",
        "STEM",
    ],
    [
        "High School European History",
        "Renaissance, reformation, industrialization, ...",
        "Humanities",
    ],
    [
        "High School Geography",
        "Population migration, rural land-use, urban processes, ...",
        "Social Sciences",
    ],
    [
        "High School Government and Politics",
        "Branches of government, civil liberties, political ideologies, ...",
        "Social Sciences",
    ],
    [
        "High School Macroeconomics",
        "Economic indicators, national income, international trade, ...",
        "Social Sciences",
    ],
    [
        "High School Mathematics",
        "Pre-algebra, algebra, trigonometry, calculus, ...",
        "STEM",
    ],
    [
        "High School Microeconomics",
        "Supply and demand, imperfect competition, market failure, ...",
        "Social Sciences",
    ],
    ["High School Physics", "Kinematics, energy, torque, fluid pressure, ...", "STEM"],
    [
        "High School Psychology",
        "Behavior, personality, emotions, learning, ...",
        "Social Sciences",
    ],
    [
        "High School Statistics",
        "Random variables, sampling distributions, chi-square tests, ...",
        "STEM",
    ],
    [
        "High School US History",
        "Civil War, the Great Depression, The Great Society, ...",
        "Humanities",
    ],
    [
        "High School World History",
        "Ottoman empire, economic imperialism, World War I, ...",
        "Humanities",
    ],
    [
        "Human Aging",
        "Senescence, dementia, longevity, personality changes, ...",
        "Other",
    ],
    [
        "Human Sexuality",
        "Pregnancy, sexual differentiation, sexual orientation, ...",
        "Social Sciences",
    ],
    [
        "International Law",
        "Human rights, sovereignty, law of the sea, use of force, ...",
        "Humanities",
    ],
    [
        "Jurisprudence",
        "Natural law, classical legal positivism, legal realism, ...",
        "Humanities",
    ],
    [
        "Logical Fallacies",
        "No true Scotsman, base rate fallacy, composition fallacy, ...",
        "Humanities",
    ],
    [
        "Machine Learning",
        "SVMs, VC dimension, deep learning architectures, ...",
        "STEM",
    ],
    ["Management", "Organizing, communication, organizational structure, ...", "Other"],
    ["Marketing", "Segmentation, pricing, market research, ...", "Other"],
    ["Medical Genetics", "Genes and cancer, common chromosome disorders, ...", "Other"],
    ["Miscellaneous", "Agriculture, Fermi estimation, pop culture, ...", "Other"],
    [
        "Moral Disputes",
        "Freedom of speech, addiction, the death penalty, ...",
        "Humanities",
    ],
    [
        "Moral Scenarios",
        "Detecting physical violence, stealing, externalities, ...",
        "Humanities",
    ],
    ["Nutrition", "Metabolism, water-soluble vitamins, diabetes, ...", "Other"],
    [
        "Philosophy",
        "Skepticism, phronesis, skepticism, Singer's Drowning Child, ...",
        "Humanities",
    ],
    [
        "Prehistory",
        "Neanderthals, Mesoamerica, extinction, stone tools, ...",
        "Humanities",
    ],
    [
        "Professional Accounting",
        "Auditing, reporting, regulation, valuation, ...",
        "Other",
    ],
    [
        "Professional Law",
        "Torts, criminal law, contracts, property, evidence, ...",
        "Humanities",
    ],
    [
        "Professional Medicine",
        "Diagnosis, pharmacotherapy, disease prevention, ...",
        "Other",
    ],
    [
        "Professional Psychology",
        "Diagnosis, biology and behavior, lifespan development, ...",
        "Social Sciences",
    ],
    [
        "Public Relations",
        "Media theory, crisis management, intelligence gathering, ...",
        "Social Sciences",
    ],
    [
        "Security Studies",
        "Environmental security, terrorism, weapons of mass destruction, ...",
        "Social Sciences",
    ],
    [
        "Sociology",
        "Socialization, cities and community, inequality and wealth, ...",
        "Social Sciences",
    ],
    [
        "US Foreign Policy",
        "Soft power, Cold War foreign policy, isolationism, ...",
        "Social Sciences",
    ],
    [
        "Virology",
        "Epidemiology, coronaviruses, retroviruses, herpesviruses, ...",
        "Other",
    ],
    [
        "World Religions",
        "Judaism, Christianity, Islam, Buddhism, Jainism, ...",
        "Humanities",
    ],
]

TASK_KEY_TO_CAT = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


def print_super_map() -> dict[str, str]:
    # strip of trailing dev / test
    to_match: dict[str, str] = dict()
    for row in tasks:
        to_match[row[0].lower().replace(" ", "_")] = row[-1].lower().replace(" ", "_")
    return to_match


def get_super_categories_from_filepath(path: str) -> str:
    # strip of trailing dev / test
    path = os.path.basename(path)
    x = path.replace("_dev", "").replace("_test", "").replace(".csv", "")
    to_match = {}
    for row in tasks:
        to_match[row[0].lower().replace(" ", "_")] = row[-1]
    return to_match[x].lower().replace(" ", "_")


if __name__ == "__main__":
    # file_paths = glob("data/mmlu/dev/*.csv")
    # base_name = [os.path.basename(f) for f in file_paths]
    # print([get_super_categories_from_filepath(i) for i in base_name])
    print(print_super_map())
