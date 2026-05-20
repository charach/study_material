"""Phase 10 — closed-set topic taxonomy for LLM classification → topic-driven retrieval.

Design (per user's direction):
  - The LLM is shown this FIXED list (name + description) and outputs only the IDs it picks.
    It never invents a topic and never writes a citation → no hallucination.
  - Each topic maps deterministically to a German search query (query_de) + priority law
    codes (codes). The picked topics drive multi-query BGE-M3 retrieval over laws + court.

Derived from val gold law-code distribution (measured):
  BGE 69, ZGB 39, StPO 36, BGer 33, OR 18, BGG 13, StGB 12, IVG 10, ATSG 7, StBOG 4,
  ZPO 4, BV 3, IPRG 2, SchKG 1.
  → BGE+BGer (case law) = 41% of gold; BGG (federal appeal) is near-universal.
"""

# id → (name, description shown to LLM, German search query, priority law codes)
TOPICS = {
    1:  ("detention_stpo",
         "Pre-trial / security detention, coercive measures in criminal procedure "
         "(grounds for detention, flight/collusion/recidivism risk, detention appeals).",
         "Untersuchungshaft Sicherheitshaft Haftgründe Fluchtgefahr Kollusionsgefahr "
         "Wiederholungsgefahr Haftbeschwerde Haftentlassung Verhältnismässigkeit",
         ["StPO", "BV", "StBOG"]),

    2:  ("criminal_procedure",
         "Criminal procedure generally (charges, evidence, appeals/Beschwerde in criminal "
         "matters, costs, defence rights) — when the dispute is procedural, not the offence itself.",
         "Strafverfahren Strafprozess Beschwerde Berufung Beweis Verfahrenskosten "
         "amtliche Verteidigung Rechtsmittel",
         ["StPO", "StBOG", "BV"]),

    3:  ("criminal_substantive",
         "Substantive criminal law — a specific offence (theft, fraud, assault, "
         "disloyal management/embezzlement, forgery, etc.) and its elements/penalty.",
         "Straftat Tatbestand Diebstahl Betrug Veruntreuung ungetreue Geschäftsbesorgung "
         "Urkundenfälschung Körperverletzung Strafzumessung Vorsatz",
         ["StGB"]),

    4:  ("family_zgb",
         "Family law — divorce, marriage, spousal/child maintenance, parental "
         "responsibility/custody, child protection measures.",
         "Scheidung Ehe Unterhalt Kindesunterhalt elterliche Sorge Obhut "
         "Kindesschutz Beistandschaft Besuchsrecht",
         ["ZGB"]),

    5:  ("inheritance_zgb",
         "Inheritance / succession — wills, intestate succession, compulsory portion, "
         "disinheritance, inheritance contracts, executors.",
         "Erbrecht Erbschaft Testament letztwillige Verfügung gesetzliche Erbfolge "
         "Pflichtteil Enterbung Erbvertrag Willensvollstrecker",
         ["ZGB"]),

    6:  ("property_zgb",
         "Property / rights in rem — ownership, possession, land register, easements, "
         "mortgages, co-ownership, acquisition in good faith.",
         "Eigentum Besitz Grundbuch Grunddienstbarkeit Grundpfand Miteigentum "
         "Stockwerkeigentum gutgläubiger Erwerb Sachenrecht",
         ["ZGB"]),

    7:  ("tenancy_or",
         "Lease / tenancy law — residential or commercial lease, rent, defects, "
         "termination, protection against unfair termination.",
         "Mietvertrag Miete Mietzins Mängel Kündigung ausserordentliche Kündigung "
         "Erstreckung Anfechtung Kündigungsschutz",
         ["OR"]),

    8:  ("employment_or",
         "Employment law — employment contract, wages, termination/dismissal, "
         "wrongful termination, protection of the employee.",
         "Arbeitsvertrag Lohn Kündigung missbräuchliche Kündigung fristlose Entlassung "
         "Arbeitszeugnis Konkurrenzverbot",
         ["OR"]),

    9:  ("contract_or",
         "General contract law and specific contracts other than lease/employment — "
         "sale, mandate/agency, contract for work (Werkvertrag), formation, performance, breach.",
         "Vertrag Kaufvertrag Werkvertrag Auftrag Mängel Gewährleistung Verzug "
         "Vertragsverletzung Willensmängel Verjährung",
         ["OR"]),

    10: ("tort_liability_or",
         "Tort / non-contractual liability and damages — fault, causation, damage, "
         "strict liability, product/owner liability.",
         "unerlaubte Handlung Haftung Schadenersatz Verschulden Kausalzusammenhang "
         "Genugtuung Kausalhaftung Werkeigentümerhaftung",
         ["OR"]),

    11: ("debt_enforcement_schkg",
         "Debt enforcement and bankruptcy — Betreibung, attachment, opposition, "
         "Rechtsöffnung, bankruptcy, composition.",
         "Betreibung Schuldbetreibung Konkurs Pfändung Arrest Rechtsöffnung "
         "Rechtsvorschlag Kollokation",
         ["SchKG"]),

    12: ("social_insurance",
         "Social insurance — disability (IV), old-age (AHV), accident (UV), health (KV) "
         "insurance; benefits, incapacity, the general social-insurance procedure (ATSG).",
         "Invalidenversicherung IV-Rente Erwerbsunfähigkeit Arbeitsunfähigkeit "
         "Sozialversicherung ATSG Unfallversicherung Altersrente",
         ["IVG", "ATSG", "UVG", "AHVG", "KVG"]),

    13: ("tax",
         "Tax law — direct federal/cantonal tax, VAT, withholding tax, assessment, deductions.",
         "Steuer direkte Bundessteuer Mehrwertsteuer Verrechnungssteuer "
         "Veranlagung Steuerabzug steuerbares Einkommen",
         ["DBG", "MWSTG", "StHG", "VStG"]),

    14: ("commercial_banking",
         "Commercial, banking and financial-market law, company law — companies, "
         "directors' liability, banking, securities, financial-market supervision; "
         "often with private-international-law (cross-border) elements.",
         "Gesellschaftsrecht Aktiengesellschaft Verwaltungsrat Bank Bankvertrag "
         "Finanzmarkt Sorgfaltspflicht Geldwäscherei internationales Privatrecht",
         ["OR", "BankG", "FINMAG", "IPRG", "GwG"]),

    15: ("civil_procedure",
         "Civil procedure — jurisdiction, lis pendens, evidence, provisional measures, "
         "appeals in civil matters (when the dispute is procedural).",
         "Zivilprozess Zuständigkeit Rechtshängigkeit Beweis vorsorgliche Massnahmen "
         "Berufung Beschwerde ZPO Klage",
         ["ZPO", "IPRG"]),

    16: ("administrative_public",
         "Administrative / public law — migration/asylum, building & planning, "
         "data protection, public-law disputes and their procedure.",
         "Verwaltungsrecht öffentliches Recht Migration Asyl Aufenthalt Datenschutz "
         "Verfügung Verwaltungsbeschwerde",
         ["AIG", "DSG", "RPG", "VwVG"]),

    17: ("federal_appeal",
         "ALWAYS APPLICABLE when the case reaches the Federal Supreme Court "
         "(Bundesgericht) on appeal — the procedural framework of the appeal itself "
         "(admissibility, deadline, grounds, cognition). Pick this for nearly every case.",
         "Beschwerde ans Bundesgericht Beschwerdefrist Beschwerdelegitimation "
         "Rügeprinzip Kognition Eintreten Endentscheid",
         ["BGG"]),

    18: ("constitutional_rights",
         "Fundamental / constitutional rights invoked in the dispute — right to be heard, "
         "personal liberty, equality, proportionality, fair trial (also ECHR/EMRK).",
         "Grundrecht rechtliches Gehör persönliche Freiheit Rechtsgleichheit "
         "Verhältnismässigkeit faires Verfahren EMRK",
         ["BV", "EMRK"]),
}

# Case-law (BGE/BGer) is 41% of gold and is retrieved from the court index using the
# same query_de keywords of the picked topics (no separate topic needed — court search
# runs on the union of picked-topic queries).
