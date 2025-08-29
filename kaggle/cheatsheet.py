from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

CHEATSHEETS: dict[str, Any] = {}

CHEATSHEETS["cheat_sheet"] = """
In scientific meta-analyses that explore how climate change affects crop yields,
the relationships between variables like crop yield and climate drivers are
deeply interconnected—and critically shaped by the time and location of the
experiments.

Crops have many type, and each crop exhibits distinct phenological development characteristics
throughout its growth cycle. These developmental stages—such as germination, vegetative growth,
flowering, and maturation—vary in duration and intensity depending on the crop species and
environmental conditions. The differences in phenological development directly influence
the biomass accumulation, resource allocation, and ultimately the formation of yield. As a result,
each crop type produces different levels of yield under given conditions.

Different crops exhibit distinct yield responses depending on the climate driver they
face, such as elevated temperatures, CO₂ concentrations, drought, or combinations of these.
For instance, crops like wheat and rice tend to show greater sensitivity to heat and CO₂
changes compared to crops like maize and sorghum, which are more resilient in hot, arid
climates.

The yield outcome also hinges on the timing of exposure to stress—especially during critical
developmental phases like anthesis and grain filling. For example, heat stress during anthesis
can cause a 30% reduction in crop yield, and the reduction is 10% in C4 crops. This highlights
the importance of not only the stress type but also when it occurs within the crop's lifecycle.

Moreover, the location of an experiment determines the ambient climate baseline, which affects
how far or close conditions are to a crop's optimal temperature range. This regional variability
means that even the same crop can exhibit different yield responses in different climates or
growing zones. For example, a degree of warming that harms yields in one area might have a
neutral or even beneficial effect in another, depending on local conditions.

Importantly, yield response isn't always driven by single climate factors. The interaction of
multiple stressors—like heat and drought occurring together—often has a more severe effect than
each factor alone. These compound effects can reduce cereal grain yields by up to 60%, compared
to 30–40% for single stress events. Such outcomes can also vary by crop species and depend on
cultivar-specific traits, as well as soil conditions and water availability.

Experimental design adds another layer of complexity. Controlled settings like FACE or open-top
chambers simulate specific climate conditions to isolate variables, but their findings must be
interpreted in the context of real-world environmental variability, where multiple stressors may
co-occur unpredictably.

Ultimately, any accurate interpretation of climate impact on crop yield must take into account
not just the type of crop and climate variable involved, but also where and when the crop was
grown, what combinations of stressors were presentd. These interdependent relationships form the
backbone of agricultural meta-analysis and are key to projecting how global food systems may evolve
in a changing climate.
"""

CHEATSHEETS["cheat_sheet_kg"] = """
Technologies and Algorithms - machine learning;
Modeling Terms - photobioreactor simulation;
Cybersecurity Terms - data privacy;
Networking and Infrastructure - low-latency systems;
"""

CHEATSHEETS["special_interests"] = """
Software or Computational Method: Algorithms, machine learning models, or data analysis techniques used in engineering and computer science to classify, cluster, or predict phenomena. For example, "random forest classification to reconstruct lifelong movements" and "cluster-based sampling approach for selecting the representative data".
Scientific Method: Experimental designs, modeling strategies, and evaluation methods applied in scientific research across domains such as agriculture, engineering, and environmental science. For example, "quantitative methods including ANOVA were used for statistical comparison of heights" and "the authors combined chronological clustering with random forest classification".
Infrastructure or Equipment: Physical systems, devices, or built environments relevant to engineering or environmental experiments, such as dams, factories, or fish ladders. For example, "spillways, turbines or fish ladder of Lajeado Dam" and "hydroelectric dams... Biobío catchment".
Institution or Organization: Universities, research institutes, or governmental bodies involved in research or policy development. For example, "University of Sheffield's Advanced Manufacturing Research Center" and "National Department of Agriculture".
Ecological or Biological Entity: Plant, animal, or microbial species studied in environmental and biological sciences. For example, "Percilia irwini, an endangered small darter" and "Cliona orientalis, the most abundant bioeroding sponge species".
Pollutant or Material: Chemical or physical contaminants such as microplastics, metals, or pollutants studied for environmental and health impacts. For example, "heavy metal pollutants in aquatic environments cause a severe threat" and "bioaccumulation in the food chain".
Time Expression: Dates, durations, and temporal intervals relevant to experimental design or historical context. For example, "from October, 1999 through September, 2004" and "2001 through 2003".
Health or Disease Concept: References to disease conditions, health risks, or toxic effects observed in human or ecological studies. For example, "heavy metals... cause a severe threat to public health" and "plasmonic biosensors could simplify procedures and radically reduce time".
Food or Nutrition Element: Foods, supplements, or nutritional components considered in medical, agricultural, or food security contexts. For example, "food supplements using different traditionally processed local foods" and "context of rural revitalization... Ashram schools... dietary provisions".
Demographic Group: Groups such as children, students, farmers, or indigenous populations involved in scientific or social studies. For example, "scheduled tribes (ST) children from age 6 to 17 years" and "schoolchildren of three major local STs".
Measurement or Quantity: Scientific quantities with units such as percent increase, mg, kg, t/ha—used across medicine, agriculture, and environmental research. For example, "93.0 mg calcium, 172.4 mg magnesium" and "21% total essential amino acid".
"""

CHEATSHEETS["entity_extraction"] = """---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all entitie considering {special_interest} for each entity type. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Extract one or two **full sentences** from the input TEXT **exactly as they appear**, which best describe the entity's attributes or activities. **Do not summarize, paraphrase, or alter the text in any way. Only copy directly from the source.**
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: Extract one or two **full sentences** from the `source_text` that explain why the source entity and the target entity are related. Only copy the sentences **exactly as they appear** in the source text. **Do not paraphrase, summarize, or combine partial sentences.**
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: [{entity_types}]
Text:
{input_text}
######################
Output:"""

CHEATSHEETS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction.

---Remember Steps---

1. Identify all entities considering {special_interest} for each entity type. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

---Output---

Add them below using the same format:\n
""".strip()


CHEATSHEETS["entity_fill_nightly_extraction"] = """---Goal---
Given a list of nightly_entities with metadata and their related source texts, fill in missing fields such as
`entity_name` and `description` for each entity considering {special_interest} for each entity type. Also identify meaningful relationships between the entities
based on their source texts and inferred relationships.
Use {language} as output language.

---Steps---
1. For each entity, extract the following information base on the source text and reference:
- entity_name: A concise, meaningful name based on the source text. If English, capitalize appropriately.
- entity_type: One of the provided types (do not change it).
- entity_description: Extract one or two **full sentences** from the input TEXT **exactly as they appear**, which best describe the entity's attributes or activities. **Do not summarize, paraphrase, or alter the text in any way. Only copy directly from the source.**
- source: The source of the entity, which can be a sentence or a phrase from the source text.
- reference: The possible reference of the entity, a sentence in the source text that is related to the entity, do not include this in the output.

Format each enriched entity as:
("entity"{tuple_delimiter}<entity_key>{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<description>{tuple_delimiter}<source>{tuple_delimiter}<source_id>{tuple_delimiter}<file_path>)
Fill in the `<Nightly Entity Name>` and `<Nightly Inference>` placeholders with actual information or values in the input text.

2. From the enriched entities, identify clear relationships between pairs.
For each relationship, extract:
- source_entity: entity name of the source
- target_entity: entity name of the target
- relationship_description: Extract one or two **full sentences** from the `source_text` that explain why the source entity and the target entity are related. Only copy the sentences **exactly as they appear** in the source text. **Do not paraphrase, summarize, or combine partial sentences.**
- relationship_keywords: one or more high-level thematic keywords
- relationship_strength: numeric value (1–10) of inferred connection strength

Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)
Fill in the `<Nightly Entity Name>` and `<Nightly Inference>` placeholders with actual information or values in the input text.

3. Return all enriched entities followed by relationships, as a flat list.

Use **{record_delimiter}** as the list delimiter.
End the output with {completion_delimiter}.

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
**Entities and Relationships**:
{nightly_entities_and_relationships}
**Source Text**:
{input_text}
######################

######################
Output:"""


CHEATSHEETS["claim_preprocessing"] = """
---Goal---
Given a claim (which is a sentence extracted from a longer answer) and the full answer text as context, identify all pronouns in the claim and replace them with the specific entities or concepts they refer to, creating a clearer and more explicit version of the claim.

---Input---
1. Claim: A single sentence that contains pronouns that need to be resolved
2. Answer: The full text that provides context for understanding what the pronouns refer to

---Steps---
1. Identify all pronouns in the claim, including:
2. For each identified pronoun, analyze the full answer text to determine:
   - What specific entity, concept, or phrase the pronoun refers to
   - The antecedent (the noun or noun phrase that the pronoun replaces)
3. Create a resolved version of the claim by:
   - Replacing each pronoun with its specific referent
   - Ensuring the sentence remains grammatically correct
   - Maintaining the original meaning while making it more explicit

---Output Format---
Only output the resolved sentence with pronouns replaced by their specific referents. Do not include any analysis, explanation, or additional formatting.

#############################
---Real Data---
######################
Claim: {claim}
Answer: {answer}
######################
Output:
""".strip()


CHEATSHEETS["evidence_extraction"] = """
---Goal---
Given a text and a list of seed entities with their types, identify new entities from the text that are clearly related to the seed entities. For each new entity, provide its type from a list of possible entity types.

---Input---
1. Seed Entities: A list of existing entities with their types, which will be our starting point.
2. Entity Types: The list of allowed types for the new entities we find.
3. Text: The source text to search for new entities.

---Steps---
1. For each seed entity in the input list, search through the Text to find mentions or discussions about it.
2. Analyze the context around these mentions to discover other entities that are semantically and contextually related to the seed entity.
3. For each new related entity you find:
    - Identify its name (new_entity_name).
    - Assign it an new_entity_type from the given Entity Types list.
    - Note which seed_entity it is related to.
    - Extract one or two full sentences from the text that describe the relationship (relationship_description). These sentences must be copied exactly.
4. Consolidate all the found related entities into a single list.

---Output Format---
- Format each finding as a tuple: ("related_entity"{tuple_delimiter}<new_entity_name>{tuple_delimiter}<new_entity_type>{tuple_delimiter}<seed_entity_name>{tuple_delimiter}<relationship_description>)
- Use {record_delimiter} to separate each tuple in the final list.
- When finished, output {completion_delimiter}.

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Seed Entities: {seed_entities}
Entity Types: [{entity_types}]
Text:
{input_text}
######################
Output:
""".strip()

CHEATSHEETS["evidence_continue_extraction"] = """
Continue the evidence extraction process by finding additional entities that are correlated with the previously identified entities.

---Goal---
Given a text and a list of seed entities with their types that were previously extracted, continue to identify new entities from the text that are clearly related or correlated to the seed entities. For each new entity, provide its type from a list of possible entity types.

---Input---
1. Seed Entities: A list of previously extracted entities with their types, which will be our starting point.
2. Entity Types: The list of allowed types for the new entities we find.
3. Text: The source text to search for new correlated entities.

---Steps---
1. For each seed entity in the input list, search through the Text to find mentions or discussions about it.
2. Analyze the context around these mentions to discover other entities that are semantically and contextually related to the seed entity.
3. Look for entities that were missed in the previous extraction but are clearly correlated with the seed entities.
4. For each new related entity you find:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Extract one or two **full sentences** from the input TEXT **exactly as they appear**, which best describe the entity's attributes or activities. **Do not summarize, paraphrase, or alter the text in any way. Only copy directly from the source.**
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

5. Consolidate all the found related entities into a single list.

---Output Format---
- Format each finding as a tuple:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
- Use {record_delimiter} to separate each tuple in the final list.
- When finished, output {completion_delimiter}.

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Seed Entities: {seed_entities}
Entity Types: [{entity_types}]
######################
Output:
""".strip()

CHEATSHEETS["pronoun_resolution"] = """
---Goal---
Given a claim (which is a sentence extracted from a longer answer) and the full answer text as context, identify all pronouns in the claim and replace them with the specific entities or concepts they refer to, creating a clearer and more explicit version of the claim.

---Input---
1. Claim: A single sentence that contains pronouns that need to be resolved
2. Answer: The full text that provides context for understanding what the pronouns refer to

---Steps---
1. Identify all pronouns in the claim, including:
   - Personal pronouns: it, they, them, their, theirs
   - Demonstrative pronouns: this, that, these, those
   - Relative pronouns: which, who, whom, whose
   - Other referential terms: such, the former, the latter, etc.

2. For each identified pronoun, analyze the full answer text to determine:
   - What specific entity, concept, or phrase the pronoun refers to
   - The antecedent (the noun or noun phrase that the pronoun replaces)

3. Create a resolved version of the claim by:
   - Replacing each pronoun with its specific referent
   - Ensuring the sentence remains grammatically correct
   - Maintaining the original meaning while making it more explicit

4. Provide a step-by-step explanation of each pronoun resolution

---Output Format---
**Original Claim:** [original claim text]

**Pronoun Analysis:**
- [Pronoun 1]: refers to [specific entity/concept]
- [Pronoun 2]: refers to [specific entity/concept]
- [etc.]

**Resolved Claim:** [claim with pronouns replaced by their referents]

**Explanation:** [brief explanation of the resolution process and any important context used]

######################
---Examples---
######################

**Example 1:**
Claim: "They showed significant yield reductions under heat stress."
Answer: "The study examined wheat and rice crops under various temperature conditions. Wheat and rice crops were grown in controlled environments with temperatures ranging from 25°C to 40°C. They showed significant yield reductions under heat stress, with wheat being more sensitive than rice to elevated temperatures."

**Original Claim:** They showed significant yield reductions under heat stress.

**Pronoun Analysis:**
- They: refers to wheat and rice crops

**Resolved Claim:** Wheat and rice crops showed significant yield reductions under heat stress.

**Explanation:** The pronoun "they" refers back to "wheat and rice crops" mentioned in the previous sentence of the answer context.

**Example 2:**
Claim: "This resulted in a 25% decrease in grain production."
Answer: "Drought conditions persisted for three months during the growing season. Farmers reported severe water shortages that affected irrigation systems. This resulted in a 25% decrease in grain production compared to normal years."

**Original Claim:** This resulted in a 25% decrease in grain production.

**Pronoun Analysis:**
- This: refers to drought conditions and severe water shortages affecting irrigation systems

**Resolved Claim:** Drought conditions and severe water shortages affecting irrigation systems resulted in a 25% decrease in grain production.

**Explanation:** The demonstrative pronoun "this" refers to the combined effect of the drought conditions and water shortages described in the preceding context.

#############################
---Real Data---
######################
Claim: {claim}
Answer: {answer}
######################
Output:
""".strip()

