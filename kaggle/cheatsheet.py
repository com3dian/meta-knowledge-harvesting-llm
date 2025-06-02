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
the importance of not only the stress type but also when it occurs within the crop’s lifecycle.

Moreover, the location of an experiment determines the ambient climate baseline, which affects
how far or close conditions are to a crop's optimal temperature range. This regional variability
means that even the same crop can exhibit different yield responses in different climates or
growing zones. For example, a degree of warming that harms yields in one area might have a
neutral or even beneficial effect in another, depending on local conditions.

Importantly, yield response isn’t always driven by single climate factors. The interaction of
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
Crop Type - Crop Yield;
Crop Type - Climate Drivers;
Crop Yield - Climate Drivers;
Crop Yield - Experimental Design;
Crop Yield - Location;
Crop Yield - Time;
"""

CHEATSHEETS["special_interests"] = """
- Numerical Variables: The crop type, including the full name and abbreviations of the crop type.
- Scientific Concepts: The numerical value crop yield, including the values of the crop yield and their units, percentage value of crop yield, or the difference of crop yield between two conditions. E.g. 10% increase in crop yield, 10% decrease in rice yield, 10% increase in wheat yield compared to the control group, 10% decrease in maize yield compared to the control group.
- : The climate drivers such as temperature, water availability and CO2 concentration, including the values of the climate drivers and their units, or percentage value of climate drivers.
- Experimental Design: The full name and abbreviations of the experimental design, or the scenario of the experiment.
- Location: The location of the experiment, including the full name and abbreviations of the location, or the latitude and longitude of the location.
- Time: The time of the experiment, including day, month, year, the duration of the experiment, or the timing of exposure to stress—especially during critical developmental phases.
"""

CHEATSHEETS["entity_extraction"] = """---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all entitie considering {special_interest} for each entity type. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Extract one or two **full sentences** from the input TEXT **exactly as they appear**, which best describe the entity’s attributes or activities. **Do not summarize, paraphrase, or alter the text in any way. Only copy directly from the source.**
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
- entity_description: Extract one or two **full sentences** from the input TEXT **exactly as they appear**, which best describe the entity’s attributes or activities. **Do not summarize, paraphrase, or alter the text in any way. Only copy directly from the source.**
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
