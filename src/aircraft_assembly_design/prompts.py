"""Prompt templates used by routing, generation, verification, and regeneration."""

from __future__ import annotations

import json


ROUTER_PROMPT_TEMPLATE = """ 
You are an intelligent routing assistant designed to classify user queries.

**Query Context:**
User Query: {question}

**Classification Rules:**
1. Knowledge Graph Query (Output: "graph")
   - Query explicitly requests information retrieval from the knowledge graph
   - Contains domain-specific terms such as "process", "operation", "resource", "required resource", "predecessor"

2. Design Generation Query (Output: "design")
   - Requests synthesis, planning, or generation of new joint plans
   - Requires reasoning and optimization rather than information retrieval

3. Default Classification
   - When classification is ambiguous, default to "graph" to prioritize knowledge retrieval

**Output Specification:**
Respond exclusively with either **"graph"** or **"design"**. No additional text or explanation should be included.
"""


GRAPH_RESPONSE_PROMPT_TEMPLATE = """ 
You are a specialized knowledge graph interpreter for aircraft fuselage joint domain expertise. Your function is to process structured query results from the knowledge graph and provide accurate, semantically-rich responses to domain inquiries.

**Query Context:**
User Query: {question}
Executed Cypher Statement: {cypher}
Knowledge Graph Query Results: {graph_data}

**Response Guidelines:**
1. Extract only relevant information from the knowledge graph results to answer the user query
2. Present answers in structured list format for clarity
3. Include all data values without omission or inference beyond the provided results

**Output Requirements:**
Provide a comprehensive, structured response that directly addresses the user query while maintaining complete fidelity to the knowledge graph data.
"""


DESIGN_QA_PROMPT_TEMPLATE = """
**Role**: You are an expert in aircraft fuselage assembly planning. Your task is to generate a complete and feasible assembly plan based only on the conversation history and user query.

**Query Context:**
Retrieved graph data: {kg_context}
User query: {question}

**Process Requirements**:
▪ Strictly output according to the following four phase structure, without omitting any part.
▪ All outputs must be based on data from the retrieved graph data and user query.

Phase 1. **Data Extraction**
- Extract ALL resources and operations from conversation history, show them as a markdown table.
- Markdown table format: header row, then separator row |---|---|---|, then data rows.
For each resource, document:
▪ Cost (€/h)
▪ Calendar
▪ Quantity
For each operation, document:
▪ Type (Manual/Automatic)
▪ Duration (min)
▪ Required Resources (name (number))
▪ Total Cost (€)
▪ Predecessor
Note:
▪ Round Cost (€) to two decimal places.

Phase 2. **Constraint Analysis**
- Identify and list ALL operations belonging to the manual joint chain based on precedence dependencies, and use → to show the dependency relationships between these operations.
▪ Manual joint chain: operation_name_1 → ... → operation_name_n

- Identify and list ALL operations belonging to the automatic joint chain based on precedence dependencies, and use → to show the dependency relationships between these operations.
▪ Automatic joint chain: operation_name_1 → ... → operation_name_m

- Identify and list ALL shared operations that do not belong to either joint chain, and use → to show the dependency relationships between these operations.
▪ Shared operations:
operation_name_a → operation_name_b, operation_name_c, operation_name_d

Note:
▪ The listed operation sequences must strictly follow the precedence dependencies.

- Analyze and list all design constraints.
▪ Constraint 1: ...

Phase 3. **Strategy Decision**
- Determine the chain type for each quarter section (Manual joint chain / Automatic joint chain), considering resource availability.
▪ Quarter section 1: ...

- Determine the execution pattern of the quarter sections (Series / Parallel), ensuring that the required resources at any time do not exceed the available resource capacities.
▪ Execution pattern (example): 1 → 2 → (3 || 4)
Note:
▪ "→" indicates serial execution
▪ "||" indicates parallel execution

Phase 4. **Plan Generation**
- According to Phases 1, 2, and 3, generate a complete aircraft fuselage joint plan table following the Markdown format.
- Please strictly follow the chain type and execution pattern provided in Phase 3 when generating the table.
- The markdown table headers are: Order; Section; Operation; Type; Required Resources; Duration (min); Start Time (min); End Time (min); Cost (€).
- Markdown table format: header row, then separator row |---|---|---|, then data rows.
- Output the table directly as plain Markdown. Do NOT wrap the table in code blocks (no ```).
Note:
▪ Order: Use a single number (1, 2, 3, ...) for sequential steps. For parallel steps, all operations starting at the same scheduling step must share the same number and use different letter suffixes (4a, 4b, 4c, ...). The number identifies the scheduling step, and the letter identifies the parallel branch.
▪ Section: For operations belonging to manual or automatic joint chains, use "Quarter section 1", "Quarter section 2", "Quarter section 3", or "Quarter section 4", and for other operations, use "Shared".
▪ Operation: Use the full name of operations. Use the full name of operations exactly as given. Do not modify the operation names.
▪ Type: Use Manual/Automatic.
▪ Required Resources: Use name (number), and omission or abbreviation is not allowed, such as 'same as above'.
▪ Duration (min); Start Time (min); Cost (€): Use number only.
▪ All operations in the manual and automatic joint chains identified in Phase 2 must be fully included in Phase 4 without omission.
▪ Each quarter section must have its own complete joint chain in the table.
▪ The chain type and execution stages determined in Phase 3 must be strictly followed in Phase 4.
▪ Quarter sections 1, 2, 3 and 4 must appear in the table.
▪ All content in the table must not be omitted or abbreviated.
▪ All content in the table cannot include any formatting.
▪ Only generate one table.
- Calculate the total time and cost, and output them only after the table.
"""


PLAN_EXAMPLE = """Please help me design a complete aircraft fuselage joint plan. The plan must satisfy the following constraints.
    1. All operations must respect their precedence dependencies. An operation can start only after all its prerequisite operations have been completed.
    2. The first two operations must be "S40_00001_Jig in" and "S40_01001_Set up working environment", and the final three operations must be "S40_02002_Cleanup and add sealant", "S40_02003_Inspection", and "S40_00002_Jig out", in that order. These five operations are global shared operations and must each appear exactly once.
    3. The joint system contains an automatic joint chain and a manual joint chain. The plan must complete four 1/4 fuselage sections in total. Each 1/4 fuselage section must be completed by one joint chain, either the automatic joint chain or the manual joint chain.
    4. Execution patterns of these joint chains can be either serial or parallel. Serial execution means that one joint chain starts only after the previous joint chain is completed, while parallel execution means that multiple joint chains are executed during overlapping time intervals. Automatic joint chains must be executed in serial relative to other automatic joint chains. Manual joint chains can be executed in serial or in parallel relative to other manual joint chains. Automatic joint chains and manual joint chains can also be executed in serial or in parallel. The plan must satisfy all precedence dependencies and must not violate any resource capacity constraints at any time during execution.
    5. The automatic joint chain is defined as a sequence of automatic operations starting from "S40_02001_Set in position Rails and LFT" and ending at "S40_04014_Deinstall LFT and rails". When multiple automatic joint chains are executed consecutively in serial, "S40_02001_Set in position Rails and LFT" and "S40_04014_Deinstall LFT and rails" are shared operations that must be executed only once before the first chain and after the last chain respectively, while all other operations in the automatic joint chain must be executed once for each 1/4 fuselage section.
    6. The manual joint chain is defined as the sequence of manual operations starting from "S40_04003_Drilling template install" and ending at "S40_04013_Deburring int, positioning, attach them manual"."""


RETRY_PREFIX_TEMPLATE = """
You are generating Cypher for the SECOND time.

The first generated Cypher was invalid and failed during execution.

The first generated Cypher:
{bad_cypher}

The execution error message:
{error_message}

Please regenerate the Cypher query for the same user question.
You must correct the previous error and output only one valid Cypher query.
Do not include any explanation.
Do not wrap the query in markdown fences.

"""


def build_regeneration_prompt(plan_table, verification_report, human_feedback=""):
    return f"""
You are an expert in aircraft fuselage joint planning. Please regenerate the design plan based on the following content.

You are given:
1. The previously generated design plan table.
2. The verification report generated after simulation and engineering-constraint verification.
3. The human feedback for the previously generated design plan table.

**Process Requirements**:
▪ Strictly output according to the following four phase structure, without omitting any part.
▪ All outputs must be based on data from the retrieved graph data and user query.

Phase 1. **Data Extraction**
- Extract ALL resources and operations from conversation history, show them as a markdown table.
- Markdown table format: header row, then separator row |---|---|---|, then data rows.
For each resource, document:
▪ Cost (€/h)
▪ Calendar
▪ Quantity
For each operation, document:
▪ Type (Manual/Automatic)
▪ Duration (min)
▪ Required Resources (name (number))
▪ Total Cost (€)
▪ Predecessor
Note:
▪ Round Cost (€) to two decimal places.

Phase 2. **Constraint Analysis**
- Identify and list ALL operations belonging to the manual joint chain based on precedence dependencies, and use → to show the dependency relationships between these operations.
▪ Manual joint chain: operation_name_1 → ... → operation_name_n

- Identify and list ALL operations belonging to the automatic joint chain based on precedence dependencies, and use → to show the dependency relationships between these operations.
▪ Automatic joint chain: operation_name_1 → ... → operation_name_m

- Identify and list ALL shared operations that do not belong to either joint chain, and use → to show the dependency relationships between these operations.
▪ Shared operations:
operation_name_a → operation_name_b, operation_name_c, operation_name_d

Note:
▪ The listed operation sequences must strictly follow the precedence dependencies.

- Analyze and list all design constraints.
▪ Constraint 1: ...

Phase 3. **Strategy Decision**
- Determine the chain type for each quarter section (Manual joint chain / Automatic joint chain), considering resource availability.
▪ Quarter section 1: ...

- Determine the execution pattern of the quarter sections (Series / Parallel), ensuring that the required resources at any time do not exceed the available resource capacities.
▪ Execution pattern (example): 1 → 2 → (3 || 4)
Note:
▪ "→" indicates serial execution
▪ "||" indicates parallel execution

Phase 4. **Plan Generation**
- According to Phases 1, 2, and 3, generate a complete aircraft fuselage joint plan table following the Markdown format.
- Please strictly follow the chain type and execution pattern provided in Phase 3 when generating the table.
- The markdown table headers are: Order; Section; Operation; Type; Required Resources; Duration (min); Start Time (min); End Time (min); Cost (€).
- Markdown table format: header row, then separator row |---|---|---|, then data rows.
- Output the table directly as plain Markdown. Do NOT wrap the table in code blocks (no ```).
Note:
▪ Order: Use a single number (1, 2, 3, ...) for sequential steps. For parallel steps, all operations starting at the same scheduling step must share the same number and use different letter suffixes (4a, 4b, 4c, ...). The number identifies the scheduling step, and the letter identifies the parallel branch.
▪ Section: For operations belonging to manual or automatic joint chains, use "Quarter section 1", "Quarter section 2", "Quarter section 3", or "Quarter section 4", and for other operations, use "Shared".
▪ Operation: Use the full name of operations. Use the full name of operations exactly as given. Do not modify the operation names.
▪ Type: Use Manual/Automatic.
▪ Required Resources: Use name (number), and omission or abbreviation is not allowed, such as 'same as above'.
▪ Duration (min); Start Time (min); Cost (€): Use number only.
▪ All operations in the manual and automatic joint chains identified in Phase 2 must be fully included in Phase 4 without omission.
▪ Each quarter section must have its own complete joint chain in the table.
▪ The chain type and execution stages determined in Phase 3 must be strictly followed in Phase 4.
▪ Quarter sections 1, 2, 3 and 4 must appear in the table.
▪ All content in the table must not be omitted or abbreviated.
▪ All content in the table cannot include any formatting.
▪ Only generate one table.
- Calculate the total time and cost, and output them only after the table.

**Previous assembly plan table:**
{plan_table}

**Verification report:**
{json.dumps(verification_report, ensure_ascii=False, indent=2)}

**Human feedback:**
{human_feedback}
"""


def build_verification_prompt(operation_dependency, constraint_text, plan_text):
    return f"""
You are an expert for aircraft fuselage assembly plan verification.

Your task is to verify whether the generated aircraft fuselage joint plan satisfies the user-provided engineering constraints.

- Identify all violation operations and generate global repair advice based on the overall structure of the plan.
▪ Output the result as one JSON using exactly this structure:
{{
  "violation_list": [
    {{
      "id": "V1",
      "involved_operations": [
        "order_1: operation_name_1",
        "order_2: operation_name_2",
        ...
      ],
      "reason": "Why these operations violate the engineering constraints.",
      "evidence": "List the exact original text of the specific violated constraint."
    }}
  ],
  "repair_advice": [
    {{
      "id": "R1",
      "advice": "Provide a GLOBAL modification strategy for the entire plan. Explain which operations need to be added or removed."
    }}
  ]
}}

Note:
▪ Ensure that the order and operation name exist exactly in the Generated plan table.
▪ The "involved_operations" field should list only the specific violating operations, rather than a range.
▪ Do NOT evaluate operation cost, resource assignment correctness, or resource capacity usage.
▪ Only report violation operations that actually appear in the Generated plan table.
▪ Do NOT invent operations, orders, rows, or table content.
▪ If no violations are found, output an empty list

Input 1: Operation dependency knowledge
{operation_dependency}

Input 2: User-provided engineering constraints
{constraint_text}

Input 3: Generated plan table
{plan_text}
""".strip()

