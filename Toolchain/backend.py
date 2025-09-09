from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pyvis.network import Network
import uuid, re, csv, os, datetime, openai
from typing import Generator, Tuple, Optional
from openai import OpenAI

load_dotenv()
client = OpenAI()

llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_2 = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
llm_3 = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

graph = Neo4jGraph()
memory = ConversationBufferWindowMemory(k=10)

router_prompt = PromptTemplate(
    input_variables=["question"],
    template=""" 
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
)

router_chain = router_prompt | llm

cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    allow_dangerous_requests=True,
    verbose=True,
    exclude_types=[
        "Class", "Relationship", "_GraphConfig", "SCO_RESTRICTION",
        "DOMAIN", "RANGE", "isSubClassOf", "isSubPropertyOf", "hasOptionalAutoOperation",
        "hasOptionalManualOperation"
    ],
    top_k=300,
    return_direct=True,
    return_intermediate_steps=True
)

graph_response_prompt = PromptTemplate(
    input_variables=["question", "graph_data", "cypher"],
    template=""" 
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
)

graph_response_chain = graph_response_prompt | llm_2

design_qa_prompt = PromptTemplate(
    input_variables=["question", "history"],
    template="""
**Role**: You are an expert in aircraft fuselage assembly planning. Your task is to generate a complete and feasible assembly plan based only on the conversation history and user query.

**Query Context:**
Conversation history: {history}
User query: {question}

**Process Requirements**:
▪ Strictly output according to the following four phase structure, without omitting any part.
▪ All outputs must be based on data from the conversation history and user query.
Phase 1. **Data Extraction**
- Extract ALL operations and resources from conversation history, show them as a markdown table.
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

Phase 2. **Constraint Analysis**
- Analysis and list all constraints.
- Analysis and list automatic and manual quarter aircraft fuselage joint logic.

Phase 3. **Plan Generation**
- Generate a complete aircraft fuselage joint plan table following the Markdown format.
- The markdown table headers are: Order; Operation; Type; Required Resources; Duration (min); Start Time (min); End Time (min); Cost (€).
- Markdown table format: header row, then separator row |---|---|---|, then data rows.
Note:
▪ Order: Use a single number (1, 2, 3, ...) if it is executed sequentially, and use number + letter suffix (4a, 4b, 4c, ...) if it is executed in parallel.
▪ Operation: Use the full name of operations.
▪ Type: Use Manual/Automatic.
▪ Required Resources: Use (name (number)), just like Crane (1), Transportation Tooling (1), ..., and omission or abbreviation is not allowed, such as 'same as above'.
▪ Duration (min); Start Time (min); Cost (€): Use number only.
▪ All content in the table must not be omitted or abbreviated.
▪ All content in the table cannot include any formatting.
▪ Only generate one table.
- Calculate the total due time and cost.

Phase 4. **Validation Report**:
- Check if the following conditions are met. Mark ✓ if met, and ✗ if not met.
   ▪ [✓/✗] Completed 4 joints of 1/4 body
   ▪ [✓/✗] Meets all constraints
   ▪ [✓/✗] Shared operations correctly positioned
   ▪ [✓/✗] The required resources at the current moment do not exceed the total number of resources
"""
)

design_qa_chain = design_qa_prompt | llm_3

def extract_phase3_table(text):
    print("\n==== [DEBUG] Start extracting Phase 3 blocks ====")
    phase3_pat = (
        r"(?:\s*(?:#+|\*\*)\s*)?Phase\s*3[^\n]*\n"
        r"([\s\S]+?)"
        r"(?=(?:\s*(?:#+|\*\*)\s*)?Phase\s*4|\Z)"
    )
    m = re.search(phase3_pat, text, re.IGNORECASE | re.MULTILINE)
    if not m:
        print('[DEBUG] Phase 3 Regular miss！')
        return None
    phase3_block = m.group(1).strip()
    print("[DEBUG] Phase 3 Preview at the beginning of the block:\n", phase3_block[:120], "..." if len(phase3_block) > 120 else "")

    print("\n==== [DEBUG] Attempt to strictly extract Markdown tables ====")
    table_pat = re.compile(
        r'((?:\|[^\n]*?\|[^\n]*?\n)+?)'
        r'((?:\|\s*[-:]+\s*){2,}\|[^\n]*?\n)'
        r'((?:(?:\|[^\n]*?\|[^\n]*?\n)+)+)',
        re.MULTILINE
    )
    t = table_pat.search(phase3_block)
    if t:
        table_text = (t.group(1) + t.group(2) + t.group(3)).strip()
        print("[DEBUG] Strict mode hits table, table preview:\n", table_text[:180], "..." if len(table_text)>180 else "")
        return table_text
    else:
        print('[DEBUG] Strict Markdown table miss, enter loose matching ..')

    greedy_lines = [ln for ln in phase3_block.split("\n") if re.match(r'^\|.*\|\s*', ln)]
    print(f"[DEBUG] Number of hit rows in fallback mode: {len(greedy_lines)}")
    if len(greedy_lines) >= 2:
        for i, ln in enumerate(greedy_lines[:3]):
            print(f"[DEBUG] fallback行{i}: {ln}")
        return "\n".join(greedy_lines)
    print("[DEBUG] Fallback does not have enough table rows")
    return None

def clean_markdown_table(markdown_table_text):
    print("\n==== [DEBUG] Enter table cleaning ====")
    print("[DEBUG] Truncate the content of the input table:\n", markdown_table_text[:200], "..." if len(markdown_table_text) > 200 else "")

    lines = [
        ln.strip() for ln in markdown_table_text.strip().split('\n')
        if "|" in ln
           and not re.match(r'^\s*\|?[\s:\-\|]+\|?\s*$', ln)
                            and not re.match(r'^\s*(Total)', ln, re.I)
    ]

    print(f"[DEBUG] Remaining valid table rows: {len(lines)}")
    for i, l in enumerate(lines[:3]):
        print(f"[DEBUG] 行{i}: {l}")

    if len(lines) < 2:
        print("[DEBUG] The number of valid rows is less than 2, and the table cleaning has failed")
        return []

    col_cnt = max(len([c for c in row.split("|") if c.strip()]) for row in lines)
    print(f"[DEBUG] Automatically identify the number of columns: {col_cnt}")

    table = []
    for i, row in enumerate(lines):
        cells = [c.strip() for c in row.strip("|").split("|")]
        while len(cells) < col_cnt:
            cells.append("")
        if i < 3:
            print(f"[DEBUG] Parsed line{i}: {cells}")
        table.append(cells)
    return table

def generate_graph_html(graph_data):
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    node_records = {}

    data_format = detect_data_format(graph_data)

    format_handlers = {
        "A": process_format_a,
        "B": process_format_b,
        "C": process_format_c,
        "D": process_format_d
    }

    handler = format_handlers.get(data_format)
    if handler:
        handler(net, graph_data, node_records)

    configure_network(net)
    return save_network(net)

def add_node_if_absent(net, node_records, node_id, label=None, color="#97c2fc", shape="box"):
    if node_id not in node_records:
        net.add_node(node_id, label=label or node_id, color=color, shape=shape)
        node_records[node_id] = True

def process_format_a(net, data, node_records):
    for item in data:
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, dict):
                    node_id = str(value.get("name", uuid.uuid4()))
                    add_node_if_absent(net, node_records, node_id, label=generate_attribute_label(value))

def process_format_b(net, data, node_records):
    if not data or not isinstance(data[0], dict):
        return

    keys = list(data[0].keys())[:2]
    if len(keys) < 2:
        return

    field1, field2 = keys
    for item in data:
        node1 = str(item.get(field1, ""))
        node2 = str(item.get(field2, ""))
        if node1 and node2:
            add_node_if_absent(net, node_records, node1, color="#97c2fc", shape="box")
            add_node_if_absent(net, node_records, node2, color="#fc9797", shape="box")
            net.add_edge(node1, node2, color="#666666")

def process_format_c(net, data, node_records):
    if not data or not isinstance(data[0], dict):
        return

    sample = data[0]
    dict_field = next((k for k, v in sample.items() if isinstance(v, dict)), None)
    if not dict_field:
        return

    for item in data:
        dict_data = item.get(dict_field, {})
        node1 = str(dict_data.get("name", uuid.uuid4()))
        add_node_if_absent(net, node_records, node1, label=generate_attribute_label(dict_data))

        other_keys = [k for k in item.keys() if k != dict_field]
        if not other_keys:
            continue

        node2_field = other_keys[0]
        node2 = str(item.get(node2_field, ""))
        add_node_if_absent(net, node_records, node2, color="#fc9797", shape="diamond")

        rel_field = next((k for k in other_keys if k != node2_field), None)
        rel = str(item.get(rel_field, "")) if rel_field else ""
        net.add_edge(node1, node2, label=rel, color="#666666")

def process_format_d(net, data, node_records):
    if not data or not isinstance(data[0], dict):
        return

    fields = list(data[0].keys())
    if len(fields) < 3:
        return

    node1_field, node2_field, rel_field = fields[:3]

    for item in data:
        node1 = str(item.get(node1_field, ""))
        node2 = str(item.get(node2_field, ""))
        rel = str(item.get(rel_field, ""))
        if node1 and node2:
            add_node_if_absent(net, node_records, node1)
            add_node_if_absent(net, node_records, node2)
            net.add_edge(node1, node2, label=rel, color="#666666")

def generate_attribute_label(node_data):
    name = node_data.get("name", "Node")
    attrs = "\n".join(f"{k}: {v}" for k, v in node_data.items() if k != "name")
    return f"{name}\n{attrs}" if attrs else name

def detect_data_format(data):
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        return "A"

    sample = data[0]
    dict_fields = [k for k, v in sample.items() if isinstance(v, dict)]
    if len(dict_fields) == 1 and len(sample) >= 3:
        return "C"
    elif len(dict_fields) == 0:
        if len(sample) == 2:
            return "B"
        elif len(sample) >= 3:
            return "D"
    return "A"

def configure_network(net):
    net.toggle_physics(False)
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        },
        "nodes": {
            "font": {
                "size": 14
            }
        }
    }
    """)

def save_network(net):
    static_dir = os.path.join(os.getcwd(), "static")
    os.makedirs(static_dir, exist_ok=True)
    filename = f"graph_{uuid.uuid4().hex}.html"
    filepath = os.path.join(static_dir, filename)
    net.save_graph(filepath)
    return f"/static/{filename}"

def get_graph_html(data):
    net = Network(height='500px', width='100%', notebook=False, directed=True)
    for item in data:
        r = item.get('r', {})
        name = r.get('name', 'Unknown')
        net.add_node(name, label=name)
    return net.generate_html()

def graph_answer_token_stream(question: str, graph_data, cypher: str):
    prompt_text = graph_response_prompt.format(question=question, graph_data=graph_data, cypher=cypher)
    stream = client.chat.completions.create(
        model      = "gpt-4o",
        messages   = [{"role": "user", "content": prompt_text}],
        temperature= 0,
        stream     = True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def design_answer_token_stream(question: str, history: str):
    prompt_text = design_qa_prompt.format(question=question, history=history)

    stream = client.chat.completions.create(
        model      = "o1",
        messages   = [{"role": "user", "content": prompt_text}],
        temperature= 0,
        stream     = True
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            yield token

def smart_qa_system(question: str) -> Generator[Tuple[str, Optional[str]], None, None]:
    history = memory.load_memory_variables({}).get('history', '')
    graph_html_path = None
    answer_full = ""

    try:
        resp_type = router_chain.invoke({"question": question}).content.strip().lower()
        resp_type = resp_type.strip('"\' ')

        if resp_type == "graph":

            cypher_output = cypher_chain.invoke({"query": question})
            graph_data    = cypher_output.get("result", cypher_output)
            cypher        = cypher_output.get("intermediate_steps", [{}])[0].get("query", "").replace("cypher", "").strip()

            graph_html_path = generate_graph_html(graph_data)

            answer_full = ""

            yield "", graph_html_path
            for tok in graph_answer_token_stream(question, graph_data, cypher):
                answer_full += tok
                yield tok, graph_html_path

        else:

            yield "", None

            for tok in design_answer_token_stream(question, history):
                answer_full += tok
                yield tok, None

            csv_msg = ""
            markdown_table = extract_phase3_table(answer_full)
            if markdown_table:
                try:
                    cleaned_rows = clean_markdown_table(markdown_table)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("./plans", exist_ok=True)
                    csv_path = f"./plans/assembly_plan_{timestamp}.csv"
                    with open(csv_path, "w", newline='', encoding="utf-8-sig") as f:
                        csv.writer(f).writerows(cleaned_rows)
                    csv_msg = f"\n\n✅ **The assembly plan has been saved as a CSV file:** `{csv_path}`"
                except Exception as e:
                    csv_msg = "\n⚠️ CSV generation failed."
            else:
                csv_msg = "\n⚠️ No formatting compliant Markdown table detected, CSV not saved."

            if csv_msg:
                answer_full += csv_msg
                yield csv_msg, None

        memory.save_context({"input": question}, {"output": answer_full})

    except Exception as e:
        err = f"Sorry, there was an error while processing your issue：{e}"
        print("【ERROR】", e)
        yield err, None

if __name__ == "__main__":
    print("The knowledge graph question answering system has been launched. Enter 'exit' or 'quit' to exit.")
    while True:
        try:
            user_input = input("\nUser query: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            response = smart_qa_system(user_input)
            print("\nAnswer:", response)
        except Exception as e:
            print(f"Error: {str(e)}")
            print("The system will continue to operate ...")