// Aircraft assembly ontology import and knowledge-graph normalization
// Tested target: Neo4j Community 5.26.1, APOC 5.26.1, Neosemantics 5.20.0
// Run the statements section by section in Neo4j Browser.

// -----------------------------------------------------------------------------
// 1. Initialize Neosemantics and import the Turtle ontology
// Replace the file URL with the absolute path of your exported .ttl file.
// -----------------------------------------------------------------------------
CALL n10s.graphconfig.init({handleVocabUris: "MAP"});

CREATE CONSTRAINT n10s_unique_uri
FOR (r:Resource)
REQUIRE r.uri IS UNIQUE;

CALL n10s.onto.import.fetch(
  "file:///ABSOLUTE/PATH/aircraft_assembly_process_ontology.ttl",
  "Turtle",
  {handleVocabUris: "MAP"}
);

// -----------------------------------------------------------------------------
// 2. Repair imported names, labels, and ontology relationships
// -----------------------------------------------------------------------------
MATCH (n)
WHERE n.label IS NOT NULL
SET n.name = n.label;

MATCH (n)
REMOVE n:Resource;

MATCH (n:Property:Relationship)
REMOVE n:Relationship
RETURN n;

MATCH ()-[r:SCO_RESTRICTION]->()
WHERE r.onPropertyURI IS NOT NULL
MATCH (n:Relationship {uri: r.onPropertyURI})
SET r.onPropertyName = n.name;

MATCH ()-[r:SCO_RESTRICTION]->()
WHERE r.onPropertyURI IS NOT NULL
MATCH (n:Property {uri: r.onPropertyURI})
SET r.onPropertyName = n.name;

MATCH ()-[r:SCO]->()
WITH r
CALL apoc.refactor.setType(r, 'isSubClassOf')
YIELD input, output
RETURN input, output;

MATCH ()-[r:SPO]->()
WITH r
CALL apoc.refactor.setType(r, 'isSubPropertyOf')
YIELD input, output
RETURN input, output;

// -----------------------------------------------------------------------------
// 3. Add operation durations and operation types
// -----------------------------------------------------------------------------
MATCH (p:Class {name: 'S40_00001_Jig in'}) SET p.duration = 60, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_00002_Jig out'}) SET p.duration = 60, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_01001_Set up working environment'}) SET p.duration = 10, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_02001_Set in position Rails and LFT'}) SET p.duration = 10, p.op_type = 'Auto';
MATCH (p:Class {name: 'S40_04001_Camera at stating holes'}) SET p.duration = 15, p.op_type = 'Auto';
MATCH (p:Class {name: 'S40_04002_Drilling orbital 4,8'}) SET p.duration = 125, p.op_type = 'Auto';
MATCH (p:Class {name: 'S40_04008_Set up the fixations LGP/Hi-Lite automatic'}) SET p.duration = 20, p.op_type = 'Auto';
MATCH (p:Class {name: 'S40_04010_Riveting buttstraps and stabiliser automatic'}) SET p.duration = 90, p.op_type = 'Auto';
MATCH (p:Class {name: 'S40_04012_Deburring int, positioning, attach them automatic'}) SET p.duration = 25, p.op_type = 'Auto';
MATCH (p:Class {name: 'S40_04014_Deinstall LFT and rails'}) SET p.duration = 35, p.op_type = 'Auto';
MATCH (p:Class {name: 'S40_04003_Drilling template install'}) SET p.duration = 25, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_04004_Fixation drilling template suite manual'}) SET p.duration = 30, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_04005_Drilling (with adapter) 3,2 on Stringers int and drilling template'}) SET p.duration = 185, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_04006_Deinstall drilling template'}) SET p.duration = 25, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_04007_Set in position temporary fastener'}) SET p.duration = 15, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_04009_Set up the fixations LGP/Hi-Lite manual'}) SET p.duration = 35, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_04011_Riveting buttstraps and stabiliser manual'}) SET p.duration = 180, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_04013_Deburring int, positioning, attach them manual'}) SET p.duration = 45, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_02002_Cleanup and add sealant'}) SET p.duration = 35, p.op_type = 'Manual';
MATCH (p:Class {name: 'S40_02003_Inspection'}) SET p.duration = 55, p.op_type = 'Manual';

// -----------------------------------------------------------------------------
// 4. Add resource costs, calendars, and capacities
// -----------------------------------------------------------------------------
MATCH (p:Class {name: 'Light Flex Track Robot'}) SET p.cost_hour = 75, p.calendar = '24x7', p.number = 2;
MATCH (p:Class {name: 'Light Flex Track Rail'}) SET p.cost_hour = 5, p.calendar = '24x7', p.number = 2;
MATCH (p:Class {name: 'Station'}) SET p.cost_hour = 80, p.calendar = '24x7', p.number = 2;
MATCH (p:Class {name: 'Station platform'}) SET p.cost_hour = 75, p.calendar = '24x7', p.number = 2;
MATCH (p:Class {name: 'Mechanical Operator'}) SET p.cost_hour = 100, p.calendar = 'shift_40h_week', p.number = 8;
MATCH (p:Class {name: 'Automation Operator'}) SET p.cost_hour = 100, p.calendar = 'shift_40h_week', p.number = 8;
MATCH (p:Class {name: 'Hand drilling machine'}) SET p.cost_hour = 10, p.calendar = '24x7', p.number = 6;
MATCH (p:Class {name: 'Drilling Template'}) SET p.cost_hour = 10, p.calendar = '24x7', p.number = 5;
MATCH (p:Class {name: 'Crane'}) SET p.cost_hour = 30, p.calendar = '24x7', p.number = 1;
MATCH (p:Class {name: 'Transportation tooling'}) SET p.cost_hour = 80, p.calendar = '24x7', p.number = 1;

// -----------------------------------------------------------------------------
// 5. Replace operation-resource requirement relationships
// -----------------------------------------------------------------------------
MATCH (p:Class)-[r]->(o:Class)
WHERE r.onPropertyName = 'requiresResource' AND p.name STARTS WITH 'S40_0'
DELETE r;

MATCH (o1:Class) WHERE o1.name STARTS WITH 'S40_01001'
MATCH (o2:Class) WHERE o2.name STARTS WITH 'S40_02001'
MATCH (o3:Class) WHERE o3.name STARTS WITH 'S40_04001'
MATCH (o4:Class) WHERE o4.name STARTS WITH 'S40_04002'
MATCH (o5:Class) WHERE o5.name STARTS WITH 'S40_04003'
MATCH (o6:Class) WHERE o6.name STARTS WITH 'S40_04004'
MATCH (o7:Class) WHERE o7.name STARTS WITH 'S40_04005'
MATCH (o8:Class) WHERE o8.name STARTS WITH 'S40_04006'
MATCH (o9:Class) WHERE o9.name STARTS WITH 'S40_04007'
MATCH (o10:Class) WHERE o10.name STARTS WITH 'S40_04008'
MATCH (o11:Class) WHERE o11.name STARTS WITH 'S40_04009'
MATCH (o12:Class) WHERE o12.name STARTS WITH 'S40_04010'
MATCH (o13:Class) WHERE o13.name STARTS WITH 'S40_04011'
MATCH (o14:Class) WHERE o14.name STARTS WITH 'S40_04012'
MATCH (o15:Class) WHERE o15.name STARTS WITH 'S40_04013'
MATCH (o16:Class) WHERE o16.name STARTS WITH 'S40_04014'
MATCH (o17:Class) WHERE o17.name STARTS WITH 'S40_02002'
MATCH (o18:Class) WHERE o18.name STARTS WITH 'S40_02003'
MATCH (o19:Class) WHERE o19.name STARTS WITH 'S40_00001'
MATCH (o20:Class) WHERE o20.name STARTS WITH 'S40_00002'
MATCH (r11:Class {name: 'Mechanical Operator'})
MATCH (r12:Class {name: 'Automation Operator'})
MATCH (r2:Class {name: 'Light Flex Track Robot'})
MATCH (r3:Class {name: 'Light Flex Track Rail'})
MATCH (r4:Class {name: 'Hand drilling machine'})
MATCH (r5:Class {name: 'Drilling Template'})
MATCH (r6:Class {name: 'Station platform'})
MATCH (r7:Class {name: 'Station'})
MATCH (r8:Class {name: 'Crane'})
MATCH (r9:Class {name: 'Transportation tooling'})
WITH
  o1, o2, o3, o4, o5, o6, o7, o8, o9, o10,
  o11, o12, o13, o14, o15, o16, o17, o18, o19, o20,
  r11, r12, r2, r3, r4, r5, r6, r7, r8, r9
UNWIND [
  {op: o1, resources: [{node: r11, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o2, resources: [{node: r12, num: 2}, {node: r2, num: 1}, {node: r3, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o3, resources: [{node: r12, num: 1}, {node: r2, num: 1}, {node: r3, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o4, resources: [{node: r12, num: 1}, {node: r2, num: 1}, {node: r3, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o5, resources: [{node: r11, num: 2}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o6, resources: [{node: r11, num: 2}, {node: r5, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o7, resources: [{node: r11, num: 2}, {node: r4, num: 2}, {node: r5, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o8, resources: [{node: r11, num: 2}, {node: r5, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o9, resources: [{node: r11, num: 2}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o10, resources: [{node: r12, num: 1}, {node: r2, num: 1}, {node: r3, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o11, resources: [{node: r11, num: 2}]},
  {op: o12, resources: [{node: r12, num: 1}, {node: r2, num: 1}, {node: r3, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o13, resources: [{node: r11, num: 2}]},
  {op: o14, resources: [{node: r12, num: 1}, {node: r2, num: 1}, {node: r3, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o15, resources: [{node: r11, num: 2}]},
  {op: o16, resources: [{node: r12, num: 1}, {node: r2, num: 1}, {node: r3, num: 1}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o17, resources: [{node: r11, num: 2}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o18, resources: [{node: r11, num: 2}, {node: r6, num: 1}, {node: r7, num: 1}]},
  {op: o19, resources: [{node: r8, num: 1}, {node: r9, num: 1}]},
  {op: o20, resources: [{node: r8, num: 1}, {node: r9, num: 1}]}
] AS config
UNWIND config.resources AS res
WITH config.op AS op, res.node AS resource, res.num AS num
MERGE (op)-[:requiresResource {number: num}]->(resource);

// -----------------------------------------------------------------------------
// 6. Create application-facing labels and relationship types
// -----------------------------------------------------------------------------
MATCH (p:Class)-[r:requiresResource]->(o:Class)
WHERE p.name STARTS WITH 'S40_0'
SET p:Operation
SET o:Resource
RETURN p, r, o;

MATCH (c:Class)
WHERE c.name STARTS WITH 'S40'
  AND NOT (c:Resource OR c:Operation)
SET c:Process
RETURN c;

MATCH (a)-[r:SCO_RESTRICTION {onPropertyName: 'hasPredecessors'}]->(b)
CREATE (a)-[newRel:hasPredecessors]->(b)
SET newRel += properties(r)
DELETE r
RETURN newRel;

MATCH (a)-[r:SCO_RESTRICTION {onPropertyName: 'hasOptionalAutoOperation'}]->(b)
CREATE (a)-[newRel:hasOptionalAutoOperation]->(b)
SET newRel += properties(r)
DELETE r
RETURN newRel;

MATCH (a)-[r:SCO_RESTRICTION {onPropertyName: 'hasOptionalManualOperation'}]->(b)
CREATE (a)-[newRel:hasOptionalManualOperation]->(b)
SET newRel += properties(r)
DELETE r
RETURN newRel;

MATCH (a)-[r:SCO_RESTRICTION {onPropertyName: 'hasEssentialOperation'}]->(b)
CREATE (a)-[newRel:hasEssentialOperation]->(b)
SET newRel += properties(r)
DELETE r
RETURN newRel;

MATCH (a)-[r:SCO_RESTRICTION {onPropertyName: 'hasSubprocess'}]->(b)
CREATE (a)-[newRel:hasSubprocess]->(b)
SET newRel += properties(r)
DELETE r
RETURN newRel;

MATCH (n)
REMOVE n.label;

// Optional cleanup: remove URI values after all URI-based repairs are complete.
MATCH (n)
WHERE n.uri IS NOT NULL
REMOVE n.uri;

MATCH ()-[r]-()
WHERE r.onPropertyURI IS NOT NULL OR r.restrictionType IS NOT NULL
REMOVE r.onPropertyURI, r.restrictionType;

MATCH ()-[r]->()
WHERE r.onPropertyName IS NOT NULL
SET r.name = r.onPropertyName
REMOVE r.onPropertyName;

// Normalize resource names. The leading/trailing spaces in the source note were removed.
MATCH (r:Resource {name: 'Station platform'})
SET r.name = 'Station Platform'
RETURN r;

MATCH (r:Resource {name: 'Transportation tooling'})
SET r.name = 'Transportation Tooling'
RETURN r;

MATCH (r:Resource {name: 'Hand drilling machine'})
SET r.name = 'Hand Drilling Machine'
RETURN r;

// Optional final cleanup. Review the graph before running these statements.
// DROP CONSTRAINT n10s_unique_uri;
// MATCH (n) REMOVE n:Property:Relationship:`_GraphConfig`:Class;
// MATCH ()-[r]->()
// WHERE type(r) IN ['DOMAIN', 'RANGE', 'SCO_RESTRICTION', 'isSubClassOf', 'isSubPropertyOf']
// DELETE r;

