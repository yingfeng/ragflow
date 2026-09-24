import { CompilationTemplateKind } from '@/constants/compilation';

export type StructureTemplateKind = CompilationTemplateKind | 'raptor';

export interface IStructureGraphEntity {
  id?: string;
  name?: string;
  aliases?: string[];
  description?: string;
  discription?: string;
  mention_count?: number;
  source_chunk_ids?: string[];
  type?: string;
  /**
   * The ontology's datatype attributes — the instance's actual VALUES
   * (`{ birth_date: '1815-12-10', alias: ['Ada Augusta Byron'] }`). Absent for a
   * row compiled by a template that declares no datatype properties.
   */
  attributes?: Record<string, unknown>;
  /** Leaf clusters only: claims sourced to this cluster's chunks. */
  claim_count?: number;
  /** page_index fact/conclusion only: gate-verified verbatim quotes. */
  evidence?: IClaimEvidence[];
}

export interface IClaimEvidence {
  quote: string;
  chunk_id: string;
  start?: number;
  end?: number;
}

export interface IClaimItem {
  name: string;
  description?: string;
  source_chunk_ids?: string[];
  type?: string;
  evidence?: IClaimEvidence[];
}

export interface IClaimsResponse {
  claims: IClaimItem[];
  total: number;
  offset: number;
  limit: number;
}

export interface IStructureGraphRelation {
  from: string;
  to: string;
  type?: string;
}

/**
 * One class of the ontology model graph. `type` is the class name and is a
 * value, not a fixed column, so a user-edited template vocabulary needs no
 * client change. `declared` is false when the class shows up in the data but is
 * not declared by the template (vocabulary drift).
 */
export interface IOntologyClass {
  type: string;
  /** Display string; fall back to `type` when absent. */
  label?: string;
  description?: string;
  /** rdfs:subClassOf. Empty for a root class. */
  parent?: string[];
  /**
   * The class's effective DATATYPE properties: its own plus every ancestor's.
   * They are not edges (a datatype property has no class as its range), so this
   * is what the node detail panel lists.
   */
  attributes?: IOntologyAttribute[];
  entities: number;
  declared: boolean;
}

/** One datatype property as it applies to a specific class. */
export interface IOntologyAttribute {
  type: string;
  datatype?: string;
  label?: string;
  description?: string;
  /** The ancestor class that declares it; empty when the class declares it. */
  inherited_from?: string;
  /** How many instances of this class carry a value for it. */
  assertions: number;
}

/**
 * One OBJECT property edge of the ontology model graph. A property whose domain
 * or range is a "|"-separated list is expanded server-side into one edge per
 * domain x range combination.
 *
 * Datatype properties never appear here; see `IOntologyClass.attributes`.
 */
export interface IOntologyProperty {
  type: string;
  label?: string;
  description?: string;
  source: string;
  target: string;
  relations: number;
  declared: boolean;
}

/** rdfs:subClassOf, drawn as its own edge type so it can be toggled. */
export interface IOntologyInheritance {
  source: string;
  target: string;
}

/** One declaration-level finding, the equivalent of a pitfalls panel. */
export interface IOntologyPitfall {
  code: string;
  severity: 'error' | 'warning';
  message: string;
  subjects: string[];
}

export interface IOntologyGraph {
  classes: IOntologyClass[];
  properties: IOntologyProperty[];
  inheritance?: IOntologyInheritance[];
  pitfalls?: IOntologyPitfall[];
  total_entities: number;
  total_relations: number;
  /** The count scan hit its row cap, so the counts are lower bounds. */
  counts_truncated: boolean;
  /** Relations whose endpoints carry no class, so they are off the model graph. */
  unattributed_relations: number;
}

/**
 * One page of a class's instances, the drill-down behind a class node of the
 * ontology model graph.
 */
export interface IOntologyEntityPage {
  class: string;
  total: number;
  offset: number;
  limit: number;
  entities: IStructureGraphEntity[];
}

/**
 * One page of a property edge's assertions. `source` / `target` echo the edge
 * the page was asked for (empty when every domain -> range combination was
 * requested).
 */
export interface IOntologyRelationPage {
  property: string;
  source?: string;
  target?: string;
  total: number;
  offset: number;
  limit: number;
  relations: IStructureGraphRelation[];
}

export interface IStructureGraphTemplate {
  kind: StructureTemplateKind;
  template_id: string;
  template_name: string;
  entities: IStructureGraphEntity[];
  relations: IStructureGraphRelation[];
  /** Set for kind=ontology only: the class-level model graph. */
  ontology?: IOntologyGraph;
}

export interface IStructureGraphResponse {
  templates: IStructureGraphTemplate[];
  total_entities?: number;
  returned_entities?: number;
}
