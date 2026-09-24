//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package service

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"ragflow/internal/dao"
	"ragflow/internal/engine/types"
)

// loadTemplateConfig reads a template's config. The declared ontology lives
// there, so a missing template simply yields no declared skeleton (the observed
// classes and properties still come through).
func loadTemplateConfig(ctx context.Context, templateDAO *dao.CompilationTemplateDAO, tenantID, templateID string) map[string]interface{} {
	template, err := templateDAO.GetTemplate(ctx, dao.DB, tenantID, templateID)
	if err != nil || template == nil {
		return nil
	}
	return map[string]interface{}(template.Config)
}

// The ontology model graph. Unlike the entity/relation graph, its nodes are
// CLASSES and its edges are PROPERTIES, which is the same shape OntoBricks'
// ontology viewer draws (nodes = classes, edges = a property's domain → range).
//
// The skeleton (which classes and properties exist, and which class each
// property runs between) is declared by the compilation template, so it is read
// from the template config and needs no index query at all. The engines only
// supply the annotation: how many instances of each class and each property edge
// were actually compiled.
//
// Counts are accumulated by paging over the rows and counting in Go because the
// engines expose no group-by: SearchRequest carries filters, match expressions
// and ordering, and SearchResult exposes only the rows plus a total.

// ontologyCountPageSize is the page size used while scanning rows for counts.
const ontologyCountPageSize = 2000

// ontologyCountScanCap bounds how many rows one ontology count scan reads. A
// scope with more rows than this reports CountsTruncated instead of scanning
// unbounded.
const ontologyCountScanCap = 20000

// OntologyClassNode is one class of the model graph.
type OntologyClassNode struct {
	// Type is the class name. It is a value, never a column name, so the
	// template's vocabulary is free to change without touching the schema.
	Type string `json:"type"`
	// Label is the template's display string for the class. It is what a
	// viewer labels the node with, falling back to Type when absent — the same
	// fallback OntoBricks' ontology viewer uses (`cls.label || cls.name`).
	Label       string `json:"label,omitempty"`
	Description string `json:"description,omitempty"`
	// Parent is rdfs:subClassOf (immediate superclasses). The acyclic guarantee
	// comes from ValidateTemplatePayload, not from here.
	Parent []string `json:"parent,omitempty"`
	// Attributes is the class's effective DATATYPE properties: its own plus every
	// ancestor's, which is standard-ontology inheritance resolved without engine
	// reasoning (ontology.md §6.5 A3). They are not drawn as edges — a datatype
	// property has no class as its range — so this list is what the node detail
	// panel shows.
	Attributes []OntologyAttribute `json:"attributes,omitempty"`
	// Entities is how many instances of this class were compiled in the scope.
	Entities int `json:"entities"`
	// Declared is false for a class that appears in the data but is not declared
	// by the template — surfaced so vocabulary drift is visible rather than
	// silently merged into the declared graph.
	Declared bool `json:"declared"`
}

// OntologyAttribute is one datatype property as it applies to a specific class.
type OntologyAttribute struct {
	Type string `json:"type"`
	// Datatype is the value type the template declares (string / date / list /
	// float / …), empty when the template leaves it open.
	Datatype    string `json:"datatype,omitempty"`
	Label       string `json:"label,omitempty"`
	Description string `json:"description,omitempty"`
	// InheritedFrom is the ancestor class that declares the attribute, empty
	// when this class declares it itself. Reported so the panel can separate
	// "own" from "inherited" without re-walking the parent chain.
	InheritedFrom string `json:"inherited_from,omitempty"`
	// Assertions is how many instances OF THIS CLASS carry a value for the
	// attribute. It is counted from the same entity scan that counts instances,
	// so it costs no extra query — and it is information a pure ontology
	// configuration (OntoBricks' viewer) cannot have.
	Assertions int `json:"assertions"`
}

// OntologyInheritanceEdge is rdfs:subClassOf drawn as its own edge type, the way
// OntoBricks' viewer renders inheritance separately from properties (and lets it
// be toggled off).
type OntologyInheritanceEdge struct {
	// Source is the superclass, Target the subclass.
	Source string `json:"source"`
	Target string `json:"target"`
}

// OntologyPitfall is one declaration-level defect, the equivalent of OntoBricks'
// Pitfalls panel. Every check here reads the template alone except orphan_class,
// which needs one count that the instance scan already produced.
type OntologyPitfall struct {
	Code string `json:"code"`
	// Severity is "error" for a declaration that cannot be satisfied and
	// "warning" for a gap that only costs coverage.
	Severity string `json:"severity"`
	Message  string `json:"message"`
	// Subjects names the classes / properties the finding is about.
	Subjects []string `json:"subjects"`
}

// OntologyPropertyEdge is one OBJECT property of the model graph. A property
// with a polymorphic domain or range yields one edge per domain × range
// combination, so the graph stays a plain class-to-class edge list.
//
// Datatype properties are deliberately absent: they have no class as their
// range, so there is no edge to draw. They appear on the owning class's
// Attributes instead, which is exactly how a viewer that reads only an ontology
// configuration draws them.
type OntologyPropertyEdge struct {
	// Type is the property name (a value).
	Type string `json:"type"`
	// Label is the template's display string, falling back to Type on the client.
	Label       string `json:"label,omitempty"`
	Description string `json:"description,omitempty"`
	// Source and Target are the endpoint classes of this edge.
	Source string `json:"source"`
	Target string `json:"target"`
	// Relations is how many instances of this property edge were compiled in the
	// scope.
	Relations int `json:"relations"`
	// Declared is false for a property observed in the data but not declared by
	// the template.
	Declared bool `json:"declared"`
}

// OntologyGraph is the ontology view of one template bucket.
//
// It is a complete description of the picture a viewer draws: nodes are classes,
// edges are OBJECT properties (a datatype property has no class as its range, so
// it is not an edge) plus a separate inheritance edge list, and every node and
// edge carries the instance count the engines supplied.
type OntologyGraph struct {
	Classes    []OntologyClassNode    `json:"classes"`
	Properties []OntologyPropertyEdge `json:"properties"`
	// Inheritance is rdfs:subClassOf, kept out of Properties so a viewer can
	// style it and toggle it independently.
	Inheritance []OntologyInheritanceEdge `json:"inheritance,omitempty"`
	// Pitfalls are declaration-level findings. Present so the view can surface
	// vocabulary drift instead of drawing a silently wrong graph.
	Pitfalls []OntologyPitfall `json:"pitfalls,omitempty"`
	// TotalEntities / TotalRelations are the scope-wide row counts, which may
	// exceed the summed per-class / per-edge counts when rows carry no class.
	TotalEntities  int `json:"total_entities"`
	TotalRelations int `json:"total_relations"`
	// CountsTruncated reports that the count scan stopped at
	// ontologyCountScanCap, so the per-class / per-edge counts are lower bounds.
	CountsTruncated bool `json:"counts_truncated"`
	// UnattributedRelations counts relations whose endpoints have no class
	// stamp, so they cannot be placed on the class-level graph. Reported so the
	// per-edge counts reconcile against TotalRelations.
	UnattributedRelations int `json:"unattributed_relations"`
}

// declaredOntology is the template-declared half of the model graph.
type declaredOntology struct {
	// classOrder preserves the template's declaration order; classes is keyed by
	// class name.
	classOrder []string
	classes    map[string]declaredClass
	// props is keyed by property name, preserving declaration order.
	propOrder []string
	props     map[string]declaredProperty
}

// declaredClass is one class of the TBox.
type declaredClass struct {
	label       string
	description string
	// parents is rdfs:subClassOf. ValidateTemplatePayload already rejects an
	// unknown parent and a cyclic chain, so the walk in effectiveAttributes
	// terminates; it still guards itself, because a config edited outside the
	// service must not hang the API.
	parents []string
}

type declaredProperty struct {
	label       string
	description string
	// kind is "object" or "datatype" as declared. Empty means the template did
	// not say, so it is inferred from the endpoints (see isObject/isDatatype).
	kind     string
	datatype string
	domains  []string
	ranges   []string
}

// isObject reports whether the property joins two individuals, which is what
// makes it an edge. A property that declares no range at all is NOT treated as
// an object property: the template never said what it connects, and inventing an
// endpoint would fabricate an edge. That is the pre-existing behaviour of the
// edge list, kept deliberately.
func (p declaredProperty) isObject() bool {
	if p.kind == "object" {
		return true
	}
	if p.kind == "datatype" {
		return false
	}
	return len(p.ranges) > 0
}

// isDatatype reports whether the property's values are literals, which is what
// makes it a node attribute rather than an edge.
func (p declaredProperty) isDatatype() bool {
	if p.kind == "datatype" {
		return true
	}
	if p.kind == "object" {
		return false
	}
	return p.datatype != "" && len(p.ranges) == 0
}

// parseDeclaredOntology reads the classes and properties a template declares.
// Only the template shape (config.entity.fields / config.relation.fields) is
// read; a property's domain and range are the structured declarations, never
// parsed out of rule prose.
func parseDeclaredOntology(cfg map[string]interface{}) declaredOntology {
	out := declaredOntology{
		classes: map[string]declaredClass{},
		props:   map[string]declaredProperty{},
	}
	entitySection := jsonMapAt(cfg, "entity")
	for _, raw := range jsonListAt(entitySection, "fields") {
		field, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		name := strings.TrimSpace(jsonStringAt(field, "type"))
		if name == "" {
			continue
		}
		if _, seen := out.classes[name]; !seen {
			out.classOrder = append(out.classOrder, name)
		}
		out.classes[name] = declaredClass{
			label:       strings.TrimSpace(jsonStringAt(field, "label")),
			description: strings.TrimSpace(jsonStringAt(field, "description")),
			parents:     splitTypeList(jsonStringAt(field, "parent")),
		}
	}

	relationSection := jsonMapAt(cfg, "relation")
	for _, raw := range jsonListAt(relationSection, "fields") {
		field, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		name := strings.TrimSpace(jsonStringAt(field, "type"))
		if name == "" {
			continue
		}
		if _, seen := out.props[name]; !seen {
			out.propOrder = append(out.propOrder, name)
		}
		out.props[name] = declaredProperty{
			label:       strings.TrimSpace(jsonStringAt(field, "label")),
			description: strings.TrimSpace(jsonStringAt(field, "description")),
			kind:        strings.TrimSpace(jsonStringAt(field, "kind")),
			datatype:    strings.TrimSpace(jsonStringAt(field, "datatype")),
			domains:     splitTypeList(jsonStringAt(field, "domain")),
			ranges:      splitTypeList(jsonStringAt(field, "range")),
		}
	}
	return out
}

// ancestorChain walks `parent` upward, nearest first, excluding the class itself.
// It stops on the first repeat, so a cyclic config yields a finite chain instead
// of spinning (the template validator rejects cycles, but this must not depend on
// having been called).
func (d declaredOntology) ancestorChain(class string) []string {
	var chain []string
	seen := map[string]bool{class: true}
	for _, parent := range d.classes[class].parents {
		if seen[parent] {
			continue
		}
		seen[parent] = true
		chain = append(chain, parent)
		chain = append(chain, d.ancestorChainFrom(parent, seen)...)
	}
	return chain
}

// ancestorChainFrom is the recursive half of ancestorChain; `seen` is shared so
// a diamond (two parents with a common ancestor) reports that ancestor once.
func (d declaredOntology) ancestorChainFrom(class string, seen map[string]bool) []string {
	var chain []string
	for _, parent := range d.classes[class].parents {
		if seen[parent] {
			continue
		}
		seen[parent] = true
		chain = append(chain, parent)
		chain = append(chain, d.ancestorChainFrom(parent, seen)...)
	}
	return chain
}

// attributeCounts is per-class, per-attribute: how many instances of the class
// carry a value. Empty when the scan was skipped.
type attributeCounts map[string]map[string]int

// effectiveAttributes returns the datatype properties a class carries: the ones
// it declares, then the ones it inherits, each deduplicated by name with the
// nearest declarer winning. Standard-ontology semantics — a subclass inherits
// its superclass's properties — and the engines have no reasoning, so the walk
// happens here (ontology.md §6.5 A3).
//
// Assertions is always counted against the class being described, never against
// the declarer: the value is stored on the instance row, so "how many persons
// have an alias" is the question a node panel asks, even though `agent` declares
// `alias`.
func (d declaredOntology) effectiveAttributes(class string, counts attributeCounts) []OntologyAttribute {
	out := make([]OntologyAttribute, 0, 4)
	seen := map[string]bool{}
	self := true
	chain := append([]string{class}, d.ancestorChain(class)...)
	for _, owner := range chain {
		inheritedFrom := ""
		if !self {
			inheritedFrom = owner
		}
		for _, name := range d.propOrder {
			prop := d.props[name]
			if !prop.isDatatype() || seen[name] || !containsStr(prop.domains, owner) {
				continue
			}
			seen[name] = true
			out = append(out, OntologyAttribute{
				Type:          name,
				Datatype:      prop.datatype,
				Label:         prop.label,
				Description:   prop.description,
				InheritedFrom: inheritedFrom,
				Assertions:    counts[class][name],
			})
		}
		self = false
	}
	return out
}

// classHasProperty reports whether any declared property touches the class, as
// its domain, as its range, or as an inherited datatype attribute.
func (d declaredOntology) classHasProperty(class string) bool {
	for name := range d.props {
		prop := d.props[name]
		if containsStr(prop.domains, class) || containsStr(prop.ranges, class) {
			return true
		}
	}
	for _, ancestor := range append([]string{class}, d.ancestorChain(class)...) {
		for name := range d.props {
			prop := d.props[name]
			if prop.isDatatype() && containsStr(prop.domains, ancestor) {
				return true
			}
		}
	}
	return false
}

// buildOntologyPitfalls runs the declaration-level checks, the equivalent of
// OntoBricks' Pitfalls panel for the findings this design can make
// (ontology.md §6.5 A5/A6, §10.8.7). Every check reads the template alone except
// orphan_class, which needs the instance count the caller already has.
func buildOntologyPitfalls(d declaredOntology, classCounts map[string]int) []OntologyPitfall {
	var out []OntologyPitfall

	// Dangling endpoints: a declared endpoint that is not a declared class. The
	// prompt renders the class list as a closed set, so this can only come from
	// a template edited by hand — and it silently narrows the model's choices,
	// which is why it is an error rather than a warning.
	var danglingProps, danglingClasses []string
	for _, name := range d.propOrder {
		prop := d.props[name]
		var unknown []string
		for _, endpoint := range append(append([]string{}, prop.domains...), prop.ranges...) {
			if _, ok := d.classes[endpoint]; !ok {
				unknown = append(unknown, endpoint)
			}
		}
		if len(unknown) == 0 {
			continue
		}
		danglingProps = append(danglingProps, name)
		danglingClasses = append(danglingClasses, unknown...)
	}
	if len(danglingProps) > 0 {
		out = append(out, OntologyPitfall{
			Code:     "dangling_property",
			Severity: "error",
			Message: fmt.Sprintf("declared property/ies name an endpoint class the template does not declare: %s -> %s",
				strings.Join(danglingProps, ", "),
				strings.Join(sortedUnique(danglingClasses), ", ")),
			Subjects: danglingProps,
		})
	}

	// A class whose declared parent is not a declared class. ValidateTemplatePayload
	// rejects this, so it means the config was edited outside the service — but the
	// inheritance edge list drops such an edge (there is no node to point at), and
	// a silently missing parent edge would look like a template that never
	// declared one.
	var danglingChildren []string
	for _, class := range d.classOrder {
		for _, parent := range d.classes[class].parents {
			if _, ok := d.classes[parent]; !ok {
				danglingChildren = append(danglingChildren, class+" -> "+parent)
			}
		}
	}
	if len(danglingChildren) > 0 {
		out = append(out, OntologyPitfall{
			Code:     "dangling_parent",
			Severity: "error",
			Message:  "declared class(es) whose parent is not a declared class: " + strings.Join(danglingChildren, ", "),
			Subjects: danglingChildren,
		})
	}

	// A class no property reaches: it can only ever appear as an isolated node,
	// and the prompt gives the model no attribute to fill for it.
	var noProperty, orphans []string
	for _, class := range d.classOrder {
		if d.classHasProperty(class) {
			continue
		}
		noProperty = append(noProperty, class)
		if classCounts[class] == 0 {
			orphans = append(orphans, class)
		}
	}
	if len(noProperty) > 0 {
		out = append(out, OntologyPitfall{
			Code:     "class_without_property",
			Severity: "warning",
			Message:  "declared class(es) that no property touches, own or inherited: " + strings.Join(noProperty, ", "),
			Subjects: noProperty,
		})
	}
	if len(orphans) > 0 {
		out = append(out, OntologyPitfall{
			Code:     "orphan_class",
			Severity: "warning",
			Message:  "declared class(es) with neither a property nor a single compiled instance: " + strings.Join(orphans, ", "),
			Subjects: orphans,
		})
	}
	return out
}

// splitTypeList splits a "|"-separated class list. A property that is
// legitimately polymorphic declares every class it may run from or to, so the
// graph can expand it into one edge per combination without any prose parsing.
func splitTypeList(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, "|")
	out := make([]string, 0, len(parts))
	seen := map[string]bool{}
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out
}

// buildOntologyGraph merges the template's declared ontology with the instances
// observed in the scope. baseFilter carries the scope and template predicates;
// the knowledge_graph_kwd discriminator is added per scan.
func (s *DatasetArtifactService) buildOntologyGraph(
	ctx context.Context,
	tenantID, datasetID string,
	baseFilter map[string]interface{},
	cfg map[string]interface{},
) (*OntologyGraph, error) {
	declared := parseDeclaredOntology(cfg)

	entityFilter := withKnowledgeGraphKWD(baseFilter, []string{"entity"})
	relationFilter := withKnowledgeGraphKWD(baseFilter, []string{"relation"})

	_, entityTotal, err := graphRowSearch(ctx, tenantID, datasetID, []string{"id"}, entityFilter, nil, 0, 1, nil)
	if err != nil {
		return nil, err
	}
	_, relationTotal, err := graphRowSearch(ctx, tenantID, datasetID, []string{"id"}, relationFilter, nil, 0, 1, nil)
	if err != nil {
		return nil, err
	}

	classCounts, attributeCounts, truncated, err := countOntologyClasses(ctx, tenantID, datasetID, entityFilter)
	if err != nil {
		return nil, err
	}
	edgeCounts, relTruncated, unattributed, err := countOntologyEdges(ctx, tenantID, datasetID, relationFilter)
	if err != nil {
		return nil, err
	}
	truncated = truncated || relTruncated

	graph := &OntologyGraph{
		TotalEntities:         int(entityTotal),
		TotalRelations:        int(relationTotal),
		CountsTruncated:       truncated,
		UnattributedRelations: unattributed,
		Classes:               mergeOntologyClasses(declared, classCounts, attributeCounts),
		Properties:            mergeOntologyProperties(declared, edgeCounts),
		Inheritance:           mergeOntologyInheritance(declared),
		Pitfalls:              buildOntologyPitfalls(declared, classCounts),
	}
	return graph, nil
}

// mergeOntologyInheritance turns each class's declared parent into one edge per
// parent. A parent the template does not declare is skipped, because a viewer
// cannot draw an edge to a node that does not exist; the config is rejected by
// ValidateTemplatePayload in that case anyway, and buildOntologyPitfalls reports
// it when it somehow slips through.
func mergeOntologyInheritance(d declaredOntology) []OntologyInheritanceEdge {
	var out []OntologyInheritanceEdge
	for _, child := range d.classOrder {
		for _, parent := range d.classes[child].parents {
			if _, declared := d.classes[parent]; !declared {
				continue
			}
			out = append(out, OntologyInheritanceEdge{Source: parent, Target: child})
		}
	}
	return out
}

// ontologyEdgeKey identifies one property edge by property and endpoint classes.
func ontologyEdgeKey(prop, source, target string) string {
	return prop + "\x00" + source + "\x00" + target
}

// countOntologyClasses scans the scope's entity rows and counts them per class,
// and — from the same rows, at no extra query cost — per class and attribute.
//
// The attribute tally rides along because the `attr` json column is already on
// the row being read: "how many persons carry an alias" is then a count over
// data the scan had to fetch anyway, and the node panel of the ontology view
// gets a real number (ontology.md §10.8.2) instead of one requiring the
// not-yet-wired JSON path filter of §8.
func countOntologyClasses(
	ctx context.Context, tenantID, datasetID string, filter map[string]interface{},
) (counts map[string]int, attrs attributeCounts, truncated bool, err error) {
	counts = map[string]int{}
	attrs = attributeCounts{}
	scanned := 0
	for offset := 0; ; offset += ontologyCountPageSize {
		page, _, searchErr := graphRowSearch(ctx, tenantID, datasetID,
			[]string{"id", "entity_type_kwd", "attr"}, filter, nil, offset, ontologyCountPageSize, nil)
		if searchErr != nil {
			return nil, nil, false, searchErr
		}
		if len(page) == 0 {
			return counts, attrs, false, nil
		}
		for _, row := range page {
			scanned++
			typ := strings.TrimSpace(firstStringValue(row["entity_type_kwd"]))
			if typ == "" {
				continue
			}
			counts[typ]++
			for name, value := range attrValues(row) {
				if value == nil {
					continue
				}
				if attrs[typ] == nil {
					attrs[typ] = map[string]int{}
				}
				attrs[typ][name]++
			}
		}
		if scanned >= ontologyCountScanCap {
			return counts, attrs, true, nil
		}
		if len(page) < ontologyCountPageSize {
			return counts, attrs, false, nil
		}
	}
}

// attrValues reads one row's `attr` json column. The engines hand a json column
// back either as its serialized text or already decoded, depending on the engine
// and the driver, so both shapes are accepted.
func attrValues(row map[string]interface{}) map[string]interface{} {
	switch v := row["attr"].(type) {
	case map[string]interface{}:
		return v
	case string:
		if strings.TrimSpace(v) == "" {
			return nil
		}
		var m map[string]interface{}
		if err := json.Unmarshal([]byte(v), &m); err != nil {
			return nil
		}
		return m
	default:
		raw := firstStringValue(row["attr"])
		if strings.TrimSpace(raw) == "" {
			return nil
		}
		var m map[string]interface{}
		if err := json.Unmarshal([]byte(raw), &m); err != nil {
			return nil
		}
		return m
	}
}

// countOntologyEdges scans the scope's relation rows and counts them per
// (property, source class, target class). Relations with an unstamped endpoint
// cannot be placed on the class-level graph and are returned separately.
func countOntologyEdges(
	ctx context.Context, tenantID, datasetID string, filter map[string]interface{},
) (counts map[string]int, truncated bool, unattributed int, err error) {
	counts = map[string]int{}
	scanned := 0
	for offset := 0; ; offset += ontologyCountPageSize {
		page, _, searchErr := graphRowSearch(ctx, tenantID, datasetID,
			[]string{"id", "prop_kwd", "from_type_kwd", "to_type_kwd"}, filter, nil, offset, ontologyCountPageSize, nil)
		if searchErr != nil {
			return nil, false, 0, searchErr
		}
		if len(page) == 0 {
			return counts, false, unattributed, nil
		}
		for _, row := range page {
			scanned++
			prop := strings.TrimSpace(firstStringValue(row["prop_kwd"]))
			source := strings.TrimSpace(firstStringValue(row["from_type_kwd"]))
			target := strings.TrimSpace(firstStringValue(row["to_type_kwd"]))
			if prop == "" || source == "" || target == "" {
				unattributed++
				continue
			}
			counts[ontologyEdgeKey(prop, source, target)]++
		}
		if scanned >= ontologyCountScanCap {
			return counts, true, unattributed, nil
		}
		if len(page) < ontologyCountPageSize {
			return counts, false, unattributed, nil
		}
	}
}

// mergeOntologyClasses returns the declared classes in declaration order
// followed by any undeclared class the data carries, so vocabulary drift shows
// up instead of disappearing.
func mergeOntologyClasses(declared declaredOntology, counts map[string]int, attrs attributeCounts) []OntologyClassNode {
	out := make([]OntologyClassNode, 0, len(declared.classOrder)+len(counts))
	seen := map[string]bool{}
	for _, name := range declared.classOrder {
		seen[name] = true
		class := declared.classes[name]
		out = append(out, OntologyClassNode{
			Type: name,
			// Label falls back to Type on the client, matching the viewer's
			// `cls.label || cls.name`.
			Label:       class.label,
			Description: class.description,
			Parent:      class.parents,
			Attributes:  declared.effectiveAttributes(name, attrs),
			Entities:    counts[name],
			Declared:    true,
		})
	}
	extra := make([]string, 0, len(counts))
	for name := range counts {
		if !seen[name] {
			extra = append(extra, name)
		}
	}
	sort.Strings(extra)
	for _, name := range extra {
		out = append(out, OntologyClassNode{Type: name, Entities: counts[name]})
	}
	return out
}

// mergeOntologyProperties expands each declared property into one edge per
// domain × range combination, then appends any observed-but-undeclared property
// edge. A declared property whose endpoints were never stamped is not dropped,
// so the declared skeleton stays complete.
func mergeOntologyProperties(declared declaredOntology, counts map[string]int) []OntologyPropertyEdge {
	out := make([]OntologyPropertyEdge, 0, len(declared.propOrder)+len(counts))
	emitted := map[string]bool{}
	for _, name := range declared.propOrder {
		prop := declared.props[name]
		if !prop.isObject() {
			// A datatype property (or one whose endpoints the template never
			// declared) is not an edge: it has no class as its range. Datatype
			// properties reach the view through Classes[].Attributes.
			continue
		}
		for _, source := range prop.domains {
			for _, target := range prop.ranges {
				key := ontologyEdgeKey(name, source, target)
				emitted[key] = true
				out = append(out, OntologyPropertyEdge{
					Type:        name,
					Label:       prop.label,
					Description: prop.description,
					Source:      source,
					Target:      target,
					Relations:   counts[key],
					Declared:    true,
				})
			}
		}
	}
	extra := make([]string, 0, len(counts))
	for key := range counts {
		if !emitted[key] {
			extra = append(extra, key)
		}
	}
	sort.Strings(extra)
	for _, key := range extra {
		parts := strings.SplitN(key, "\x00", 3)
		if len(parts) != 3 {
			continue
		}
		out = append(out, OntologyPropertyEdge{
			Type:      parts[0],
			Source:    parts[1],
			Target:    parts[2],
			Relations: counts[key],
		})
	}
	return out
}

// withKnowledgeGraphKWD clones a filter and pins the row-kind discriminator.
func withKnowledgeGraphKWD(filter map[string]interface{}, kinds []string) map[string]interface{} {
	out := make(map[string]interface{}, len(filter)+1)
	for k, v := range filter {
		out[k] = v
	}
	out["knowledge_graph_kwd"] = kinds
	return out
}

// documentBucketFilter scopes an ontology count scan to one document and one
// template.
func documentBucketFilter(documentID, templateID string) map[string]interface{} {
	filter := map[string]interface{}{"doc_id": []string{documentID}}
	if templateID != "" {
		filter["compilation_template_ids"] = []string{templateID}
	}
	return filter
}

// datasetBucketFilter scopes an ontology count scan to one dataset-scope
// template bucket.
func datasetBucketFilter(resolvedKind, templateID string) map[string]interface{} {
	filter := map[string]interface{}{"scope_kwd": []string{"dataset"}}
	if resolvedKind != "" {
		filter["compilation_template_kind_kwd"] = []string{resolvedKind}
	}
	if templateID != "" {
		filter["compilation_template_ids"] = []string{templateID}
	}
	return filter
}

// jsonMapAt reads a nested object. Callers normalise a template's JSONMap into
// a plain map before handing it over, so the plain shape is the only one here.
func jsonMapAt(container map[string]interface{}, key string) map[string]interface{} {
	if container == nil {
		return nil
	}
	if v, ok := container[key].(map[string]interface{}); ok {
		return v
	}
	return nil
}

// jsonListAt reads a nested array of field declarations.
func jsonListAt(container map[string]interface{}, key string) []interface{} {
	if container == nil {
		return nil
	}
	switch v := container[key].(type) {
	case []interface{}:
		return v
	}
	return nil
}

// jsonStringAt reads a string-valued key.
func jsonStringAt(container map[string]interface{}, key string) string {
	if container == nil {
		return ""
	}
	if s, ok := container[key].(string); ok {
		return s
	}
	return ""
}

// ---- Drilling into the graph: a class's instances, a property's assertions ----
//
// The graph above is a skeleton plus counts. These two pages are what turn one
// node or one edge into rows — the drill-down the ontology view needs
// (ontology.md §10.8.4). Both are engine-paged: a viewer renders a list, not an
// unbounded dump, and the engines expose no aggregation. Both return the DTOs the
// structure graph already returns, so the frontend reuses one renderer.
//
// Deliberately no text-match parameter: the view's search box searches CLASS and
// PROPERTY names, which live in the template config and are matched locally with
// zero queries (ontology.md §10.8.4). Searching instances is a separate, optional
// capability that would need its own decision about ranking.

const (
	// ontologyDrilldownDefaultLimit is the page size when the caller omits one.
	ontologyDrilldownDefaultLimit = 30
	// ontologyDrilldownMaxLimit caps one page. The engines accept more, but this
	// is a list in a panel.
	ontologyDrilldownMaxLimit = 200
)

// OntologyClassEntitiesInput asks for one page of a class's instances.
type OntologyClassEntitiesInput struct {
	TenantID  string
	DatasetID string
	// Class is the class name as the template declares it (a value, never a
	// column name).
	Class string
	// TemplateID narrows to one compilation template's rows, DocumentID to one
	// document. Both empty means the whole dataset scope.
	TemplateID string
	DocumentID string
	Offset     int
	Limit      int
}

// OntologyEntityPage is one page of a class's instances.
type OntologyEntityPage struct {
	Class  string `json:"class"`
	Total  int64  `json:"total"`
	Offset int    `json:"offset"`
	Limit  int    `json:"limit"`
	// Entities is never nil, so the frontend can map over it without a guard.
	Entities []StructureGraphNode `json:"entities"`
}

// OntologyPropertyRelationsInput asks for one page of a property's assertions.
type OntologyPropertyRelationsInput struct {
	TenantID  string
	DatasetID string
	// Property is the declared property name.
	Property string
	// SourceType / TargetType pin one domain → range combination of a
	// polymorphic property (the graph draws one edge per combination, and the
	// drill-down must return that same edge's rows). Empty means every
	// combination.
	SourceType string
	TargetType string
	TemplateID string
	DocumentID string
	Offset     int
	Limit      int
}

// OntologyRelationPage is one page of a property's assertions.
type OntologyRelationPage struct {
	Property string `json:"property"`
	Source   string `json:"source,omitempty"`
	Target   string `json:"target,omitempty"`
	Total    int64  `json:"total"`
	Offset   int    `json:"offset"`
	Limit    int    `json:"limit"`
	// Relations is never nil, for the same reason as Entities.
	Relations []StructureGraphRelation `json:"relations"`
}

// ontologyDrilldownPage normalizes the requested window into a legal one.
func ontologyDrilldownPage(offset, limit int) (int, int) {
	if offset < 0 {
		offset = 0
	}
	if limit <= 0 {
		limit = ontologyDrilldownDefaultLimit
	}
	if limit > ontologyDrilldownMaxLimit {
		limit = ontologyDrilldownMaxLimit
	}
	return limit, offset
}

// ontologyInstanceFilter is the shared scope of every ontology drill-down: the
// row kind plus whichever scope the caller narrowed to. The template predicate
// uses compilation_template_ids (what the writer stamps) rather than the
// template's kind, so two ontology templates in one dataset stay separable.
func ontologyInstanceFilter(kind, templateID, documentID string) map[string]interface{} {
	filter := map[string]interface{}{"knowledge_graph_kwd": []string{kind}}
	if templateID != "" {
		filter["compilation_template_ids"] = []string{templateID}
	}
	if documentID != "" {
		filter["doc_id"] = []string{documentID}
	}
	return filter
}

// GetOntologyClassEntities pages the instances of one class (ontology.md §6.5 B1).
//
// entity_type_kwd is matched by term, not by text: class names are snake_case
// single tokens, which is exactly why they are usable as filter values while
// entity names (which contain spaces) are not (ontology.md §4.5).
func (s *DatasetArtifactService) GetOntologyClassEntities(ctx context.Context, in OntologyClassEntitiesInput) (*OntologyEntityPage, error) {
	class := strings.TrimSpace(in.Class)
	page := &OntologyEntityPage{
		Class:    class,
		Entities: []StructureGraphNode{},
	}
	if class == "" {
		return page, nil
	}

	filter := ontologyInstanceFilter("entity", in.TemplateID, in.DocumentID)
	filter["entity_type_kwd"] = []string{class}

	limit, offset := ontologyDrilldownPage(in.Offset, in.Limit)
	page.Offset, page.Limit = offset, limit

	// Ranked by mentions so the page shows the class's most substantial
	// instances first; the graph's own large-bucket sampling uses the same key.
	rows, total, err := graphRowSearch(ctx, in.TenantID, in.DatasetID, graphEntityFields, filter,
		(&types.OrderByExpr{}).Desc("mention_count_int"), offset, limit, nil)
	if err != nil {
		return nil, err
	}
	page.Total = total
	for _, row := range rows {
		if node := projectEntity(row); node != nil {
			page.Entities = append(page.Entities, node)
		}
	}
	page.Entities = dedupEntities(page.Entities)
	return page, nil
}

// GetOntologyPropertyRelations pages the assertions of one property edge
// (ontology.md §6.5 B3/B4).
func (s *DatasetArtifactService) GetOntologyPropertyRelations(ctx context.Context, in OntologyPropertyRelationsInput) (*OntologyRelationPage, error) {
	prop := strings.TrimSpace(in.Property)
	page := &OntologyRelationPage{
		Property:  prop,
		Source:    strings.TrimSpace(in.SourceType),
		Target:    strings.TrimSpace(in.TargetType),
		Relations: []StructureGraphRelation{},
	}
	if prop == "" {
		return page, nil
	}

	filter := ontologyInstanceFilter("relation", in.TemplateID, in.DocumentID)
	filter["prop_kwd"] = []string{prop}
	if page.Source != "" {
		filter["from_type_kwd"] = []string{page.Source}
	}
	if page.Target != "" {
		filter["to_type_kwd"] = []string{page.Target}
	}

	limit, offset := ontologyDrilldownPage(in.Offset, in.Limit)
	page.Offset, page.Limit = offset, limit

	rows, total, err := graphRowSearch(ctx, in.TenantID, in.DatasetID, graphRelationFields, filter, nil, offset, limit, nil)
	if err != nil {
		return nil, err
	}
	page.Total = total
	for _, row := range rows {
		if edge := projectRelation(row); edge != nil {
			page.Relations = append(page.Relations, edge)
		}
	}
	return page, nil
}
