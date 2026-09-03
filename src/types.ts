export type FactCategory =
  | 'preference'
  | 'biographical'
  | 'emotional'
  | 'relationship'
  | 'goal'
  | 'temporal'
  | 'technical'
  | 'other';

export type FactStatus = 'active' | 'superseded';

export type ReconciliationAction = 'KEEP' | 'SUPERSEDE' | 'UPDATE' | 'NOOP';

export interface Fact {
  id: string;
  text: string;
  subject: string;
  predicate: string;
  object: string;
  category: FactCategory;
  confidence: number;
  status: FactStatus;
  created_at: string;
  updated_at: string;
  last_accessed_at?: string | null;
  superseded_by?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ConversationTurn {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  extracted_facts?: Fact[];
  reconciliation_actions?: string[];
}

export interface PersonaConfig {
  name: string;
  tone: string;
  background: string;
  core_beliefs: string[];
  forbidden_phrases: string[];
  conversational_limits: string[];
}

export interface ScenarioAssertion {
  type: string;
  after_turn?: number;
  predicate?: string;
  expected_object?: string;
  must_contain_any?: string[];
  must_not_contain?: string[];
  forbidden_phrases?: string[];
  category?: string;
  text_contains?: string;
  min_confidence?: number;
  min_active?: number;
}

export interface ScenarioTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface EvalScenario {
  id: string;
  name: string;
  tags: string[];
  turns: ScenarioTurn[];
  assertions: ScenarioAssertion[];
}

export interface EvalScenarioResult {
  scenario_id: string;
  name: string;
  total_turns: number;
  passed: number;
  failed: number;
  pass_rate: number;
  details: {
    assertion: ScenarioAssertion;
    passed: boolean;
    detail: string;
  }[];
}
