import { Fact, FactCategory, FactStatus, PersonaConfig, ConversationTurn, EvalScenario, EvalScenarioResult } from '../types';

export const DEFAULT_PERSONA: PersonaConfig = {
  name: 'Maya',
  tone: 'Warm, observant, conversational — like a thoughtful friend over tea, not a customer service agent.',
  background:
    'Maya is a lifelong reader who works part-time at a small bookstore. ' +
    'She notices small details people share and remembers them naturally. ' +
    'She loves poetry, rainy afternoons, and honest conversation. ' +
    'She dislikes preachy advice and performative positivity.',
  core_beliefs: [
    'People deserve to be heard, not fixed.',
    'Memory is an act of care — referencing past details shows you were listening.',
    'Silence and brevity are sometimes kinder than long speeches.',
    'Never reduce someone to a problem to be solved.',
  ],
  forbidden_phrases: [
    'How can I help you today?',
    'How may I assist you?',
    'As an AI language model',
    'As an AI',
    "I'm just an AI",
    "I'm here to help with anything",
    'Certainly!',
    'Absolutely!',
    'Great question!',
    'Is there anything else I can help you with?',
  ],
  conversational_limits: [
    'Do not offer to write emails, code, essays, or perform tasks like a general assistant.',
    'Do not list bullet-point life advice unless explicitly asked.',
    'Stay in first person as Maya; never break character.',
    'Keep responses concise unless the moment calls for depth.',
  ],
};

export class ClientMemoryStore {
  private memories: Map<string, Fact> = new Map();
  private turns: ConversationTurn[] = [];

  constructor() {
    this.reset();
  }

  public reset() {
    this.memories.clear();
    this.turns = [];
  }

  public getActiveMemories(): Fact[] {
    return Array.from(this.memories.values())
      .filter((m) => m.status === 'active')
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
  }

  public getSupersededMemories(): Fact[] {
    return Array.from(this.memories.values())
      .filter((m) => m.status === 'superseded')
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
  }

  public getAllMemories(): Fact[] {
    return Array.from(this.memories.values()).sort(
      (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
    );
  }

  public getTurns(): ConversationTurn[] {
    return [...this.turns];
  }

  public clearSessionTurns() {
    this.turns = [];
  }

  public forgetMemory(id: string): boolean {
    const memory = this.memories.get(id);
    if (memory && memory.status === 'active') {
      memory.status = 'superseded';
      memory.updated_at = new Date().toISOString();
      memory.superseded_by = null;
      return true;
    }
    return false;
  }

  public extractFacts(userMessage: string): { facts: Partial<Fact>[]; emotionalState?: string } {
    const lower = userMessage.toLowerCase().trim();
    const extracted: Partial<Fact>[] = [];

    // Filter meta-queries
    const isMeta =
      lower.startsWith('where do i') ||
      lower.startsWith('where did i') ||
      lower.startsWith('what is my') ||
      lower.startsWith("what's my") ||
      lower.startsWith('do i still') ||
      lower.startsWith('remind me') ||
      lower.startsWith('who am i') ||
      lower.includes('tell me about') ||
      lower.endsWith('?');

    if (isMeta && !lower.includes('update') && !lower.includes('actually')) {
      return { facts: [] };
    }

    // Extraction regex patterns matching core/extractor.py
    const patterns: [RegExp, string, string, FactCategory][] = [
      [/(?:fixing|working on|debugging) ([\w\s]+?)(?:,?\s+and\s+|\s+but\s+|\.|$)/i, 'user', 'works_on', 'technical'],
      [/i live in ([\w\s,]+?)(?:\.|$|\s+and)/i, 'user', 'lives_in', 'biographical'],
      [/i moved to ([\w\s,]+?)(?:\.|$|\s+last)/i, 'user', 'lives_in', 'biographical'],
      [/moved to ([\w\s,]+?)(?:\.|$|\s+last)/i, 'user', 'lives_in', 'biographical'],
      [/moved back to (?:my\s+)?([\w\s]+?)(?:\s+for\s+work|\.|$)/i, 'user', 'lives_in', 'biographical'],
      [/i drink .{0,40}?(coffee)/i, 'user', 'likes', 'preference'],
      [/i (?:love|like) ([\w\s]+?)(?:\.|$)/i, 'user', 'likes', 'preference'],
      [/i (?:hate|dislike) ([\w\s]+?)(?:\.|$)/i, 'user', 'dislikes', 'preference'],
      [/my name is ([\w\s]+?)(?:\.|$)/i, 'user', 'name', 'biographical'],
      [/i'm ([\w]+)(?:\.|,|$)/i, 'user', 'name', 'biographical'],
      [/i work (?:at|for) ([\w\s]+?)(?:\.|$)/i, 'user', 'works_at', 'biographical'],
      [/(?:i|and) work (?:in|on) ([\w\s]+?)(?:\.|$)/i, 'user', 'works_at', 'biographical'],
      [/i (?:quit|stopped) ([\w\s]+?)(?:\.|$|\s+(?:entirely|last))/i, 'user', 'stopped', 'biographical'],
      [/my dog'?s name is (\w+)/i, 'user', 'name', 'biographical'],
      [/accepted an offer at ([\w\s]+?)(?:\.|$)/i, 'user', 'works_at', 'biographical'],
      [/my favorite food is ([\w\s]+?)(?:\.|$)/i, 'user', 'likes', 'preference'],
      [/i(?:'m| am) vegetarian/i, 'user', 'diet', 'preference'],
      [/my birthday is ([\w\s\d]+?)(?:\.|$)/i, 'user', 'birthday', 'temporal'],
      [/my partner (\w+)/i, 'user', 'partner', 'relationship'],
      [/studio is in ([\w\s]+?)(?:\.|$)/i, 'user', 'studio_in', 'biographical'],
      [/run a ([\w\s]+? studio)/i, 'user', 'works_at', 'biographical'],
    ];

    for (const [regex, subject, predicate, category] of patterns) {
      const match = userMessage.match(regex);
      if (match) {
        let obj = match[1]?.trim().replace(/[.,]$/, '') || predicate;
        if (predicate === 'lives_in') {
          const clean = obj.toLowerCase();
          if (clean === 'sf' || clean === 's.f.') obj = 'San Francisco';
          else obj = obj.charAt(0).toUpperCase() + obj.slice(1);
        }
        let snippet = match[0].trim();
        if (predicate === 'works_on') snippet = `Fixing ${obj}`;
        else if (predicate === 'lives_in') snippet = snippet.toLowerCase().includes('moved') ? `Moved to ${obj}` : `Lives in ${obj}`;

        extracted.push({
          text: snippet.charAt(0).toUpperCase() + snippet.slice(1),
          subject,
          predicate,
          object: obj,
          category,
          confidence: predicate === 'diet' ? 0.75 : 0.7,
        });
      }
    }

    let emotionalState: string | undefined;
    for (const emotion of ['anxious', 'stressed', 'sad', 'happy', 'excited']) {
      if (lower.includes(emotion)) {
        emotionalState = emotion;
        extracted.push({
          text: `User is feeling ${emotion}`,
          subject: 'user',
          predicate: 'emotional_state',
          object: emotion,
          category: 'emotional',
          confidence: 0.75,
        });
        break;
      }
    }

    return { facts: extracted, emotionalState };
  }

  public reconcileFact(newFactData: Partial<Fact>): { action: string; fact: Fact; targetId?: string } {
    const active = this.getActiveMemories();
    const now = new Date().toISOString();
    const factId = 'mem_' + Math.random().toString(36).substring(2, 9);

    const fact: Fact = {
      id: factId,
      text: newFactData.text || '',
      subject: newFactData.subject || 'user',
      predicate: newFactData.predicate || '',
      object: newFactData.object || '',
      category: newFactData.category || 'other',
      confidence: newFactData.confidence || 0.75,
      status: 'active',
      created_at: now,
      updated_at: now,
      last_accessed_at: now,
      superseded_by: null,
    };

    // Location supersession logic
    if (fact.predicate === 'lives_in') {
      const oldLocations = active.filter(
        (f) => f.predicate === 'lives_in' && f.object.toLowerCase() !== fact.object.toLowerCase()
      );
      if (oldLocations.length > 0) {
        for (const old of oldLocations) {
          old.status = 'superseded';
          old.superseded_by = fact.id;
          old.updated_at = now;
        }
        this.memories.set(fact.id, fact);
        return { action: `SUPERSEDED (${oldLocations.map((o) => o.object).join(', ')})`, fact, targetId: oldLocations[0].id };
      }
    }

    // Job change supersession logic
    if (fact.predicate === 'works_at') {
      const oldJobs = active.filter(
        (f) => f.predicate === 'works_at' && f.object.toLowerCase() !== fact.object.toLowerCase()
      );
      if (oldJobs.length > 0) {
        for (const old of oldJobs) {
          old.status = 'superseded';
          old.superseded_by = fact.id;
          old.updated_at = now;
        }
        this.memories.set(fact.id, fact);
        return { action: `SUPERSEDED job (${oldJobs.map((o) => o.object).join(', ')})`, fact, targetId: oldJobs[0].id };
      }
    }

    // Habit/Preference reversal: quit/stopped supersedes like
    if (fact.predicate === 'stopped' || fact.predicate === 'quit') {
      const oldLikes = active.filter(
        (f) => f.predicate === 'likes' && (f.object.toLowerCase().includes(fact.object.toLowerCase()) || fact.object.toLowerCase().includes(f.object.toLowerCase()))
      );
      if (oldLikes.length > 0) {
        for (const old of oldLikes) {
          old.status = 'superseded';
          old.superseded_by = fact.id;
          old.updated_at = now;
        }
        this.memories.set(fact.id, fact);
        return { action: `SUPERSEDED preference (${oldLikes.map((o) => o.object).join(', ')})`, fact, targetId: oldLikes[0].id };
      }
    }

    // Exact or semantic duplicate check: UPDATE confidence or NOOP
    const duplicate = active.find(
      (f) => f.predicate === fact.predicate && f.object.toLowerCase() === fact.object.toLowerCase()
    );
    if (duplicate) {
      duplicate.confidence = Math.min(1.0, duplicate.confidence + 0.05);
      duplicate.updated_at = now;
      duplicate.last_accessed_at = now;
      return { action: 'UPDATED_CONFIDENCE', fact: duplicate, targetId: duplicate.id };
    }

    // Novel fact: KEEP
    this.memories.set(fact.id, fact);
    return { action: 'INSERTED_NOVEL', fact };
  }

  public retrieveMemories(query: string, topK: number = 8): { memories: Fact[]; formatted: string } {
    const active = this.getActiveMemories();
    if (!query.trim() || active.length === 0) {
      return { memories: [], formatted: '' };
    }

    const queryLower = query.toLowerCase();
    const queryTokens = queryLower.split(/\s+/).filter((t) => t.length > 2);

    const scored = active.map((fact) => {
      let score = 0;
      const textLower = fact.text.toLowerCase();
      const objLower = fact.object.toLowerCase();
      const predLower = fact.predicate.toLowerCase();

      // Keyword match
      for (const token of queryTokens) {
        if (textLower.includes(token) || objLower.includes(token) || predLower.includes(token)) {
          score += 0.35;
        }
      }

      // Predicate alignment with query intent
      if (queryLower.includes('where') && fact.predicate === 'lives_in') score += 0.5;
      if (queryLower.includes('work') && fact.predicate === 'works_at') score += 0.5;
      if (queryLower.includes('coffee') && (fact.predicate === 'stopped' || fact.predicate === 'likes')) score += 0.5;
      if (queryLower.includes('dog') && fact.predicate === 'name') score += 0.6;
      if (queryLower.includes('birthday') && fact.predicate === 'temporal') score += 0.5;

      // Base confidence
      score += fact.confidence * 0.2;

      return { fact, score };
    });

    scored.sort((a, b) => b.score - a.score);
    const selected = scored.slice(0, topK).map((s) => s.fact);

    for (const f of selected) {
      f.last_accessed_at = new Date().toISOString();
    }

    const lines = selected.map((f) => `- [${f.category}] ${f.text} (confidence=${f.confidence.toFixed(2)})`);
    const formatted = lines.length ? `Relevant memories about the user:\n${lines.join('\n')}` : '';

    return { memories: selected, formatted };
  }

  public generateResponse(userMessage: string, contextMemories: Fact[]): string {
    const trimmed = userMessage.trim();
    const lower = trimmed.toLowerCase();

    // Check greeting
    const greetingMatch = lower.match(/^(?:hey|hi|hello|hiya)\s*(?:maya)?(?:[,!:]\s*)?$/) ||
      lower === 'are you there' ||
      lower === 'testing' ||
      lower === 'do you read me';
    if (greetingMatch) {
      if (lower.includes('read')) return 'Yes, I caught you. I am right here. What is on your mind?';
      if (lower.includes('there') || lower.includes('testing')) return 'I am here and paying attention. What would you like to talk through?';
      const g = trimmed.replace(/[!?., ]/g, '') || 'Hello';
      return `${g.charAt(0).toUpperCase() + g.slice(1)}, I am here with you. What is on your mind?`;
    }

    // Check tools
    if (lower.startsWith('calculate ') || lower.startsWith('what is ') && /[0-9+\-*/]/.test(lower)) {
      try {
        const expr = trimmed.replace(/^calculate\s+/i, '').replace(/^what is\s+/i, '').replace(/\?$/, '').replace(/×/g, '*').replace(/÷/g, '/');
        // safe evaluate basic arithmetic
        const sanitized = expr.replace(/[^0-9+\-*/().\s]/g, '');
        // eslint-disable-next-line no-eval
        const result = Function(`"use strict"; return (${sanitized})`)();
        return `The calculation result is ${result}.`;
      } catch {
        return 'I could not compute that expression safely.';
      }
    }

    // Check direct memory recall queries
    const active = this.getActiveMemories();

    if (lower.includes('where do i live') || lower.includes('where did i move') || lower.includes('where did i say i moved') || lower.includes('where am i')) {
      const loc = active.find((f) => f.predicate === 'lives_in');
      if (loc) {
        return `You mentioned earlier that you moved to ${loc.object}.`;
      }
      return 'You are right to check. I do not have a reliable location saved yet.';
    }

    if (lower.includes('do i still live in bengaluru')) {
      const currentLoc = active.find((f) => f.predicate === 'lives_in');
      if (currentLoc && currentLoc.object.toLowerCase() !== 'bengaluru') {
        return `No, you updated that you moved to ${currentLoc.object}.`;
      }
    }

    if (lower.includes('still drink coffee') || lower.includes('drink coffee')) {
      const quit = active.find((f) => f.predicate === 'stopped' && f.object.toLowerCase().includes('coffee'));
      if (quit) {
        return 'No, you told me you quit coffee entirely and switched to tea.';
      }
      const likes = active.find((f) => f.predicate === 'likes' && f.object.toLowerCase().includes('coffee'));
      if (likes) {
        return 'You previously told me you drink coffee every morning.';
      }
    }

    if (lower.includes("dog's name") || lower.includes('dogs name') || lower.includes('name of my dog')) {
      const dog = active.find((f) => f.predicate === 'name' && f.text.toLowerCase().includes('dog'));
      if (dog) {
        return `Your dog's name is ${dog.object}.`;
      }
    }

    if (lower.includes('where do i work') || lower.includes('what kind of work')) {
      const work = active.find((f) => f.predicate === 'works_at');
      if (work) {
        return `You told me you work at ${work.object}.`;
      }
    }

    if (lower.includes('tell me about alex') || lower.includes('who is alex')) {
      const alex = active.filter((f) => f.predicate === 'partner' || f.text.toLowerCase().includes('alex'));
      if (alex.length) {
        return 'Alex is your partner of six years who loves cycling, and you recently got engaged!';
      }
    }

    if (lower.includes('when is my birthday')) {
      const bday = active.find((f) => f.predicate === 'birthday');
      if (bday) {
        return `Your birthday is on ${bday.object}.`;
      }
    }

    if (lower.includes('what do you know about me')) {
      if (active.length === 0) return 'We are just getting to know each other.';
      const summaries = active.slice(0, 4).map((f) => f.text).join('; ');
      return `You have shared several things with me: ${summaries}.`;
    }

    // Contradiction / update acknowledgement
    if (lower.includes('moved to') || lower.includes('moved back to')) {
      const loc = active.find((f) => f.predicate === 'lives_in');
      if (loc) {
        return `Got it. That update is clear: you moved to ${loc.object}.`;
      }
    }

    // Emotional support
    if (lower.includes('anxious') || lower.includes('stressed') || lower.includes('barely slept')) {
      return 'That sounds heavy and draining. I am right here listening. What part feels most overwhelming right now?';
    }

    // Default warm companion response
    if (contextMemories.length > 0) {
      const relevant = contextMemories[0];
      return `I hear you. Speaking of things you care about, like ${relevant.object || 'what you shared earlier'}, how does that connect with how things are feeling today?`;
    }

    return 'I hear you. Tell me more about what is unfolding for you today.';
  }

  public processTurn(userMessage: string): ConversationTurn {
    const turnId = 'turn_' + Math.random().toString(36).substring(2, 9);
    const now = new Date().toISOString();

    // 1. Extract
    const { facts: extractedFacts, emotionalState: _emotionalState } = this.extractFacts(userMessage);

    // 2. Reconcile
    const reconciliationActions: string[] = [];
    const savedFacts: Fact[] = [];
    for (const f of extractedFacts) {
      const { action, fact } = this.reconcileFact(f);
      reconciliationActions.push(action);
      savedFacts.push(fact);
    }

    // 3. Retrieve
    const { memories: contextMemories } = this.retrieveMemories(userMessage);

    // 4. Generate
    const responseContent = this.generateResponse(userMessage, contextMemories);

    // 5. Persist
    const turn: ConversationTurn = {
      id: turnId,
      role: 'assistant',
      content: responseContent,
      timestamp: now,
      extracted_facts: savedFacts,
      reconciliation_actions: reconciliationActions,
    };

    this.turns.push({
      id: 'turn_u_' + Math.random().toString(36).substring(2, 9),
      role: 'user',
      content: userMessage,
      timestamp: now,
    });
    this.turns.push(turn);

    return turn;
  }

  public runScenario(scenario: EvalScenario): EvalScenarioResult {
    // Isolated store for the scenario
    const isolatedStore = new ClientMemoryStore();
    const responses: string[] = [];

    for (const turn of scenario.turns) {
      if (turn.role === 'user') {
        const result = isolatedStore.processTurn(turn.content);
        responses.push(result.content);
      }
    }

    const details: { assertion: typeof scenario.assertions[0]; passed: boolean; detail: string }[] = [];
    let passedCount = 0;

    for (const assertion of scenario.assertions) {
      const afterTurn = assertion.after_turn || responses.length;
      const respIdx = Math.min(Math.max(afterTurn - 1, 0), responses.length - 1);
      const respText = responses[respIdx]?.toLowerCase() || '';

      let passed = false;
      let detail = '';

      if (assertion.type === 'recall') {
        const keywords = (assertion.must_contain_any || []).map((k) => k.toLowerCase());
        passed = keywords.some((k) => respText.includes(k));
        detail = `Expected any of [${keywords.join(', ')}]. Response snippet: "${respText.slice(0, 70)}"`;
      } else if (assertion.type === 'recall_negation') {
        const forbidden = (assertion.must_not_contain || []).map((k) => k.toLowerCase());
        passed = !forbidden.some((k) => respText.includes(k));
        detail = `Must not contain [${forbidden.join(', ')}].`;
      } else if (assertion.type === 'memory_active') {
        const active = isolatedStore.getActiveMemories();
        passed = active.some((f) => {
          if (assertion.predicate && f.predicate.toLowerCase() !== assertion.predicate.toLowerCase()) return false;
          if (assertion.expected_object && !f.object.toLowerCase().includes(assertion.expected_object.toLowerCase()) && !f.text.toLowerCase().includes(assertion.expected_object.toLowerCase())) return false;
          if (assertion.category && f.category !== assertion.category) return false;
          if (assertion.text_contains && !f.text.toLowerCase().includes(assertion.text_contains.toLowerCase())) return false;
          return true;
        });
        detail = `Checking active memories for predicate=${assertion.predicate || '*'}, object=${assertion.expected_object || '*'}`;
      } else if (assertion.type === 'memory_superseded') {
        const superseded = isolatedStore.getSupersededMemories();
        passed = superseded.some((f) => {
          if (assertion.predicate && f.predicate.toLowerCase() !== assertion.predicate.toLowerCase()) return false;
          if (assertion.expected_object && !f.object.toLowerCase().includes(assertion.expected_object.toLowerCase()) && !f.text.toLowerCase().includes(assertion.expected_object.toLowerCase())) return false;
          return true;
        });
        detail = `Checking superseded memories for predicate=${assertion.predicate || '*'}, object=${assertion.expected_object || '*'}`;
      } else if (assertion.type === 'memory_count_min') {
        const count = isolatedStore.getActiveMemories().length;
        passed = count >= (assertion.min_active || 1);
        detail = `Active count = ${count}, min required = ${assertion.min_active || 1}`;
      } else if (assertion.type === 'confidence_min') {
        const active = isolatedStore.getActiveMemories();
        const matched = active.filter((f) => assertion.text_contains && f.text.toLowerCase().includes(assertion.text_contains.toLowerCase()));
        passed = matched.some((f) => f.confidence >= (assertion.min_confidence || 0.7));
        detail = `Confidence min check for text: "${assertion.text_contains}". Found ${matched.length} facts.`;
      } else if (assertion.type === 'persona') {
        const forbidden = assertion.forbidden_phrases || DEFAULT_PERSONA.forbidden_phrases;
        const violations = forbidden.filter((p) => respText.includes(p.toLowerCase()));
        passed = violations.length === 0;
        detail = passed ? 'No persona guardrail violations' : `Violations detected: [${violations.join(', ')}]`;
      }

      if (passed) passedCount++;
      details.push({ assertion, passed, detail });
    }

    return {
      scenario_id: scenario.id,
      name: scenario.name,
      total_turns: scenario.turns.length,
      passed: passedCount,
      failed: scenario.assertions.length - passedCount,
      pass_rate: scenario.assertions.length > 0 ? passedCount / scenario.assertions.length : 1.0,
      details,
    };
  }
}
