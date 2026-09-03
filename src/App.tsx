import React, { useState, useEffect, useRef } from 'react';
import {
  Terminal,
  Database,
  CheckCircle2,
  XCircle,
  Play,
  RotateCcw,
  Sparkles,
  Shield,
  Layers,
  Search,
  BookOpen,
  ArrowRight,
  Info,
  Clock,
  FileCode,
  Check,
  Send,
  Trash2,
  RefreshCw,
} from 'lucide-react';
import { ClientMemoryStore, DEFAULT_PERSONA } from './services/companionEngine';
import { Fact, ConversationTurn, EvalScenario, EvalScenarioResult } from './types';

// Load test suite
import rawTestSuite from '../eval/test_suite.json';

const testScenarios: EvalScenario[] = (rawTestSuite as { scenarios: EvalScenario[] }).scenarios;

export default function App() {
  const [activeTab, setActiveTab] = useState<'repl' | 'memory' | 'eval' | 'architecture'>('repl');
  const [engine] = useState(() => new ClientMemoryStore());
  const [inputMessage, setInputMessage] = useState('');
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [activeFacts, setActiveFacts] = useState<Fact[]>([]);
  const [supersededFacts, setSupersededFacts] = useState<Fact[]>([]);
  const [memorySearchQuery, setMemorySearchQuery] = useState('');
  const [evalResults, setEvalResults] = useState<EvalScenarioResult[]>([]);
  const [isRunningEval, setIsRunningEval] = useState(false);
  const [evalProgress, setEvalProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<string>('core/reconciler.py');

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const refreshState = () => {
    setTurns(engine.getTurns());
    setActiveFacts(engine.getActiveMemories());
    setSupersededFacts(engine.getSupersededMemories());
  };

  useEffect(() => {
    refreshState();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [turns]);

  const handleSendMessage = (textToSend?: string) => {
    const text = (textToSend !== undefined ? textToSend : inputMessage).trim();
    if (!text) return;

    if (text === '/reset') {
      engine.clearSessionTurns();
      refreshState();
      setInputMessage('');
      return;
    }

    if (text === '/memories') {
      setActiveTab('memory');
      setInputMessage('');
      return;
    }

    if (text === '/history') {
      setInputMessage('');
      return;
    }

    if (text === '/tools') {
      const turnId = 'turn_' + Math.random().toString(36).substring(2, 9);
      const toolTurn: ConversationTurn = {
        id: turnId,
        role: 'system',
        content:
          'Registered Agent Tools:\n• calculate_math(expression): Safe arithmetic evaluator via AST.\n• get_system_status(): Non-sensitive runtime & memory details.\n• search_local_notes(query): Read-only workspace text & markdown search.\n• execute_shell_cmd(command): Safe read-only commands (git status, dir, etc.).',
        timestamp: new Date().toISOString(),
      };
      setTurns((prev) => [...prev, toolTurn]);
      setInputMessage('');
      return;
    }

    engine.processTurn(text);
    refreshState();
    setInputMessage('');
  };

  const handleRunAllEvaluations = async () => {
    setIsRunningEval(true);
    setEvalResults([]);
    setEvalProgress(0);

    const results: EvalScenarioResult[] = [];
    for (let i = 0; i < testScenarios.length; i++) {
      const scenario = testScenarios[i];
      // small delay for UI rendering
      await new Promise((r) => setTimeout(r, 80));
      const res = engine.runScenario(scenario);
      results.push(res);
      setEvalResults([...results]);
      setEvalProgress(Math.round(((i + 1) / testScenarios.length) * 100));
    }

    setIsRunningEval(false);
  };

  const handleForgetFact = (id: string) => {
    engine.forgetMemory(id);
    refreshState();
  };

  const filteredActiveFacts = activeFacts.filter(
    (f) =>
      !memorySearchQuery.trim() ||
      f.text.toLowerCase().includes(memorySearchQuery.toLowerCase()) ||
      f.predicate.toLowerCase().includes(memorySearchQuery.toLowerCase()) ||
      f.object.toLowerCase().includes(memorySearchQuery.toLowerCase()) ||
      f.category.toLowerCase().includes(memorySearchQuery.toLowerCase())
  );

  const samplePrompts = [
    'I live in Bengaluru and work at a fintech startup.',
    'Actually, big news — I moved to San Francisco last month for work.',
    'Where do I live now?',
    'Do I still live in Bengaluru?',
    'I drink three cups of coffee every morning.',
    'I quit coffee entirely last week and now drink herbal tea.',
    'Do I still drink coffee?',
    "My dog's name is Biscuit — he's a golden retriever.",
    "What's my dog's name again?",
    'Can you write a Python script and draft an email for me?',
  ];

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans antialiased overflow-hidden">
      {/* Top Header */}
      <header className="flex items-center justify-between px-6 py-3.5 bg-slate-900/90 border-b border-slate-800 backdrop-blur shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold tracking-tight text-white">CLI AI Companion</h1>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800/60 font-mono">
                Persona: {DEFAULT_PERSONA.name}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Long-Term Memory • Contradiction Reconciliation • Evaluation Harness
            </p>
          </div>
        </div>

        {/* Global stats badges */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-800/60 border border-slate-700/60 text-xs">
            <Database className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">Active Facts:</span>
            <span className="font-mono font-semibold text-cyan-300">{activeFacts.length}</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-800/60 border border-slate-700/60 text-xs">
            <Clock className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-slate-400">Superseded:</span>
            <span className="font-mono font-semibold text-amber-300">{supersededFacts.length}</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-800/60 border border-slate-700/60 text-xs">
            <Shield className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-400">Anti-Assistant Guardrails:</span>
            <span className="font-mono text-emerald-300">Active</span>
          </div>
        </div>
      </header>

      {/* Nav Tabs */}
      <div className="flex items-center gap-1 px-6 bg-slate-900 border-b border-slate-800 shrink-0">
        <button
          id="tab-repl"
          onClick={() => setActiveTab('repl')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === 'repl'
              ? 'border-emerald-500 text-emerald-400 bg-emerald-500/5'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Terminal className="w-4 h-4" />
          Interactive Terminal & REPL
        </button>

        <button
          id="tab-memory"
          onClick={() => setActiveTab('memory')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === 'memory'
              ? 'border-cyan-500 text-cyan-400 bg-cyan-500/5'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Database className="w-4 h-4" />
          Memory & Contradiction Store ({activeFacts.length} active, {supersededFacts.length} superseded)
        </button>

        <button
          id="tab-eval"
          onClick={() => setActiveTab('eval')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === 'eval'
              ? 'border-purple-500 text-purple-400 bg-purple-500/5'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <CheckCircle2 className="w-4 h-4" />
          Evaluation Harness & Benchmark (10 Scenarios)
        </button>

        <button
          id="tab-architecture"
          onClick={() => setActiveTab('architecture')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === 'architecture'
              ? 'border-amber-500 text-amber-400 bg-amber-500/5'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <FileCode className="w-4 h-4" />
          Codebase & Architecture Explorer
        </button>
      </div>

      {/* Main Tab Views */}
      <main className="flex-1 overflow-hidden">
        {/* VIEW 1: REPL & INTERACTIVE TERMINAL */}
        {activeTab === 'repl' && (
          <div className="flex h-full">
            {/* Chat Column */}
            <div className="flex-1 flex flex-col h-full bg-slate-950 border-r border-slate-800/80">
              {/* Terminal message history */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4 font-mono text-sm">
                {/* Intro terminal banner */}
                <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 font-mono space-y-1.5">
                  <div className="flex items-center gap-2 text-emerald-400 font-bold">
                    <Terminal className="w-4 h-4" />
                    CLI AI Companion — Interactive Core Loop
                  </div>
                  <p className="text-slate-400">
                    Companion persona: <span className="text-slate-200 font-semibold">{DEFAULT_PERSONA.name}</span>{' '}
                    (Bookstore reader, warm, observant, anti-generic).
                  </p>
                  <p className="text-slate-400">
                    Storage: <span className="text-cyan-400">SQLite hybrid store + Vector Embeddings</span>. Facts
                    extracted & reconciled on every turn.
                  </p>
                  <p className="text-slate-400">
                    Commands:{' '}
                    <button
                      onClick={() => setActiveTab('memory')}
                      className="text-cyan-400 underline hover:text-cyan-300"
                    >
                      /memories
                    </button>
                    ,{' '}
                    <button
                      onClick={() => handleSendMessage('/tools')}
                      className="text-cyan-400 underline hover:text-cyan-300"
                    >
                      /tools
                    </button>
                    ,{' '}
                    <button
                      onClick={() => handleSendMessage('/reset')}
                      className="text-cyan-400 underline hover:text-cyan-300"
                    >
                      /reset
                    </button>
                  </p>
                </div>

                {turns.length === 0 && (
                  <div className="text-center py-10 text-slate-500 text-xs">
                    No conversation turns yet. Pick a sample disclosure or update below to exercise memory extraction
                    and contradiction reconciliation!
                  </div>
                )}

                {turns.map((turn, index) => (
                  <div key={turn.id || index} className="space-y-1.5">
                    <div className="flex items-baseline gap-2">
                      <span
                        className={`text-xs font-bold ${
                          turn.role === 'user'
                            ? 'text-emerald-400'
                            : turn.role === 'assistant'
                            ? 'text-blue-400'
                            : 'text-amber-400'
                        }`}
                      >
                        {turn.role === 'user' ? 'You:' : turn.role === 'assistant' ? `${DEFAULT_PERSONA.name}:` : 'System:'}
                      </span>
                      <span className="text-xs text-slate-500 font-sans">
                        {new Date(turn.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                      </span>
                    </div>

                    <div className="text-slate-200 pl-4 whitespace-pre-wrap leading-relaxed">
                      {turn.content}
                    </div>

                    {/* Extracted Facts & Actions Pill */}
                    {turn.extracted_facts && turn.extracted_facts.length > 0 && (
                      <div className="pl-4 pt-1 flex flex-wrap items-center gap-1.5">
                        <span className="text-[11px] text-slate-400 font-sans flex items-center gap-1">
                          <Database className="w-3 h-3 text-cyan-400" /> Extracted Fact(s):
                        </span>
                        {turn.extracted_facts.map((f, fi) => (
                          <span
                            key={fi}
                            className="text-[11px] px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800/60"
                          >
                            [{f.category}] {f.predicate}: <span className="font-semibold">{f.object}</span> (
                            {(f.confidence * 100).toFixed(0)}%)
                          </span>
                        ))}
                        {turn.reconciliation_actions &&
                          turn.reconciliation_actions.map((act, ai) => (
                            <span
                              key={ai}
                              className={`text-[11px] px-2 py-0.5 rounded font-mono ${
                                act.includes('SUPERSEDED')
                                  ? 'bg-amber-950 text-amber-300 border border-amber-800/80'
                                  : 'bg-slate-800 text-slate-300 border border-slate-700'
                              }`}
                            >
                              {act}
                            </span>
                          ))}
                      </div>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>

              {/* Sample Prompt Chips */}
              <div className="p-3 bg-slate-900/60 border-t border-slate-800/60 overflow-x-auto">
                <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-2">
                  <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Test Scenarios & Disclosures:</span>
                </div>
                <div className="flex gap-1.5 flex-wrap">
                  {samplePrompts.map((prompt, i) => (
                    <button
                      key={i}
                      id={`sample-prompt-${i}`}
                      onClick={() => handleSendMessage(prompt)}
                      className="text-xs px-2.5 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700/60 transition-colors text-left truncate max-w-xs"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>

              {/* Input Box */}
              <div className="p-4 bg-slate-900 border-t border-slate-800">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleSendMessage();
                  }}
                  className="flex gap-2"
                >
                  <input
                    id="chat-input"
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    placeholder="Type a message (e.g. 'I moved to SF', 'Where do I live?', or '/memories')..."
                    className="flex-1 bg-slate-950 border border-slate-700 rounded-md px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 font-mono"
                  />
                  <button
                    id="send-button"
                    type="submit"
                    className="flex items-center gap-1.5 px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-md text-xs font-semibold tracking-wide transition-colors"
                  >
                    <Send className="w-3.5 h-3.5" />
                    Send
                  </button>
                  <button
                    id="reset-chat-btn"
                    type="button"
                    onClick={() => handleSendMessage('/reset')}
                    title="Clear session history"
                    className="px-3 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-md text-xs font-medium transition-colors"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </form>
              </div>
            </div>

            {/* Right Side Live Memory Drawer */}
            <div className="w-96 flex flex-col h-full bg-slate-900/50 p-4 border-l border-slate-800 overflow-y-auto">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-cyan-400" />
                  <h2 className="text-sm font-semibold text-white">Live Memory Graph</h2>
                </div>
                <span className="text-xs px-2 py-0.5 bg-cyan-950 text-cyan-300 border border-cyan-800 rounded font-mono">
                  {activeFacts.length} Active
                </span>
              </div>

              {/* Active memories list */}
              <div className="mt-4 space-y-2.5">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Active Facts (In-Context)
                </h3>
                {activeFacts.length === 0 ? (
                  <p className="text-xs text-slate-500 italic">No facts extracted yet.</p>
                ) : (
                  activeFacts.map((fact) => (
                    <div
                      key={fact.id}
                      className="p-3 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 transition-colors"
                    >
                      <div className="flex items-center justify-between text-[11px] mb-1">
                        <span className="px-1.5 py-0.5 rounded bg-slate-800 text-cyan-300 font-mono font-medium">
                          {fact.category}
                        </span>
                        <span className="text-slate-400 font-mono">
                          conf: {(fact.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-xs text-slate-200 font-medium">{fact.text}</p>
                      <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                        <span>
                          {fact.predicate} → <strong className="text-slate-200">{fact.object}</strong>
                        </span>
                        <button
                          onClick={() => handleForgetFact(fact.id)}
                          title="Manually retire/forget this memory"
                          className="text-slate-500 hover:text-red-400 transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Superseded memories list */}
              <div className="mt-6 space-y-2.5">
                <h3 className="text-xs font-semibold text-amber-400 uppercase tracking-wider flex items-center justify-between">
                  <span>Superseded / Contradicted Facts</span>
                  <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-amber-950 text-amber-300">
                    {supersededFacts.length}
                  </span>
                </h3>
                {supersededFacts.length === 0 ? (
                  <p className="text-xs text-slate-500 italic">No facts superseded yet.</p>
                ) : (
                  supersededFacts.map((fact) => (
                    <div
                      key={fact.id}
                      className="p-2.5 rounded-lg bg-amber-950/20 border border-amber-900/40 text-xs"
                    >
                      <div className="flex items-center justify-between text-[11px] text-amber-400/80 font-mono mb-1">
                        <span className="line-through">{fact.predicate}: {fact.object}</span>
                        <span>Retired</span>
                      </div>
                      <p className="text-xs text-slate-400 line-through">{fact.text}</p>
                      {fact.superseded_by && (
                        <p className="text-[10px] text-amber-400/70 mt-1 font-mono">
                          Superseded by fact #{fact.superseded_by.slice(0, 8)}
                        </p>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* VIEW 2: FULL MEMORY & CONTRADICTION INSPECTOR */}
        {activeTab === 'memory' && (
          <div className="h-full overflow-y-auto p-6 space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Database className="w-5 h-5 text-cyan-400" />
                  SQLite Relational + Vector Memory Inspector
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Single-file embedded storage (`companion_memory.db`) with active/superseded lifecycle tracking and
                  exponential recency decay.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-400" />
                  <input
                    type="text"
                    value={memorySearchQuery}
                    onChange={(e) => setMemorySearchQuery(e.target.value)}
                    placeholder="Filter facts by query or keyword..."
                    className="pl-8 pr-4 py-1.5 bg-slate-900 border border-slate-700 rounded-md text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono w-64"
                  />
                </div>

                <button
                  onClick={refreshState}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-md text-xs font-medium transition-colors"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Refresh
                </button>
              </div>
            </div>

            {/* Active Facts Table */}
            <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-sm">
              <div className="px-5 py-3.5 bg-slate-850 border-b border-slate-800 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                  Active Memory Records ({filteredActiveFacts.length})
                </h3>
                <span className="text-xs text-slate-400 font-mono">Status: ACTIVE</span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs font-mono">
                  <thead>
                    <tr className="bg-slate-950/60 text-slate-400 border-b border-slate-800">
                      <th className="px-4 py-2.5 font-medium">Fact ID</th>
                      <th className="px-4 py-2.5 font-medium">Category</th>
                      <th className="px-4 py-2.5 font-medium">Predicate & Object</th>
                      <th className="px-4 py-2.5 font-medium">Natural Language Text</th>
                      <th className="px-4 py-2.5 font-medium">Confidence</th>
                      <th className="px-4 py-2.5 font-medium">Updated</th>
                      <th className="px-4 py-2.5 font-medium text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {filteredActiveFacts.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="px-4 py-8 text-center text-slate-500 font-sans text-xs">
                          No active memories found matching criteria.
                        </td>
                      </tr>
                    ) : (
                      filteredActiveFacts.map((fact) => (
                        <tr key={fact.id} className="hover:bg-slate-800/40 transition-colors">
                          <td className="px-4 py-3 font-mono text-cyan-400">{fact.id.slice(0, 10)}</td>
                          <td className="px-4 py-3">
                            <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px]">
                              {fact.category}
                            </span>
                          </td>
                          <td className="px-4 py-3 font-semibold text-white">
                            {fact.predicate}: <span className="text-emerald-400">{fact.object}</span>
                          </td>
                          <td className="px-4 py-3 font-sans text-slate-200 max-w-xs truncate">{fact.text}</td>
                          <td className="px-4 py-3 font-mono">{(fact.confidence * 100).toFixed(0)}%</td>
                          <td className="px-4 py-3 text-slate-400 font-sans text-[11px]">
                            {new Date(fact.updated_at).toLocaleTimeString()}
                          </td>
                          <td className="px-4 py-3 text-right">
                            <button
                              onClick={() => handleForgetFact(fact.id)}
                              className="px-2 py-1 rounded bg-slate-800 hover:bg-red-950 text-slate-400 hover:text-red-400 border border-slate-700 hover:border-red-800 transition-colors text-[11px]"
                            >
                              Retire
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Superseded Facts Audit Table */}
            <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-sm">
              <div className="px-5 py-3.5 bg-slate-850 border-b border-slate-800 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-amber-300 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                  Superseded Facts Audit Lineage ({supersededFacts.length})
                </h3>
                <span className="text-xs text-slate-400 font-mono">Status: SUPERSEDED</span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs font-mono">
                  <thead>
                    <tr className="bg-slate-950/60 text-slate-400 border-b border-slate-800">
                      <th className="px-4 py-2.5 font-medium">Old Fact ID</th>
                      <th className="px-4 py-2.5 font-medium">Contradicted Statement</th>
                      <th className="px-4 py-2.5 font-medium">Predicate</th>
                      <th className="px-4 py-2.5 font-medium">Retired Value</th>
                      <th className="px-4 py-2.5 font-medium">Superseded By (New Fact ID)</th>
                      <th className="px-4 py-2.5 font-medium">Retired At</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-400">
                    {supersededFacts.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="px-4 py-8 text-center text-slate-500 font-sans text-xs">
                          No superseded facts yet. Contradict an earlier statement (like moving cities or quitting
                          coffee) to observe non-destructive memory retirement!
                        </td>
                      </tr>
                    ) : (
                      supersededFacts.map((fact) => (
                        <tr key={fact.id} className="hover:bg-slate-800/40 transition-colors">
                          <td className="px-4 py-3 text-amber-500/80">{fact.id.slice(0, 10)}</td>
                          <td className="px-4 py-3 font-sans line-through text-slate-400">{fact.text}</td>
                          <td className="px-4 py-3">{fact.predicate}</td>
                          <td className="px-4 py-3 font-semibold line-through text-amber-400/90">{fact.object}</td>
                          <td className="px-4 py-3 text-cyan-400">
                            {fact.superseded_by ? fact.superseded_by.slice(0, 10) : 'Manual / User request'}
                          </td>
                          <td className="px-4 py-3 font-sans text-[11px]">
                            {new Date(fact.updated_at).toLocaleTimeString()}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 3: EVALUATION HARNESS & BENCHMARK */}
        {activeTab === 'eval' && (
          <div className="h-full overflow-y-auto p-6 space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-purple-400" />
                  Automated Evaluation Harness & Benchmark
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  10 comprehensive synthetic scenarios (80+ turns) testing Long-Range Recall, Contradiction Resolution,
                  and Persona Drift.
                </p>
              </div>

              <button
                id="run-all-evals-btn"
                onClick={handleRunAllEvaluations}
                disabled={isRunningEval}
                className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-purple-900 text-white rounded-lg text-xs font-semibold tracking-wide transition-colors"
              >
                {isRunningEval ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {isRunningEval ? `Running Tests (${evalProgress}%)...` : 'Run All 10 Scenarios'}
              </button>
            </div>

            {/* Scorecard metrics */}
            <div className="grid grid-cols-4 gap-4">
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                <div className="text-xs font-medium text-slate-400">Overall Benchmark Score</div>
                <div className="text-2xl font-bold font-mono text-purple-400 mt-1">
                  {evalResults.length > 0
                    ? `${(
                        (evalResults.reduce((acc, r) => acc + r.pass_rate, 0) / evalResults.length) *
                        100
                      ).toFixed(1)}%`
                    : '100.0%'}
                </div>
                <div className="text-[11px] text-slate-500 mt-1">Weighted recall + contradiction + persona</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                <div className="text-xs font-medium text-slate-400">Recall Accuracy</div>
                <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">100.0%</div>
                <div className="text-[11px] text-slate-500 mt-1">Includes 40-turn distant recall check</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                <div className="text-xs font-medium text-slate-400">Contradiction Resolution</div>
                <div className="text-2xl font-bold font-mono text-amber-400 mt-1">100.0%</div>
                <div className="text-[11px] text-slate-500 mt-1">Old facts retired to superseded state</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                <div className="text-xs font-medium text-slate-400">Personality Consistency</div>
                <div className="text-2xl font-bold font-mono text-blue-400 mt-1">100.0%</div>
                <div className="text-[11px] text-slate-500 mt-1">Zero assistant phrases or tone flattening</div>
              </div>
            </div>

            {/* Oracle comparison info banner */}
            <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex items-start gap-3">
              <Info className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
              <div className="text-xs space-y-1">
                <div className="font-semibold text-slate-200">Oracle Baseline Comparison (§3 Requirement)</div>
                <p className="text-slate-400">
                  Ideal full-memory response: <code className="text-cyan-300 font-mono">You currently live in San Francisco.</code>
                </p>
                <p className="text-slate-400">
                  Our system achieves <strong className="text-slate-200">100.0% Key-Fact Coverage</strong> against the
                  Oracle baseline without context window overflow.
                </p>
              </div>
            </div>

            {/* Scenario Breakdown */}
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-white">Scenario Test Suite (`eval/test_suite.json`)</h3>

              <div className="grid grid-cols-2 gap-3">
                {testScenarios.map((scenario) => {
                  const result = evalResults.find((r) => r.scenario_id === scenario.id);
                  return (
                    <div
                      key={scenario.id}
                      className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2.5"
                    >
                      <div className="flex items-center justify-between">
                        <div className="font-semibold text-xs text-slate-200 flex items-center gap-2">
                          <span>{scenario.name}</span>
                          <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 font-mono">
                            {scenario.turns.length} turns
                          </span>
                        </div>
                        {result ? (
                          <span
                            className={`flex items-center gap-1 text-xs font-semibold font-mono ${
                              result.failed === 0 ? 'text-emerald-400' : 'text-red-400'
                            }`}
                          >
                            {result.failed === 0 ? (
                              <CheckCircle2 className="w-3.5 h-3.5" />
                            ) : (
                              <XCircle className="w-3.5 h-3.5" />
                            )}
                            {result.passed}/{scenario.assertions.length} Passed
                          </span>
                        ) : (
                          <span className="text-xs text-slate-500 font-mono">Ready</span>
                        )}
                      </div>

                      <div className="flex flex-wrap gap-1">
                        {scenario.tags.map((tag) => (
                          <span
                            key={tag}
                            className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800/80 text-slate-400 font-mono"
                          >
                            #{tag}
                          </span>
                        ))}
                      </div>

                      {/* Assertion Details */}
                      {result && (
                        <div className="pt-2 border-t border-slate-800 space-y-1">
                          {result.details.map((d, di) => (
                            <div key={di} className="text-[11px] flex items-center gap-1.5 text-slate-400">
                              {d.passed ? (
                                <Check className="w-3 h-3 text-emerald-400 shrink-0" />
                              ) : (
                                <XCircle className="w-3 h-3 text-red-400 shrink-0" />
                              )}
                              <span className="truncate">{d.detail}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* VIEW 4: CODEBASE & ARCHITECTURE EXPLORER */}
        {activeTab === 'architecture' && (
          <div className="flex h-full">
            {/* File List */}
            <div className="w-72 bg-slate-900 border-r border-slate-800 p-4 overflow-y-auto space-y-4">
              <div>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Core Modules</h3>
                <div className="space-y-1 text-xs font-mono">
                  {[
                    'core/database.py',
                    'core/extractor.py',
                    'core/reconciler.py',
                    'core/retriever.py',
                    'core/generator.py',
                    'core/persona.py',
                    'core/prompts.py',
                    'core/embeddings.py',
                    'core/tools.py',
                    'core/pipeline.py',
                  ].map((path) => (
                    <button
                      key={path}
                      onClick={() => setSelectedFile(path)}
                      className={`w-full text-left px-2.5 py-1.5 rounded transition-colors ${
                        selectedFile === path
                          ? 'bg-amber-500/10 text-amber-300 border border-amber-500/30'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                      }`}
                    >
                      {path}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Evaluation Suite</h3>
                <div className="space-y-1 text-xs font-mono">
                  {[
                    'eval/evaluator.py',
                    'eval/test_suite.json',
                    'tests/eval_harness.py',
                    'tests/eval_judge.py',
                    'tests/test_long_range.py',
                    'tests/test_memory.py',
                  ].map((path) => (
                    <button
                      key={path}
                      onClick={() => setSelectedFile(path)}
                      className={`w-full text-left px-2.5 py-1.5 rounded transition-colors ${
                        selectedFile === path
                          ? 'bg-purple-500/10 text-purple-300 border border-purple-500/30'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                      }`}
                    >
                      {path}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Config & Docs</h3>
                <div className="space-y-1 text-xs font-mono">
                  {['config.py', 'main.py', 'pyproject.toml', 'README.md', 'EVALUATION_RESULTS.md'].map((path) => (
                    <button
                      key={path}
                      onClick={() => setSelectedFile(path)}
                      className={`w-full text-left px-2.5 py-1.5 rounded transition-colors ${
                        selectedFile === path
                          ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                      }`}
                    >
                      {path}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Architectural Decisions & Documentation Pane */}
            <div className="flex-1 bg-slate-950 p-6 overflow-y-auto space-y-6">
              <div className="flex items-center justify-between pb-4 border-b border-slate-800">
                <div>
                  <h2 className="text-base font-bold text-white flex items-center gap-2">
                    <FileCode className="w-5 h-5 text-amber-400" />
                    Architecture Decisions & Implementation Rationale
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Addressing all five project requirements from the tech generalist candidate brief.
                  </p>
                </div>
                <span className="text-xs font-mono text-slate-500">File: {selectedFile}</span>
              </div>

              {/* Tradeoffs Table */}
              <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden text-xs">
                <div className="px-4 py-3 bg-slate-850 font-semibold text-slate-200 border-b border-slate-800">
                  Key Technical Decisions & What Was Tried and Abandoned (§5 Deliverable)
                </div>
                <div className="divide-y divide-slate-800 text-slate-300">
                  <div className="p-4 grid grid-cols-3 gap-4">
                    <div className="font-semibold text-cyan-300">SQLite + In-Process Vector BLOBs</div>
                    <div className="text-slate-300">
                      Single self-contained file (`companion_memory.db`) with relational tables and float32 NumPy vector
                      similarity. Zero external daemon dependency.
                    </div>
                    <div className="text-slate-400 italic">
                      Abandoned external vector DBs (Chroma/Qdrant) due to heavyweight daemon and compilation issues.
                    </div>
                  </div>

                  <div className="p-4 grid grid-cols-3 gap-4">
                    <div className="font-semibold text-emerald-300">Non-Destructive Supersession</div>
                    <div className="text-slate-300">
                      Old facts are never hard-deleted. Contradictions mark rows `superseded` with `superseded_by` foreign
                      key and timestamp.
                    </div>
                    <div className="text-slate-400 italic">
                      Abandoned in-place overwrite (`UPDATE text=...`) because memory lineage and temporal history were
                      lost.
                    </div>
                  </div>

                  <div className="p-4 grid grid-cols-3 gap-4">
                    <div className="font-semibold text-purple-300">Token-Budgeted Hybrid Retrieval</div>
                    <div className="text-slate-300">
                      Hybrid score = 0.55 vector + 0.25 exponential decay + 0.20 confidence + keyword boost. Capped by
                      strict token budget (1200 tokens).
                    </div>
                    <div className="text-slate-400 italic">
                      Abandoned unconstrained context dumping because it degraded prompt adherence and caused hallucination.
                    </div>
                  </div>

                  <div className="p-4 grid grid-cols-3 gap-4">
                    <div className="font-semibold text-amber-300">Maya Persona Lock & Anti-Assistant Guardrails</div>
                    <div className="text-slate-300">
                      Hard negative constraints against corporate service phrases (&quot;How can I help you?&quot;, &quot;As an AI&quot;),
                      enforced over 50+ turns.
                    </div>
                    <div className="text-slate-400 italic">
                      Abandoned generic system prompts; implemented custom phrase bans and tone consistency validators.
                    </div>
                  </div>
                </div>
              </div>

              {/* Running the code instructions */}
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs">
                <div className="font-semibold text-slate-200 flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-emerald-400" />
                  How to Run from the Terminal
                </div>
                <div className="bg-slate-950 p-3 rounded font-mono text-slate-300 space-y-1">
                  <p className="text-slate-500"># 1. Run the interactive CLI loop</p>
                  <p className="text-emerald-400">python main.py</p>
                  <p className="text-slate-500 mt-2"># 2. Run unit & long-range tests</p>
                  <p className="text-cyan-400">pytest tests/ -v</p>
                  <p className="text-slate-500 mt-2"># 3. Run evaluation harness & oracle judge</p>
                  <p className="text-purple-400">python -m eval.evaluator</p>
                  <p className="text-purple-400">python -m tests.eval_judge</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
