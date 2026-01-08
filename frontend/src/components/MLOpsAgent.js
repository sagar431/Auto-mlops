import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

// Icons
const Icons = {
  Send: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>,
  Brain: () => <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>,
  Loader: () => <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>,
  Check: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>,
  CheckCircle: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
  Circle: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" strokeWidth={2}/></svg>,
  Play: () => <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>,
  Zap: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>,
  Database: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" /></svg>,
  Flask: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>,
  Box: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" /></svg>,
  Activity: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>,
  Refresh: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>,
  ChevronRight: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>,
  Plus: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>,
  Clock: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
  TrendingUp: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>,
  Settings: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
  Folder: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" /></svg>,
  Terminal: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>,
  Cloud: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" /></svg>,
  GitBranch: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 3v12m0 0a3 3 0 103 3 3 3 0 00-3-3zm12-3a3 3 0 11-6 0 3 3 0 016 0zm-3 0v-6a3 3 0 00-3-3H9" /></svg>,
  BarChart: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>,
  Sparkles: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" /></svg>,
};

// Mock data
const mockMetricsHistory = [
  { epoch: 1, accuracy: 0.45, loss: 1.2, val_accuracy: 0.42 },
  { epoch: 2, accuracy: 0.58, loss: 0.9, val_accuracy: 0.55 },
  { epoch: 3, accuracy: 0.65, loss: 0.7, val_accuracy: 0.62 },
  { epoch: 4, accuracy: 0.72, loss: 0.5, val_accuracy: 0.68 },
  { epoch: 5, accuracy: 0.78, loss: 0.4, val_accuracy: 0.74 },
  { epoch: 6, accuracy: 0.82, loss: 0.3, val_accuracy: 0.79 },
  { epoch: 7, accuracy: 0.85, loss: 0.25, val_accuracy: 0.82 },
  { epoch: 8, accuracy: 0.87, loss: 0.2, val_accuracy: 0.84 },
];

const PIPELINE_STAGES = [
  { id: 'setup', name: 'Setup', icon: Icons.Settings },
  { id: 'data', name: 'Data', icon: Icons.Database },
  { id: 'config', name: 'Config', icon: Icons.Flask },
  { id: 'training', name: 'Training', icon: Icons.Activity },
  { id: 'eval', name: 'Eval', icon: Icons.BarChart },
  { id: 'deploy', name: 'Deploy', icon: Icons.Cloud },
];

const QUICK_ACTIONS = [
  {
    icon: Icons.Zap,
    title: 'Quick Deploy',
    desc: 'Deploy a model with default settings',
    gradient: 'from-orange-500 to-amber-500',
    query: 'Deploy my model with production-ready defaults'
  },
  {
    icon: Icons.Database,
    title: 'Data Pipeline',
    desc: 'Set up DVC for data versioning',
    gradient: 'from-blue-500 to-cyan-500',
    query: 'Set up DVC with S3 remote for my dataset'
  },
  {
    icon: Icons.Activity,
    title: 'Train & Track',
    desc: 'Train with MLflow tracking',
    gradient: 'from-purple-500 to-pink-500',
    query: 'Train ResNet model and track with MLflow'
  },
  {
    icon: Icons.GitBranch,
    title: 'CI/CD Setup',
    desc: 'GitHub Actions workflow',
    gradient: 'from-green-500 to-emerald-500',
    query: 'Create CI/CD pipeline with GitHub Actions'
  },
];

const RECENT_RUNS = [
  { id: 1, name: 'ResNet-50 Training', status: 'completed', accuracy: 0.87, time: '2h ago', icon: '🎯' },
  { id: 2, name: 'Data Augmentation', status: 'completed', accuracy: 0.85, time: '5h ago', icon: '📊' },
  { id: 3, name: 'Hyperparameter Tune', status: 'running', accuracy: 0.82, time: 'Running', icon: '⚡' },
  { id: 4, name: 'Baseline Model', status: 'completed', accuracy: 0.72, time: '1d ago', icon: '🔬' },
];

export default function MLOpsAgent() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentPhase, setCurrentPhase] = useState('idle');
  const [pipelineStage, setPipelineStage] = useState(null);
  const [completedStages, setCompletedStages] = useState([]);
  const [showWelcome, setShowWelcome] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [executionSteps, setExecutionSteps] = useState([]);
  const [metricsData, setMetricsData] = useState([]);
  const [currentAccuracy, setCurrentAccuracy] = useState(0);
  const [targetAccuracy] = useState(0.85);
  const [agentThinking, setAgentThinking] = useState('');
  const [selectedRun, setSelectedRun] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (isProcessing && pipelineStage === 'training') {
      const interval = setInterval(() => {
        setMetricsData(prev => {
          if (prev.length >= mockMetricsHistory.length) {
            clearInterval(interval);
            return prev;
          }
          const newData = [...prev, mockMetricsHistory[prev.length]];
          setCurrentAccuracy(mockMetricsHistory[prev.length].accuracy);
          return newData;
        });
      }, 1200);
      return () => clearInterval(interval);
    }
  }, [isProcessing, pipelineStage]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  const simulateExecution = async (query) => {
    setShowWelcome(false);
    setIsProcessing(true);
    setMetricsData([]);
    setCurrentAccuracy(0);
    setExecutionSteps([]);
    setCompletedStages([]);

    setMessages(prev => [...prev, { id: Date.now(), role: 'user', content: query }]);

    // Perception
    setCurrentPhase('perception');
    setAgentThinking('Analyzing your request with AI...');
    await sleep(1500);
    setMessages(prev => [...prev, {
      id: Date.now(),
      role: 'assistant',
      content: '🔍 **Understanding Request**\n\nI\'ve analyzed your query and identified:\n- **Task Type:** ML Pipeline Deployment\n- **Model:** ResNet/CNN Architecture\n- **Target:** 85% accuracy threshold\n- **Tools:** Hydra, MLflow, DVC, Docker',
      phase: 'perception'
    }]);

    // Decision
    setCurrentPhase('decision');
    setAgentThinking('Creating optimized execution plan...');
    await sleep(1500);

    const steps = [
      { id: '0', description: 'Initialize Hydra configuration', tool: 'Hydra', status: 'pending' },
      { id: '1', description: 'Set up MLflow experiment tracking', tool: 'MLflow', status: 'pending' },
      { id: '2', description: 'Configure DVC data pipeline', tool: 'DVC', status: 'pending' },
      { id: '3', description: 'Build training Docker image', tool: 'Docker', status: 'pending' },
      { id: '4', description: 'Execute model training', tool: 'PyTorch', status: 'pending' },
      { id: '5', description: 'Run evaluation & metrics', tool: 'MLflow', status: 'pending' },
      { id: '6', description: 'Deploy CI/CD pipeline', tool: 'GitHub', status: 'pending' },
    ];
    setExecutionSteps(steps);

    setMessages(prev => [...prev, {
      id: Date.now(),
      role: 'assistant',
      content: `📋 **Execution Plan Ready**\n\nI'll execute ${steps.length} steps to build your complete MLOps pipeline. Each step is optimized for production use.`,
      phase: 'decision'
    }]);

    // Execute
    setCurrentPhase('execution');
    const stageMap = { 0: 'config', 1: 'setup', 2: 'data', 3: 'setup', 4: 'training', 5: 'eval', 6: 'deploy' };

    for (let i = 0; i < steps.length; i++) {
      setAgentThinking(`Executing: ${steps[i].description}`);
      setExecutionSteps(prev => prev.map((s, idx) => idx === i ? { ...s, status: 'running' } : s));

      if (stageMap[i]) {
        setPipelineStage(stageMap[i]);
      }

      await sleep(i === 4 ? 10000 : 2000);

      setExecutionSteps(prev => prev.map((s, idx) => idx === i ? { ...s, status: 'completed' } : s));
      if (stageMap[i] && !completedStages.includes(stageMap[i])) {
        setCompletedStages(prev => [...prev, stageMap[i]]);
      }
    }

    // Summary
    setCurrentPhase('summary');
    const finalAccuracy = 0.87;
    setCurrentAccuracy(finalAccuracy);
    setCompletedStages(['setup', 'data', 'config', 'training', 'eval', 'deploy']);

    setMessages(prev => [...prev, {
      id: Date.now(),
      role: 'assistant',
      content: `## ✅ Pipeline Deployed Successfully!\n\n**Performance:** ${(finalAccuracy * 100).toFixed(1)}% accuracy (Target: 85%) ✓\n\n**Artifacts Created:**\n\`\`\`\n📁 configs/config.yaml      - Hydra configuration\n📊 mlruns/                   - MLflow experiment data\n📦 .dvc/                     - Data versioning setup\n🐳 Dockerfile               - Training container\n⚡ .github/workflows/       - CI/CD pipeline\n\`\`\`\n\n**Next Steps:**\n1. Push to GitHub to trigger CI/CD\n2. Monitor training in MLflow UI\n3. Scale with Kubernetes (optional)`,
      phase: 'summary'
    }]);

    setIsProcessing(false);
    setCurrentPhase('idle');
    setAgentThinking('');
  };

  const handleSubmit = () => {
    if (!input.trim() || isProcessing) return;
    simulateExecution(input.trim());
    setInput('');
  };

  const handleQuickAction = (query) => {
    setInput(query);
    simulateExecution(query);
  };

  const getStageStatus = (stageId) => {
    if (completedStages.includes(stageId)) return 'completed';
    if (pipelineStage === stageId) return 'active';
    return 'pending';
  };

  return (
    <div className="h-screen flex bg-[#0a0a0a] text-gray-100 overflow-hidden">
      {/* Left Sidebar */}
      <div className="w-72 bg-[#111] border-r border-white/5 flex flex-col">
        {/* Logo */}
        <div className="p-5 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
              <Icons.Sparkles />
            </div>
            <div>
              <h1 className="font-bold text-lg">MLOps Agent</h1>
              <p className="text-xs text-gray-500">AI-Powered Automation</p>
            </div>
          </div>
        </div>

        {/* New Run Button */}
        <div className="p-4">
          <button
            onClick={() => { setShowWelcome(true); setMessages([]); setExecutionSteps([]); }}
            className="w-full py-3 px-4 bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 rounded-xl font-medium flex items-center justify-center gap-2 transition-all shadow-lg shadow-orange-500/20 hover:shadow-orange-500/30"
          >
            <Icons.Plus /> New Pipeline
          </button>
        </div>

        {/* Navigation */}
        <div className="px-4 mb-4">
          <div className="flex gap-1 p-1 bg-white/5 rounded-lg">
            {['overview', 'history'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-2 text-sm rounded-md transition-all ${
                  activeTab === tab ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Recent Runs */}
        <div className="flex-1 overflow-y-auto px-4">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Icons.Clock /> Recent Runs
          </h3>
          <div className="space-y-2">
            {RECENT_RUNS.map((run) => (
              <div
                key={run.id}
                onClick={() => setSelectedRun(run.id)}
                className={`p-3 rounded-xl cursor-pointer transition-all ${
                  selectedRun === run.id
                    ? 'bg-orange-500/10 border border-orange-500/30'
                    : 'bg-white/5 hover:bg-white/10 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="text-lg">{run.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">{run.name}</div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        run.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                        run.status === 'running' ? 'bg-blue-500/20 text-blue-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {run.status === 'running' && <span className="inline-block w-1.5 h-1.5 bg-blue-400 rounded-full mr-1 animate-pulse" />}
                        {run.accuracy ? `${(run.accuracy * 100).toFixed(0)}%` : run.status}
                      </span>
                      <span className="text-xs text-gray-500">{run.time}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Status Footer */}
        <div className="p-4 border-t border-white/5">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2 text-gray-400">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              All systems operational
            </div>
            <button className="text-gray-500 hover:text-white transition-colors">
              <Icons.Settings />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Pipeline Progress Header */}
        <div className="h-16 border-b border-white/5 bg-[#111]/50 backdrop-blur-xl flex items-center px-6">
          <div className="flex items-center gap-1">
            {PIPELINE_STAGES.map((stage, idx) => {
              const status = getStageStatus(stage.id);
              const StageIcon = stage.icon;
              return (
                <React.Fragment key={stage.id}>
                  <div className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
                    status === 'completed' ? 'bg-green-500/10 text-green-400' :
                    status === 'active' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                    'text-gray-600'
                  }`}>
                    {status === 'completed' ? <Icons.CheckCircle /> : <StageIcon />}
                    <span className="text-sm font-medium">{stage.name}</span>
                  </div>
                  {idx < PIPELINE_STAGES.length - 1 && (
                    <div className={`w-8 h-0.5 ${
                      status === 'completed' ? 'bg-green-500/50' : 'bg-white/10'
                    }`} />
                  )}
                </React.Fragment>
              );
            })}
          </div>

          {/* Live Accuracy Badge */}
          {currentAccuracy > 0 && (
            <div className="ml-auto flex items-center gap-4">
              <div className={`flex items-center gap-3 px-4 py-2 rounded-xl ${
                currentAccuracy >= targetAccuracy
                  ? 'bg-green-500/10 border border-green-500/30'
                  : 'bg-orange-500/10 border border-orange-500/30'
              }`}>
                <Icons.TrendingUp />
                <div>
                  <div className="text-xs text-gray-400">Accuracy</div>
                  <div className={`text-lg font-bold ${
                    currentAccuracy >= targetAccuracy ? 'text-green-400' : 'text-orange-400'
                  }`}>
                    {(currentAccuracy * 100).toFixed(1)}%
                  </div>
                </div>
                {currentAccuracy >= targetAccuracy && (
                  <div className="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center">
                    <Icons.Check />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto p-6">
            {showWelcome ? (
              /* Welcome Screen */
              <div className="py-12">
                {/* Hero */}
                <div className="text-center mb-12">
                  <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-orange-500/10 border border-orange-500/20 text-orange-400 text-sm mb-6">
                    <Icons.Zap /> Powered by AI
                  </div>
                  <h1 className="text-4xl font-bold mb-4">
                    What would you like to{' '}
                    <span className="bg-gradient-to-r from-orange-400 to-amber-400 bg-clip-text text-transparent">
                      build today?
                    </span>
                  </h1>
                  <p className="text-gray-400 text-lg max-w-xl mx-auto">
                    Describe your ML pipeline in natural language. I'll handle Hydra configs,
                    MLflow tracking, DVC versioning, Docker containers, and CI/CD setup.
                  </p>
                </div>

                {/* Quick Actions */}
                <div className="grid grid-cols-2 gap-4 mb-12">
                  {QUICK_ACTIONS.map((action, i) => {
                    const ActionIcon = action.icon;
                    return (
                      <button
                        key={i}
                        onClick={() => handleQuickAction(action.query)}
                        className="group p-5 rounded-2xl bg-white/5 border border-white/10 hover:border-white/20 text-left transition-all hover:bg-white/[0.07]"
                      >
                        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${action.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform shadow-lg`}>
                          <ActionIcon />
                        </div>
                        <h3 className="font-semibold text-lg mb-1">{action.title}</h3>
                        <p className="text-gray-400 text-sm">{action.desc}</p>
                      </button>
                    );
                  })}
                </div>

                {/* Tool Stack */}
                <div className="flex items-center justify-center gap-8 py-6 border-y border-white/5">
                  {[
                    { name: 'Hydra', icon: '⚙️' },
                    { name: 'MLflow', icon: '📊' },
                    { name: 'DVC', icon: '📦' },
                    { name: 'Docker', icon: '🐳' },
                    { name: 'GitHub Actions', icon: '⚡' },
                  ].map((tool) => (
                    <div key={tool.name} className="flex items-center gap-2 text-gray-500">
                      <span>{tool.icon}</span>
                      <span className="text-sm">{tool.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              /* Chat Messages */
              <div className="space-y-6 py-6">
                {messages.map((msg) => (
                  <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-2xl ${
                      msg.role === 'user'
                        ? 'bg-gradient-to-r from-orange-500 to-amber-500 rounded-2xl rounded-tr-sm px-5 py-3'
                        : 'bg-white/5 border border-white/10 rounded-2xl rounded-tl-sm px-5 py-4'
                    }`}>
                      <div className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</div>
                    </div>
                  </div>
                ))}

                {/* Execution Steps */}
                {executionSteps.length > 0 && (
                  <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-semibold flex items-center gap-2">
                        <Icons.Terminal /> Pipeline Execution
                      </h3>
                      <span className="text-sm text-gray-400">
                        {executionSteps.filter(s => s.status === 'completed').length}/{executionSteps.length} completed
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden mb-6">
                      <div
                        className="h-full bg-gradient-to-r from-orange-500 to-amber-500 transition-all duration-500"
                        style={{ width: `${(executionSteps.filter(s => s.status === 'completed').length / executionSteps.length) * 100}%` }}
                      />
                    </div>

                    <div className="space-y-3">
                      {executionSteps.map((step, idx) => (
                        <div
                          key={step.id}
                          className={`flex items-center gap-4 p-3 rounded-xl transition-all ${
                            step.status === 'running' ? 'bg-orange-500/10 border border-orange-500/30' :
                            step.status === 'completed' ? 'bg-green-500/5' : 'bg-white/5'
                          }`}
                        >
                          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                            step.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                            step.status === 'running' ? 'bg-orange-500/20 text-orange-400' :
                            'bg-white/10 text-gray-500'
                          }`}>
                            {step.status === 'completed' ? <Icons.Check /> :
                             step.status === 'running' ? <Icons.Loader /> :
                             <span className="text-sm">{idx + 1}</span>}
                          </div>
                          <div className="flex-1">
                            <div className="text-sm font-medium">{step.description}</div>
                            <div className="text-xs text-gray-500">{step.tool}</div>
                          </div>
                          {step.status === 'running' && (
                            <div className="text-xs text-orange-400 animate-pulse">Running...</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Thinking Indicator */}
                {agentThinking && (
                  <div className="flex items-center gap-3 text-gray-400 px-2">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}} />
                      <div className="w-2 h-2 bg-yellow-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}} />
                    </div>
                    <span className="text-sm">{agentThinking}</span>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-white/5 bg-[#111]/50 backdrop-blur-xl p-4">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSubmit())}
                placeholder="Describe your ML pipeline... (e.g., 'Train a ResNet model on ImageNet with 90% accuracy')"
                disabled={isProcessing}
                rows={1}
                className="w-full bg-white/5 border border-white/10 focus:border-orange-500/50 rounded-2xl px-5 py-4 pr-14 text-white placeholder-gray-500 resize-none focus:outline-none focus:ring-2 focus:ring-orange-500/20 transition-all"
              />
              <button
                onClick={handleSubmit}
                disabled={!input.trim() || isProcessing}
                className={`absolute right-3 top-1/2 -translate-y-1/2 p-2.5 rounded-xl transition-all ${
                  input.trim() && !isProcessing
                    ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white shadow-lg shadow-orange-500/25 hover:shadow-orange-500/40'
                    : 'bg-white/10 text-gray-500'
                }`}
              >
                {isProcessing ? <Icons.Loader /> : <Icons.Send />}
              </button>
            </div>

            <div className="flex items-center justify-between mt-3 px-2">
              <div className="flex items-center gap-4 text-xs text-gray-500">
                <span>Press Enter to send</span>
                <span>•</span>
                <span>Shift + Enter for new line</span>
              </div>
              <div className="text-xs text-gray-500">
                Powered by GPT-4 & Gemini
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Sidebar - Metrics */}
      <div className="w-80 bg-[#111] border-l border-white/5 flex flex-col">
        {/* Tabs */}
        <div className="flex border-b border-white/5">
          {['Metrics', 'Config', 'Logs'].map(tab => (
            <button
              key={tab}
              className={`flex-1 py-4 text-sm font-medium transition-all ${
                tab === 'Metrics' ? 'text-orange-400 border-b-2 border-orange-400' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          {/* Stats Grid */}
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Accuracy', value: `${(currentAccuracy * 100).toFixed(1)}%`, color: 'text-green-400', bg: 'bg-green-500/10' },
              { label: 'Target', value: `${(targetAccuracy * 100).toFixed(0)}%`, color: 'text-cyan-400', bg: 'bg-cyan-500/10' },
              { label: 'Epoch', value: `${metricsData.length}/8`, color: 'text-purple-400', bg: 'bg-purple-500/10' },
              { label: 'Loss', value: metricsData.length > 0 ? metricsData[metricsData.length - 1].loss.toFixed(2) : '—', color: 'text-orange-400', bg: 'bg-orange-500/10' },
            ].map((stat, i) => (
              <div key={i} className={`p-4 rounded-xl ${stat.bg}`}>
                <div className="text-xs text-gray-400 mb-1">{stat.label}</div>
                <div className={`text-2xl font-bold ${stat.color}`}>{stat.value}</div>
              </div>
            ))}
          </div>

          {/* Chart */}
          {metricsData.length > 0 && (
            <div className="bg-white/5 rounded-xl p-4">
              <h4 className="text-sm font-medium mb-4 flex items-center gap-2">
                <Icons.TrendingUp /> Training Progress
              </h4>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metricsData}>
                    <defs>
                      <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#f97316" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="epoch" stroke="#4b5563" fontSize={10} />
                    <YAxis stroke="#4b5563" fontSize={10} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '12px' }}
                      labelStyle={{ color: '#9ca3af' }}
                    />
                    <Area type="monotone" dataKey="accuracy" stroke="#f97316" strokeWidth={2} fillOpacity={1} fill="url(#colorAcc)" />
                    <Line type="monotone" dataKey="val_accuracy" stroke="#06b6d4" strokeDasharray="5 5" strokeWidth={2} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex items-center justify-center gap-6 mt-3 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-orange-500" />
                  <span className="text-gray-400">Train Acc</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 bg-cyan-500" style={{width: '12px'}} />
                  <span className="text-gray-400">Val Acc</span>
                </div>
              </div>
            </div>
          )}

          {/* Threshold Progress */}
          <div className="bg-white/5 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium">Target Threshold</span>
              <span className={`text-sm font-medium ${
                currentAccuracy >= targetAccuracy ? 'text-green-400' : 'text-orange-400'
              }`}>
                {currentAccuracy >= targetAccuracy ? '✓ Achieved' : 'In Progress'}
              </span>
            </div>
            <div className="h-3 bg-white/10 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  currentAccuracy >= targetAccuracy
                    ? 'bg-gradient-to-r from-green-500 to-emerald-500'
                    : 'bg-gradient-to-r from-orange-500 to-amber-500'
                }`}
                style={{ width: `${Math.min((currentAccuracy / targetAccuracy) * 100, 100)}%` }}
              />
            </div>
            <div className="flex justify-between mt-2 text-xs text-gray-500">
              <span>0%</span>
              <span>{(targetAccuracy * 100).toFixed(0)}% target</span>
            </div>
          </div>

          {/* Quick Info */}
          <div className="bg-gradient-to-br from-orange-500/10 to-amber-500/10 border border-orange-500/20 rounded-xl p-4">
            <div className="flex items-center gap-2 text-orange-400 mb-2">
              <Icons.Sparkles />
              <span className="text-sm font-medium">Self-Improvement Active</span>
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">
              The agent automatically analyzes results and suggests optimizations to reach your target accuracy.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
