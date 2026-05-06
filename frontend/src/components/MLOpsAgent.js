import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const Icon = ({ children, className = 'h-4 w-4' }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    {children}
  </svg>
);

const Icons = {
  Activity: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M22 12h-4l-3 8L9 4l-3 8H2" />
    </Icon>
  ),
  Alert: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v4m0 4h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    </Icon>
  ),
  Check: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m5 13 4 4L19 7" />
    </Icon>
  ),
  Clock: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" />
    </Icon>
  ),
  Database: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7c0 2.21 3.58 4 8 4s8-1.79 8-4M4 7c0-2.21 3.58-4 8-4s8 1.79 8 4m-16 0v10c0 2.21 3.58 4 8 4s8-1.79 8-4V7" />
    </Icon>
  ),
  File: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l5 5v11a2 2 0 0 1-2 2z" />
    </Icon>
  ),
  Gate: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 21V9m16 12V9M4 9h16M7 9V5a5 5 0 0 1 10 0v4" />
    </Icon>
  ),
  Git: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 3v12m0 0a3 3 0 1 0 3 3 3 3 0 0 0-3-3zm12-3a3 3 0 1 1-6 0 3 3 0 0 1 6 0zm-3 0V6a3 3 0 0 0-3-3H9" />
    </Icon>
  ),
  Play: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5v14l11-7-11-7z" />
    </Icon>
  ),
  Refresh: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h5M20 20v-5h-5M5 15a7 7 0 0 0 11.9 3.9M19 9A7 7 0 0 0 7.1 5.1" />
    </Icon>
  ),
  Server: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2M5 12a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-4a2 2 0 0 0-2-2" />
    </Icon>
  ),
  Terminal: (props) => (
    <Icon {...props}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m8 9 3 3-3 3m5 0h3M5 20h14a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2z" />
    </Icon>
  ),
};

const WORKFLOWS = [
  { id: 'setup', name: 'Setup', phase: '0', status: 'succeeded', count: 18 },
  { id: 'data', name: 'Data', phase: '1', status: 'blocked', count: 9 },
  { id: 'train', name: 'Train', phase: '2', status: 'succeeded', count: 14 },
  { id: 'deploy', name: 'Deploy', phase: '3', status: 'deferred', count: 7 },
  { id: 'container-ci', name: 'Container/CI', phase: '4', status: 'failed', count: 5 },
  { id: 'capstone', name: 'Capstone', phase: '5', status: 'succeeded', count: 3 },
];

const RUNS = [
  {
    id: 'run-1842',
    workflow: 'Capstone',
    command: 'Phase 5 readiness audit with data, container, CI, and endpoint evidence',
    status: 'succeeded',
    time: '12 min ago',
    artifacts: 12,
    gates: '2/2 approved',
  },
  {
    id: 'run-1839',
    workflow: 'Data',
    command: 'Validate data-stage lineage, DVC remote, schema drift, and quality report',
    status: 'blocked',
    time: '44 min ago',
    artifacts: 8,
    gates: '1 approval pending',
  },
  {
    id: 'run-1835',
    workflow: 'Container/CI',
    command: 'Build image, run smoke tests, and publish CI evidence',
    status: 'failed',
    time: '2h ago',
    artifacts: 6,
    gates: 'blocked by CI',
  },
  {
    id: 'run-1828',
    workflow: 'Deploy',
    command: 'Prepare endpoint rollout and defer Kubernetes scaling extension',
    status: 'deferred',
    time: 'Yesterday',
    artifacts: 5,
    gates: 'extension deferred',
  },
];

const TIMELINE = [
  {
    phase: 'Phase 0',
    label: 'Command normalized',
    status: 'succeeded',
    detail: 'Workflow intent mapped to capstone readiness contract.',
    evidence: 'intent.json',
  },
  {
    phase: 'Phase 1',
    label: 'Data-stage evidence collected',
    status: 'succeeded',
    detail: 'Lineage, schema, and quality reports available for review.',
    evidence: 'data_manifest.yaml',
  },
  {
    phase: 'Phase 2',
    label: 'Training artifacts resolved',
    status: 'succeeded',
    detail: 'MLflow run, metrics summary, and model card are linked.',
    evidence: 'mlflow_run.json',
  },
  {
    phase: 'Phase 3',
    label: 'Approval gate evaluated',
    status: 'blocked',
    detail: 'Endpoint promotion requires operator approval before deploy.',
    evidence: 'approval_gate.md',
  },
  {
    phase: 'Phase 4',
    label: 'Container and CI checked',
    status: 'failed',
    detail: 'Smoke test is red on missing model checksum environment variable.',
    evidence: 'ci_smoke_test.log',
  },
  {
    phase: 'Phase 5',
    label: 'Capstone readiness scored',
    status: 'deferred',
    detail: 'Kubernetes autoscaling evidence is deferred, not implemented here.',
    evidence: 'readiness_report.md',
  },
];

const ARTIFACTS = [
  { name: 'workflow_evidence_manifest.json', type: 'manifest', owner: 'phase-5', status: 'signed' },
  { name: 'data_quality_report.html', type: 'data', owner: 'phase-1', status: 'ready' },
  { name: 'mlflow_metrics_summary.json', type: 'training', owner: 'phase-2', status: 'ready' },
  { name: 'container_sbom.spdx', type: 'container', owner: 'phase-4', status: 'ready' },
  { name: 'github_actions_smoke.log', type: 'ci', owner: 'phase-4', status: 'failed' },
  { name: 'endpoint_contract_check.json', type: 'deploy', owner: 'phase-3', status: 'blocked' },
];

const APPROVAL_GATES = [
  { name: 'Data acceptance', status: 'approved', by: 'operator', updated: '10:18 UTC' },
  { name: 'Endpoint promotion', status: 'blocked', by: 'pending', updated: 'awaiting approval' },
  { name: 'Phase 6 extension', status: 'deferred', by: 'roadmap', updated: 'out of scope' },
];

const DATA_EVIDENCE = [
  ['Lineage', 'data/raw -> features/train.parquet', 'succeeded'],
  ['Schema', '42 columns validated, 0 breaking changes', 'succeeded'],
  ['Quality', '2 warnings under threshold', 'blocked'],
  ['Versioning', 'DVC remote configured', 'succeeded'],
];

const CONTAINER_EVIDENCE = [
  ['Image build', 'ghcr.io/auto-mlops/capstone:1842', 'succeeded'],
  ['SBOM', 'SPDX document attached', 'succeeded'],
  ['Smoke test', 'MODEL_CHECKSUM missing in CI environment', 'failed'],
  ['Workflow run', 'GitHub Actions evidence linked', 'blocked'],
];

const CONTRACT_CHECKS = [
  ['API compatibility', 'Current backend endpoints only', 'succeeded'],
  ['Workflow contracts', 'No contract edits detected', 'succeeded'],
  ['Phase 6', 'Deferred capability remains unimplemented', 'deferred'],
  ['Approval semantics', 'Blocked states require operator action', 'blocked'],
];

const READINESS_DATA = [
  { label: 'P0', score: 82 },
  { label: 'P1', score: 88 },
  { label: 'P2', score: 91 },
  { label: 'P3', score: 78 },
  { label: 'P4', score: 64 },
  { label: 'P5', score: 84 },
];

const statusStyles = {
  approved: 'border-emerald-700 bg-emerald-950/40 text-emerald-300',
  blocked: 'border-amber-700 bg-amber-950/40 text-amber-300',
  deferred: 'border-sky-700 bg-sky-950/40 text-sky-300',
  failed: 'border-rose-700 bg-rose-950/40 text-rose-300',
  ready: 'border-emerald-700 bg-emerald-950/40 text-emerald-300',
  signed: 'border-emerald-700 bg-emerald-950/40 text-emerald-300',
  succeeded: 'border-emerald-700 bg-emerald-950/40 text-emerald-300',
};

const statusIcon = {
  approved: Icons.Check,
  blocked: Icons.Clock,
  deferred: Icons.Refresh,
  failed: Icons.Alert,
  ready: Icons.File,
  signed: Icons.File,
  succeeded: Icons.Check,
};

function StatusBadge({ status }) {
  const StatusIcon = statusIcon[status] || Icons.Activity;

  return (
    <span className={`inline-flex min-w-0 items-center gap-1 rounded-md border px-2 py-1 text-xs font-medium ${statusStyles[status] || 'border-slate-700 bg-slate-900 text-slate-300'}`}>
      <StatusIcon className="h-3.5 w-3.5 flex-none" />
      <span className="truncate capitalize">{status}</span>
    </span>
  );
}

function Section({ title, icon: SectionIcon, children, action }) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-950">
      <div className="flex items-center justify-between gap-3 border-b border-slate-800 px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          {SectionIcon ? <SectionIcon className="h-4 w-4 flex-none text-slate-400" /> : null}
          <h2 className="truncate text-sm font-semibold text-slate-100">{title}</h2>
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

function EvidenceTable({ rows }) {
  return (
    <div className="divide-y divide-slate-800">
      {rows.map(([name, detail, status]) => (
        <div key={name} className="grid grid-cols-[96px_minmax(0,1fr)_92px] items-start gap-3 px-4 py-3 text-sm max-sm:grid-cols-1">
          <div className="font-medium text-slate-200">{name}</div>
          <div className="min-w-0 break-words text-slate-400">{detail}</div>
          <div className="justify-self-end max-sm:justify-self-start">
            <StatusBadge status={status} />
          </div>
        </div>
      ))}
    </div>
  );
}

function MetricCard({ label, value, status }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="truncate text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
        {status ? <StatusBadge status={status} /> : null}
      </div>
      <div className="truncate text-2xl font-semibold text-slate-100">{value}</div>
    </div>
  );
}

export default function MLOpsAgent() {
  const [input, setInput] = useState('Run Phase 5 capstone readiness audit with data, container, CI, and endpoint evidence');
  const [selectedWorkflow, setSelectedWorkflow] = useState('capstone');
  const [selectedRunId, setSelectedRunId] = useState(RUNS[0].id);
  const [currentRun, setCurrentRun] = useState(RUNS[0]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [backendMetrics, setBackendMetrics] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  const selectedRun = useMemo(
    () => (selectedRunId === currentRun.id ? currentRun : RUNS.find((run) => run.id === selectedRunId) || currentRun),
    [currentRun, selectedRunId]
  );

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/metrics`);
      if (response.ok) {
        const data = await response.json();
        setBackendMetrics(data);
        setIsConnected(true);
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
      setIsConnected(false);
    }
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/logs?page_size=50`);
      if (response.ok) {
        const data = await response.json();
        setLogs(data.logs || []);
      }
    } catch (error) {
      console.error('Failed to fetch logs:', error);
    }
  }, []);

  const initDemoData = useCallback(async () => {
    try {
      await fetch(`${API_BASE_URL}/metrics/demo`);
      fetchMetrics();
      fetchLogs();
    } catch (error) {
      console.error('Failed to init demo data:', error);
    }
  }, [fetchMetrics, fetchLogs]);

  useEffect(() => {
    let shouldReconnect = true;

    const connectWebSocket = () => {
      const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/metrics`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => setIsConnected(true);
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'metrics_update') {
            setBackendMetrics(data.data);
          }
        } catch (error) {
          console.error('WebSocket message error:', error);
        }
      };
      wsRef.current.onerror = () => setIsConnected(false);
      wsRef.current.onclose = () => {
        setIsConnected(false);
        if (shouldReconnect) {
          setTimeout(connectWebSocket, 5000);
        }
      };
    };

    fetchMetrics();
    fetchLogs();
    connectWebSocket();

    const pollInterval = setInterval(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        fetchMetrics();
      }
      fetchLogs();
    }, 10000);

    return () => {
      shouldReconnect = false;
      clearInterval(pollInterval);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [fetchMetrics, fetchLogs]);

  const runCommand = () => {
    const command = input.trim();
    if (!command || isProcessing) return;

    setIsProcessing(true);
    const nextRun = {
      id: `run-${Math.floor(1900 + Math.random() * 90)}`,
      workflow: WORKFLOWS.find((workflow) => workflow.id === selectedWorkflow)?.name || 'Workflow',
      command,
      status: 'blocked',
      time: 'Running',
      artifacts: 0,
      gates: 'evaluating',
    };
    setCurrentRun(nextRun);
    setSelectedRunId(nextRun.id);

    setTimeout(() => {
      const normalized = command.toLowerCase();
      const status = normalized.includes('fail')
        ? 'failed'
        : normalized.includes('defer') || normalized.includes('phase 6')
          ? 'deferred'
          : normalized.includes('approve')
            ? 'succeeded'
            : 'blocked';

      setCurrentRun({
        ...nextRun,
        status,
        time: 'Just now',
        artifacts: status === 'failed' ? 7 : 12,
        gates: status === 'succeeded' ? '2/2 approved' : status === 'deferred' ? 'extension deferred' : 'operator action required',
      });
      setIsProcessing(false);
    }, 900);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)_340px]">
        <aside className="border-b border-slate-800 bg-slate-950 lg:border-b-0 lg:border-r">
          <div className="border-b border-slate-800 px-4 py-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-700 bg-slate-900 text-sm font-semibold text-cyan-300">
                AM
              </div>
              <div className="min-w-0">
                <h1 className="truncate text-sm font-semibold">Auto-mlops</h1>
                <p className="truncate text-xs text-slate-500">Workflow evidence console</p>
              </div>
            </div>
          </div>

          <div className="px-3 py-4">
            <div className="mb-2 px-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Workflows</div>
            <nav className="space-y-1">
              {WORKFLOWS.map((workflow) => (
                <button
                  key={workflow.id}
                  onClick={() => setSelectedWorkflow(workflow.id)}
                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition ${
                    selectedWorkflow === workflow.id
                      ? 'border border-slate-700 bg-slate-900 text-white'
                      : 'text-slate-400 hover:bg-slate-900 hover:text-slate-100'
                  }`}
                >
                  <span className="flex h-7 w-7 flex-none items-center justify-center rounded-md border border-slate-800 bg-slate-950 text-xs text-slate-400">
                    P{workflow.phase}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-medium">{workflow.name}</span>
                    <span className="block truncate text-xs text-slate-500">{workflow.count} runs tracked</span>
                  </span>
                  <span className={`h-2 w-2 flex-none rounded-full ${
                    workflow.status === 'succeeded' ? 'bg-emerald-400' :
                    workflow.status === 'blocked' ? 'bg-amber-400' :
                    workflow.status === 'failed' ? 'bg-rose-400' :
                    'bg-sky-400'
                  }`} />
                </button>
              ))}
            </nav>
          </div>

          <div className="border-t border-slate-800 px-3 py-4">
            <div className="mb-2 px-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Recent Runs</div>
            <div className="space-y-2">
              {[currentRun, ...RUNS.filter((run) => run.id !== currentRun.id)].slice(0, 5).map((run) => (
                <button
                  key={run.id}
                  onClick={() => setSelectedRunId(run.id)}
                  className={`w-full rounded-lg border p-3 text-left transition ${
                    selectedRun.id === run.id
                      ? 'border-slate-600 bg-slate-900'
                      : 'border-slate-800 bg-transparent hover:bg-slate-900'
                  }`}
                >
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <span className="truncate text-xs font-medium text-slate-300">{run.id}</span>
                    <StatusBadge status={run.status} />
                  </div>
                  <div className="line-clamp-2 break-words text-xs leading-5 text-slate-500">{run.command}</div>
                </button>
              ))}
            </div>
          </div>
        </aside>

        <main className="min-w-0 bg-slate-950">
          <div className="border-b border-slate-800 px-4 py-4 sm:px-6">
            <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <span className="rounded-md border border-slate-800 px-2 py-1">Phases 0-5</span>
              <span className="rounded-md border border-slate-800 px-2 py-1">Backend-compatible API calls</span>
              <span className="rounded-md border border-slate-800 px-2 py-1">No Phase 6 execution</span>
            </div>
            <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
              <label htmlFor="workflow-command" className="mb-2 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                Workflow Command Composer
              </label>
              <div className="flex gap-2 max-sm:flex-col">
                <textarea
                  id="workflow-command"
                  rows={2}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault();
                      runCommand();
                    }
                  }}
                  className="min-h-[72px] flex-1 resize-none rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm leading-6 text-slate-100 placeholder:text-slate-600 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                  placeholder="Describe a Phase 0-5 workflow command..."
                />
                <button
                  onClick={runCommand}
                  disabled={!input.trim() || isProcessing}
                  className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-md bg-cyan-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
                >
                  {isProcessing ? <Icons.Refresh className="h-4 w-4 animate-spin" /> : <Icons.Play className="h-4 w-4" />}
                  <span>{isProcessing ? 'Running' : 'Run'}</span>
                </button>
              </div>
            </div>
          </div>

          <div className="space-y-4 p-4 sm:p-6">
            <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
              <MetricCard label="Current status" value={selectedRun.status} status={selectedRun.status} />
              <MetricCard label="Artifacts" value={selectedRun.artifacts} />
              <MetricCard label="Approval gates" value={selectedRun.gates} />
              <MetricCard label="Backend sessions" value={backendMetrics?.agent?.total_sessions || 0} />
            </div>

            <Section title="Current Workflow Run" icon={Icons.Terminal}>
              <div className="space-y-4 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="mb-1 flex flex-wrap items-center gap-2">
                      <span className="font-mono text-xs text-slate-500">{selectedRun.id}</span>
                      <StatusBadge status={selectedRun.status} />
                    </div>
                    <h2 className="break-words text-lg font-semibold text-slate-100">{selectedRun.workflow}</h2>
                    <p className="mt-1 break-words text-sm leading-6 text-slate-400">{selectedRun.command}</p>
                  </div>
                  <div className="rounded-md border border-slate-800 px-3 py-2 text-right text-xs text-slate-500">
                    Updated<br />
                    <span className="text-slate-300">{selectedRun.time}</span>
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-3">
                  <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                    <div className="text-xs text-slate-500">Data-stage evidence</div>
                    <div className="mt-1 text-sm font-medium text-slate-100">Lineage and quality present</div>
                  </div>
                  <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                    <div className="text-xs text-slate-500">Container/CI evidence</div>
                    <div className="mt-1 text-sm font-medium text-amber-300">Smoke test needs operator review</div>
                  </div>
                  <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                    <div className="text-xs text-slate-500">Capstone readiness</div>
                    <div className="mt-1 text-sm font-medium text-slate-100">84% ready across P0-P5</div>
                  </div>
                </div>
              </div>
            </Section>

            <Section title="Evidence Timeline" icon={Icons.Activity}>
              <div className="divide-y divide-slate-800">
                {TIMELINE.map((item) => (
                  <div key={`${item.phase}-${item.label}`} className="grid grid-cols-[78px_minmax(0,1fr)_124px] gap-3 px-4 py-3 text-sm max-sm:grid-cols-1">
                    <div className="font-mono text-xs text-slate-500">{item.phase}</div>
                    <div className="min-w-0">
                      <div className="break-words font-medium text-slate-100">{item.label}</div>
                      <div className="mt-1 break-words text-slate-400">{item.detail}</div>
                      <div className="mt-1 truncate font-mono text-xs text-slate-500">{item.evidence}</div>
                    </div>
                    <div className="justify-self-end max-sm:justify-self-start">
                      <StatusBadge status={item.status} />
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            <div className="grid gap-4 xl:grid-cols-2">
              <Section title="Approval Gates" icon={Icons.Gate}>
                <div className="divide-y divide-slate-800">
                  {APPROVAL_GATES.map((gate) => (
                    <div key={gate.name} className="flex items-start justify-between gap-3 px-4 py-3">
                      <div className="min-w-0">
                        <div className="break-words text-sm font-medium text-slate-100">{gate.name}</div>
                        <div className="mt-1 truncate text-xs text-slate-500">{gate.by} / {gate.updated}</div>
                      </div>
                      <StatusBadge status={gate.status} />
                    </div>
                  ))}
                </div>
              </Section>

              <Section title="Capstone Pipeline Readiness" icon={Icons.Activity}>
                <div className="h-56 p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={READINESS_DATA} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
                      <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                      <XAxis dataKey="label" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{ background: '#020617', border: '1px solid #334155', borderRadius: 8 }}
                        labelStyle={{ color: '#cbd5e1' }}
                      />
                      <Area dataKey="score" type="monotone" stroke="#06b6d4" strokeWidth={2} fill="#0891b2" fillOpacity={0.18} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </Section>
            </div>

            <div className="grid gap-4 xl:grid-cols-2">
              <Section title="Data-Stage Evidence" icon={Icons.Database}>
                <EvidenceTable rows={DATA_EVIDENCE} />
              </Section>
              <Section title="Container/CI Evidence" icon={Icons.Git}>
                <EvidenceTable rows={CONTAINER_EVIDENCE} />
              </Section>
            </div>
          </div>
        </main>

        <aside className="border-t border-slate-800 bg-slate-950 lg:border-l lg:border-t-0">
          <div className="flex items-center justify-between gap-3 border-b border-slate-800 px-4 py-3">
            <div className="flex min-w-0 items-center gap-2 text-xs">
              <span className={`h-2 w-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-rose-400'}`} />
              <span className="truncate text-slate-400">{isConnected ? 'Backend connected' : 'Backend disconnected'}</span>
            </div>
            <button
              onClick={initDemoData}
              className="rounded-md border border-slate-700 px-2 py-1 text-xs font-medium text-slate-300 hover:bg-slate-900"
            >
              Load demo
            </button>
          </div>

          <div className="space-y-4 p-4">
            <Section title="Artifact Manifest" icon={Icons.File}>
              <div className="divide-y divide-slate-800">
                {ARTIFACTS.map((artifact) => (
                  <div key={artifact.name} className="px-4 py-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="break-words font-mono text-xs text-slate-200">{artifact.name}</div>
                        <div className="mt-1 truncate text-xs text-slate-500">{artifact.owner} / {artifact.type}</div>
                      </div>
                      <StatusBadge status={artifact.status} />
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            <Section title="Contract Checks" icon={Icons.Check}>
              <EvidenceTable rows={CONTRACT_CHECKS} />
            </Section>

            <Section title="Deferred Capabilities" icon={Icons.Refresh}>
              <div className="space-y-2 p-4 text-sm text-slate-400">
                <div className="rounded-lg border border-sky-800 bg-sky-950/30 p-3">
                  <div className="mb-1 font-medium text-sky-200">Phase 6 automation</div>
                  <p className="break-words text-xs leading-5">Recorded as deferred evidence only. The console does not execute roadmap capabilities.</p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                  <div className="mb-1 font-medium text-slate-200">Kubernetes autoscaling</div>
                  <p className="break-words text-xs leading-5">Tracked in readiness output without changing workflow contracts.</p>
                </div>
              </div>
            </Section>

            <Section title="Endpoint/Evidence Summary" icon={Icons.Server}>
              <div className="space-y-3 p-4 text-sm">
                <div className="flex justify-between gap-3">
                  <span className="text-slate-500">API base</span>
                  <span className="min-w-0 truncate font-mono text-xs text-slate-300">{API_BASE_URL}</span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-slate-500">Success rate</span>
                  <span className="text-slate-200">{backendMetrics?.agent?.success_rate?.toFixed(1) || '0.0'}%</span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-slate-500">Tools available</span>
                  <span className="text-slate-200">{backendMetrics?.pipeline?.tools_available || 28}</span>
                </div>
              </div>
            </Section>

            <Section title="Recent Evidence Logs" icon={Icons.Terminal}>
              <div className="max-h-72 overflow-y-auto p-3">
                {logs.length === 0 ? (
                  <div className="rounded-lg border border-slate-800 p-3 text-sm text-slate-500">No backend logs loaded yet.</div>
                ) : (
                  <div className="space-y-2">
                    {logs.slice(0, 8).map((log, index) => (
                      <div key={log.id || index} className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                        <div className="mb-1 flex items-center justify-between gap-2">
                          <span className="truncate text-xs font-medium uppercase text-slate-400">{log.level || 'info'}</span>
                          <span className="truncate text-xs text-slate-600">{log.source || 'backend'}</span>
                        </div>
                        <p className="break-words text-xs leading-5 text-slate-300">{log.message}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </Section>
          </div>
        </aside>
      </div>
    </div>
  );
}
