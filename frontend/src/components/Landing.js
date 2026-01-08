import React, { useState, useEffect, useRef } from 'react';

// Animated Counter Component
const AnimatedCounter = ({ end, duration = 2000, suffix = '', prefix = '' }) => {
  const [count, setCount] = useState(0);
  const [hasAnimated, setHasAnimated] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated) {
          setHasAnimated(true);
          let startTime;
          const animate = (currentTime) => {
            if (!startTime) startTime = currentTime;
            const progress = Math.min((currentTime - startTime) / duration, 1);
            setCount(Math.floor(progress * end));
            if (progress < 1) {
              requestAnimationFrame(animate);
            }
          };
          requestAnimationFrame(animate);
        }
      },
      { threshold: 0.1 }
    );

    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [end, duration, hasAnimated]);

  return (
    <span ref={ref}>
      {prefix}{count.toLocaleString()}{suffix}
    </span>
  );
};

// Interactive Terminal Component
const InteractiveTerminal = () => {
  const [input, setInput] = useState('');
  const [history, setHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const inputRef = useRef(null);

  const exampleQueries = [
    'Train a ResNet model on ImageNet with 90% accuracy',
    'Set up DVC with S3 for my dataset',
    'Create a CI/CD pipeline for model deployment',
    'Deploy cat-dog classifier with MLflow tracking',
  ];

  const simulatedSteps = [
    { text: 'Analyzing request...', delay: 800 },
    { text: '✓ Detected: ML training pipeline', delay: 600 },
    { text: '✓ Initializing Hydra configuration', delay: 700 },
    { text: '✓ Setting up MLflow experiment tracking', delay: 800 },
    { text: '✓ Configuring DVC data pipeline', delay: 700 },
    { text: '✓ Creating Docker training environment', delay: 900 },
    { text: '⚡ Starting model training...', delay: 1000 },
    { text: '📊 Epoch 1/10 - Accuracy: 45.2%', delay: 500 },
    { text: '📊 Epoch 5/10 - Accuracy: 78.6%', delay: 500 },
    { text: '📊 Epoch 10/10 - Accuracy: 91.3%', delay: 500 },
    { text: '🎯 Target accuracy achieved!', delay: 600 },
    { text: '✅ Pipeline deployed successfully!', delay: 500 },
  ];

  const handleSubmit = async (query) => {
    if (isProcessing) return;

    const finalQuery = query || input;
    if (!finalQuery.trim()) return;

    setHistory([{ type: 'input', text: finalQuery }]);
    setInput('');
    setIsProcessing(true);
    setCurrentStep(0);

    for (let i = 0; i < simulatedSteps.length; i++) {
      await new Promise(r => setTimeout(r, simulatedSteps[i].delay));
      setHistory(prev => [...prev, { type: 'output', text: simulatedSteps[i].text }]);
      setCurrentStep(i + 1);
    }

    setIsProcessing(false);
  };

  const handleExampleClick = (query) => {
    setInput(query);
    handleSubmit(query);
  };

  return (
    <div className="relative max-w-4xl mx-auto">
      <div className="absolute inset-0 bg-gradient-to-r from-orange-500/20 via-purple-500/20 to-blue-500/20 blur-3xl" />
      <div className="relative bg-[#0d0d0d] rounded-2xl border border-white/10 overflow-hidden shadow-2xl">
        {/* Terminal Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-white/5 border-b border-white/10">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="ml-4 text-sm text-gray-500">mlops-agent — playground</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-green-400 flex items-center gap-1">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              Live
            </span>
          </div>
        </div>

        {/* Terminal Body */}
        <div className="p-6 font-mono text-sm min-h-[300px] max-h-[400px] overflow-y-auto">
          {history.length === 0 ? (
            <div className="text-gray-500">
              <div className="mb-4">Welcome to MLOps Agent Playground! Try these examples:</div>
              <div className="space-y-2">
                {exampleQueries.map((query, i) => (
                  <button
                    key={i}
                    onClick={() => handleExampleClick(query)}
                    className="block w-full text-left px-3 py-2 rounded-lg bg-white/5 hover:bg-orange-500/20 border border-white/10 hover:border-orange-500/30 transition-all text-gray-400 hover:text-orange-400"
                  >
                    → "{query}"
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              {history.map((item, i) => (
                <div key={i} className={`flex items-start gap-2 ${
                  item.type === 'input' ? 'text-orange-400' :
                  item.text.includes('✓') ? 'text-green-400' :
                  item.text.includes('✅') ? 'text-green-400' :
                  item.text.includes('🎯') ? 'text-yellow-400' :
                  item.text.includes('📊') ? 'text-blue-400' :
                  item.text.includes('⚡') ? 'text-purple-400' :
                  'text-gray-400'
                }`}>
                  {item.type === 'input' ? (
                    <>
                      <span className="text-orange-500">→</span>
                      <span>"{item.text}"</span>
                    </>
                  ) : (
                    <span className="ml-4">{item.text}</span>
                  )}
                </div>
              ))}
              {isProcessing && (
                <div className="flex items-center gap-2 text-gray-500 ml-4">
                  <span className="animate-pulse">●</span>
                  <span>Processing...</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Terminal Input */}
        <div className="px-4 py-3 bg-white/5 border-t border-white/10">
          <div className="flex items-center gap-3">
            <span className="text-orange-500">→</span>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
              placeholder="Type your ML pipeline request..."
              disabled={isProcessing}
              className="flex-1 bg-transparent text-white placeholder-gray-600 focus:outline-none"
            />
            <button
              onClick={() => handleSubmit()}
              disabled={isProcessing || !input.trim()}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                isProcessing || !input.trim()
                  ? 'bg-gray-800 text-gray-500'
                  : 'bg-orange-500 hover:bg-orange-600 text-white'
              }`}
            >
              {isProcessing ? 'Running...' : 'Run'}
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        {isProcessing && (
          <div className="h-1 bg-gray-800">
            <div
              className="h-full bg-gradient-to-r from-orange-500 to-yellow-500 transition-all duration-500"
              style={{ width: `${(currentStep / simulatedSteps.length) * 100}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

const Landing = ({ onGetStarted, onLogin }) => {
  const features = [
    {
      icon: '⚡',
      title: 'Natural Language to Pipeline',
      description: 'Describe your ML workflow in plain English. Our agent handles the rest.'
    },
    {
      icon: '🔄',
      title: 'Self-Improving Agent',
      description: 'Automatically iterates until your accuracy target is met.'
    },
    {
      icon: '🛠️',
      title: 'Full MLOps Stack',
      description: 'Hydra, MLflow, DVC, Docker, and GitHub Actions—all configured automatically.'
    },
    {
      icon: '📊',
      title: 'Real-time Monitoring',
      description: 'Watch your training progress with live metrics and logs.'
    }
  ];

  const stats = [
    { value: 50000, suffix: '+', label: 'Pipelines Deployed', icon: '🚀' },
    { value: 2, suffix: 'M+', label: 'Training Runs', icon: '⚡' },
    { value: 10000, suffix: '+', label: 'ML Engineers', icon: '👨‍💻' },
    { value: 99.9, suffix: '%', label: 'Uptime SLA', icon: '✅' },
  ];

  const companies = ['Google', 'Meta', 'OpenAI', 'Anthropic', 'Netflix', 'Uber'];

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-[#0a0a0a]/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center font-bold text-sm">
              ML
            </div>
            <span className="font-semibold text-lg">MLOps Agent</span>
          </div>
          <div className="flex items-center gap-8">
            <a href="#playground" className="text-sm text-gray-400 hover:text-white transition-colors">Try it</a>
            <a href="#features" className="text-sm text-gray-400 hover:text-white transition-colors">Features</a>
            <a href="#pricing" className="text-sm text-gray-400 hover:text-white transition-colors">Pricing</a>
            <button
              onClick={onLogin}
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Log in
            </button>
            <button
              onClick={onGetStarted}
              className="px-4 py-2 bg-orange-500 hover:bg-orange-600 rounded-lg text-sm font-medium transition-colors"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-16 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-orange-500/10 border border-orange-500/20 text-orange-400 text-sm mb-8">
            <span className="w-2 h-2 rounded-full bg-orange-500 animate-pulse" />
            Now with GPT-4 & Gemini support
          </div>

          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            Deploy ML Pipelines
            <br />
            <span className="bg-gradient-to-r from-orange-400 via-orange-500 to-yellow-500 bg-clip-text text-transparent">
              with One Sentence
            </span>
          </h1>

          <p className="text-xl text-gray-400 mb-10 max-w-2xl mx-auto leading-relaxed">
            The AI-powered MLOps agent that turns your ideas into production-ready
            ML pipelines. No YAML wrestling. No config hell. Just results.
          </p>

          <div className="flex items-center justify-center gap-4 mb-8">
            <button
              onClick={onGetStarted}
              className="px-8 py-4 bg-orange-500 hover:bg-orange-600 rounded-xl text-lg font-semibold transition-all hover:scale-105 hover:shadow-lg hover:shadow-orange-500/25"
            >
              Start Building Free
            </button>
            <a
              href="#playground"
              className="px-8 py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-lg font-semibold transition-all"
            >
              Try Demo ↓
            </a>
          </div>
        </div>
      </section>

      {/* Stats Counter Section */}
      <section className="py-16 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {stats.map((stat, i) => (
              <div key={i} className="text-center p-6 rounded-2xl bg-gradient-to-b from-white/5 to-transparent border border-white/10">
                <div className="text-3xl mb-2">{stat.icon}</div>
                <div className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-orange-400 to-yellow-400 bg-clip-text text-transparent mb-2">
                  <AnimatedCounter end={stat.value} suffix={stat.suffix} />
                </div>
                <div className="text-gray-400 text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Interactive Playground */}
      <section id="playground" className="py-20 px-6 bg-gradient-to-b from-transparent via-orange-500/5 to-transparent">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-sm mb-4">
              🎮 Interactive Demo
            </div>
            <h2 className="text-4xl font-bold mb-4">Try it yourself</h2>
            <p className="text-gray-400 text-lg">No signup required. Just type and see the magic happen.</p>
          </div>

          <InteractiveTerminal />

          <div className="text-center mt-8">
            <p className="text-gray-500 text-sm mb-4">Like what you see?</p>
            <button
              onClick={onGetStarted}
              className="px-6 py-3 bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 rounded-xl font-semibold transition-all hover:scale-105"
            >
              Get Full Access — Free
            </button>
          </div>
        </div>
      </section>

      {/* Logos */}
      <section className="py-16 border-y border-white/5">
        <div className="max-w-6xl mx-auto px-6">
          <p className="text-center text-sm text-gray-500 mb-8">TRUSTED BY ML TEAMS AT</p>
          <div className="flex items-center justify-center gap-12 flex-wrap">
            {companies.map((company) => (
              <span key={company} className="text-xl font-semibold text-gray-600">{company}</span>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Everything you need for MLOps</h2>
            <p className="text-gray-400 text-lg">Stop writing boilerplate. Start shipping models.</p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, i) => (
              <div
                key={i}
                className="p-8 rounded-2xl bg-gradient-to-br from-white/5 to-transparent border border-white/10 hover:border-orange-500/50 transition-all group"
              >
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold mb-2 group-hover:text-orange-400 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-gray-400 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it Works */}
      <section id="how-it-works" className="py-24 px-6 bg-gradient-to-b from-transparent via-purple-500/5 to-transparent">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">How it works</h2>
            <p className="text-gray-400 text-lg">Three steps to production</p>
          </div>

          <div className="space-y-12">
            {[
              { step: '01', title: 'Describe your goal', desc: 'Tell the agent what you want to build in natural language.' },
              { step: '02', title: 'Watch it build', desc: 'The agent creates configs, sets up tracking, and trains your model.' },
              { step: '03', title: 'Deploy & iterate', desc: 'Get a production-ready pipeline with CI/CD included.' }
            ].map((item, i) => (
              <div key={i} className="flex items-start gap-8">
                <div className="text-5xl font-bold text-orange-500/30">{item.step}</div>
                <div>
                  <h3 className="text-2xl font-semibold mb-2">{item.title}</h3>
                  <p className="text-gray-400 text-lg">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-sm mb-4">
              🇮🇳 Special India Pricing
            </div>
            <h2 className="text-4xl font-bold mb-4">Simple, transparent pricing</h2>
            <p className="text-gray-400 text-lg">Start free. Scale as you grow. No hidden fees.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {/* Free Tier */}
            <div className="p-8 rounded-2xl bg-white/5 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-sm text-gray-400 mb-2">For individuals</div>
              <h3 className="text-2xl font-bold mb-1">Free</h3>
              <div className="text-4xl font-bold mb-1">₹0</div>
              <div className="text-gray-500 text-sm mb-6">forever</div>

              <ul className="space-y-3 mb-8">
                {[
                  '5 pipeline runs/month',
                  'Basic MLflow tracking',
                  'Community support',
                  'Public projects only',
                  'Single user'
                ].map((feature, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-gray-300">
                    <span className="text-green-500">✓</span> {feature}
                  </li>
                ))}
              </ul>

              <button
                onClick={onGetStarted}
                className="w-full py-3 bg-white/10 hover:bg-white/20 rounded-xl font-medium transition-all"
              >
                Get Started Free
              </button>
            </div>

            {/* Pro Tier */}
            <div className="p-8 rounded-2xl bg-gradient-to-b from-orange-500/20 to-orange-500/5 border-2 border-orange-500/50 relative">
              <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-orange-500 rounded-full text-xs font-semibold">
                MOST POPULAR
              </div>
              <div className="text-sm text-orange-400 mb-2">For professionals</div>
              <h3 className="text-2xl font-bold mb-1">Pro</h3>
              <div className="flex items-baseline gap-1 mb-1">
                <span className="text-4xl font-bold">₹999</span>
                <span className="text-gray-400">/month</span>
              </div>
              <div className="text-gray-500 text-sm mb-6">billed annually (₹11,988/yr)</div>

              <ul className="space-y-3 mb-8">
                {[
                  'Unlimited pipeline runs',
                  'Advanced MLflow + DVC',
                  'Priority support',
                  'Private projects',
                  'GPU training support',
                  'Custom Docker images',
                  'GitHub Actions CI/CD'
                ].map((feature, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-gray-300">
                    <span className="text-orange-500">✓</span> {feature}
                  </li>
                ))}
              </ul>

              <button
                onClick={onGetStarted}
                className="w-full py-3 bg-orange-500 hover:bg-orange-600 rounded-xl font-semibold transition-all"
              >
                Start 14-day Free Trial
              </button>
            </div>

            {/* Team Tier */}
            <div className="p-8 rounded-2xl bg-white/5 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-sm text-gray-400 mb-2">For teams</div>
              <h3 className="text-2xl font-bold mb-1">Team</h3>
              <div className="flex items-baseline gap-1 mb-1">
                <span className="text-4xl font-bold">₹2,499</span>
                <span className="text-gray-400">/user/mo</span>
              </div>
              <div className="text-gray-500 text-sm mb-6">min 3 users, billed annually</div>

              <ul className="space-y-3 mb-8">
                {[
                  'Everything in Pro',
                  'Team collaboration',
                  'Shared experiments',
                  'Admin dashboard',
                  'SSO & SAML',
                  'Dedicated support',
                  'SLA guarantee',
                  'On-premise option'
                ].map((feature, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-gray-300">
                    <span className="text-blue-500">✓</span> {feature}
                  </li>
                ))}
              </ul>

              <button className="w-full py-3 bg-white/10 hover:bg-white/20 rounded-xl font-medium transition-all">
                Contact Sales
              </button>
            </div>
          </div>

          {/* Enterprise callout */}
          <div className="mt-12 p-6 rounded-2xl bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20 max-w-3xl mx-auto text-center">
            <h4 className="text-lg font-semibold mb-2">🏢 Enterprise</h4>
            <p className="text-gray-400 mb-4">
              Need custom deployment, dedicated infrastructure, or volume pricing?
              We offer special rates for Indian startups and educational institutions.
            </p>
            <button className="text-purple-400 hover:text-purple-300 font-medium transition-colors">
              Talk to our team →
            </button>
          </div>

          {/* Trust badges */}
          <div className="flex items-center justify-center gap-8 mt-12 text-sm text-gray-500">
            <span className="flex items-center gap-2">🔒 SOC 2 Compliant</span>
            <span className="flex items-center gap-2">💳 UPI & Cards Accepted</span>
            <span className="flex items-center gap-2">📄 GST Invoice Available</span>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Ready to automate your MLOps?
          </h2>
          <p className="text-xl text-gray-400 mb-10">
            Join thousands of ML engineers shipping faster with MLOps Agent.
          </p>
          <button
            onClick={onGetStarted}
            className="px-8 py-4 bg-orange-500 hover:bg-orange-600 rounded-xl text-lg font-semibold transition-all hover:scale-105"
          >
            Get Started — It's Free
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-white/5">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center font-bold text-xs">
              ML
            </div>
            <span className="text-sm text-gray-500">© 2024 MLOps Agent</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-gray-500">
            <a href="#" className="hover:text-white transition-colors">Twitter</a>
            <a href="#" className="hover:text-white transition-colors">GitHub</a>
            <a href="#" className="hover:text-white transition-colors">Discord</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
