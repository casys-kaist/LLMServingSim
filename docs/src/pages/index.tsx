import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import {ArrowRight, Check} from 'lucide-react';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import GithubStats from '@site/src/components/GithubStats';
import ValidationHighlights from '@site/src/components/ValidationHighlights';
import Recognition from '@site/src/components/Recognition';

import styles from './index.module.css';

const LLMSERVINGSIM_REPO = 'https://github.com/casys-kaist/LLMServingSim';
const SIMTHESIZER_SITE = 'https://simthesizer.ai/';
const SIMTHESIZER_REPO = 'https://github.com/casys-kaist/Simthesizer';

// Key points on each half of the project split. The 1.2% bound is the
// largest "worst of 15" value in the headline table of docs/validation.md;
// keep the two in sync when the committed bench examples are refreshed.
const LLMSERVINGSIM_POINTS: ReactNode[] = [
  <>
    Validated against real vLLM with a max error of <strong>1.2%</strong>
  </>,
  <>
    Detailed <strong>memory hierarchy</strong> and <strong>KV-cache</strong>{' '}
    models
  </>,
  <>
    Network simulation based on <strong>ASTRA-Sim</strong>
  </>,
  <>
    Heterogeneous systems, including <strong>CXL</strong> and{' '}
    <strong>PIM</strong>
  </>,
];

const SIMTHESIZER_POINTS: ReactNode[] = [
  <>
    <strong>Coding-agent-friendly</strong> architecture
  </>,
  <>
    <strong>Modular</strong> serving components
  </>,
  <>
    <strong>Rapid prototyping</strong> of new models, policies and mechanisms
  </>,
];

function PointList({points}: {points: ReactNode[]}) {
  return (
    <ul className={styles.points}>
      {points.map((point, idx) => (
        <li key={idx} className={styles.point}>
          <Check
            size={18}
            strokeWidth={2.25}
            className={styles.pointIcon}
            aria-hidden="true"
          />
          <span>{point}</span>
        </li>
      ))}
    </ul>
  );
}

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const wordmark = useBaseUrl(
    '/img/llmservingsim_full_primary_transparent.png',
  );
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <h1 className={styles.heroTitle}>
          <img
            src={wordmark}
            alt={siteConfig.title}
            className={styles.heroWordmark}
          />
        </h1>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <GithubStats
          action={
            <Link
              className={styles.heroAction}
              to="/docs/getting-started/overview">
              Get Started
              <ArrowRight size={16} strokeWidth={2.25} aria-hidden="true" />
            </Link>
          }
        />
      </div>
    </header>
  );
}

function ProjectSplit() {
  return (
    <section
      className={styles.split}
      aria-label="LLMServingSim and Simthesizer">
      <div className={clsx(styles.panel, styles.panelPrimary)}>
        <div className={styles.panelInner}>
          <span className={styles.panelEyebrow}>
            High-fidelity · Validated · Heterogeneous systems
          </span>
          <h2 className={styles.panelTitle}>LLMServingSim 2.0</h2>
          <p className={styles.panelLead}>
            A full-stack LLM serving simulator grounded in real GPU
            measurements, scaling from a single GPU to heterogeneous,
            disaggregated clusters.
          </p>
          <PointList points={LLMSERVINGSIM_POINTS} />
          <div className={styles.panelFooter}>
            <div className={styles.panelUse}>
              <span className={styles.panelUseLabel}>Best for</span>
              <p className={styles.panelUseText}>
                Trustworthy numbers for detailed performance and bottleneck
                analysis of your LLM serving system.
              </p>
            </div>
            <div className={styles.panelButtons}>
              <Link
                className={clsx('button button--lg', styles.btnSolid)}
                to="/docs/getting-started/overview">
                Explore LLMServingSim
              </Link>
              <Link
                className={clsx('button button--lg', styles.btnGhost)}
                href={LLMSERVINGSIM_REPO}
                aria-label="LLMServingSim on GitHub">
                GitHub
              </Link>
            </div>
          </div>
        </div>
      </div>
      <div className={clsx(styles.panel, styles.panelSibling)}>
        <div className={styles.panelInner}>
          <span className={styles.panelEyebrow}>
            Agent-native · Modular · Rapid prototyping
          </span>
          <h2 className={styles.panelTitle}>Simthesizer</h2>
          <p className={styles.panelLead}>
            An LLM serving simulator structured so that coding agents can
            quickly add new models, hardware and serving features to one
            shared, extensible codebase.
          </p>
          <PointList points={SIMTHESIZER_POINTS} />
          <div className={styles.panelFooter}>
            <div className={styles.panelUse}>
              <span className={styles.panelUseLabel}>Best for</span>
              <p className={styles.panelUseText}>
                Quickly exploring your own ideas or requirements and
                simulating them right away.
              </p>
            </div>
            <div className={styles.panelButtons}>
              <Link
                className={clsx('button button--lg', styles.btnSolid)}
                href={SIMTHESIZER_SITE}>
                Explore Simthesizer
              </Link>
              <Link
                className={clsx('button button--lg', styles.btnGhost)}
                href={SIMTHESIZER_REPO}
                aria-label="Simthesizer on GitHub">
                GitHub
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Toward unified simulation of heterogeneous and disaggregated LLM serving infrastructure">
      <HomepageHeader />
      <ProjectSplit />
      <main>
        <ValidationHighlights />
        <HomepageFeatures />
        <Recognition />
      </main>
    </Layout>
  );
}
