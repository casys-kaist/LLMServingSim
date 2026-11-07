import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import GithubStats from '@site/src/components/GithubStats';
import Recognition from '@site/src/components/Recognition';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const wordmark = useBaseUrl(
    '/img/llmservingsim_full_primary_transparent.png',
  );
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <img
          src={wordmark}
          alt={siteConfig.title}
          className={styles.heroWordmark}
        />
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <GithubStats />
      </div>
    </header>
  );
}

function NewsBanner() {
  return (
    <Link
      className={styles.newsBanner}
      href="https://github.com/casys-kaist/Simthesizer">
      <span className={styles.newsBadge}>News!</span>
      <span className={styles.newsText}>
        We released a new project, <strong>Simthesizer</strong>: a
        coding-agent-native LLM serving simulator.
      </span>
      <span className={styles.newsCta}>Check it out →</span>
    </Link>
  );
}

function ProjectSplit() {
  return (
    <section className={styles.split} aria-label="Choose a simulator">
      <div className={clsx(styles.panel, styles.panelManual)}>
        <div className={styles.panelInner}>
          <span className={styles.panelEyebrow}>Manual · Human-verified · Accurate</span>
          <h2 className={styles.panelTitle}>LLMServingSim 2.0</h2>
          <p className={styles.panelLead}>
            Every model catalog, profiling contract and hardware fit is
            written and reviewed by experts, then validated end to end
            against real vLLM benchmark runs.
          </p>
          <div className={styles.panelChoose}>
            <span className={styles.panelChooseLabel}>Choose LLMServingSim when</span>
            <p className={styles.panelChooseText}>
              you need trustworthy numbers for detailed performance and
              bottleneck analysis of your LLM serving system.
            </p>
          </div>
          <div className={styles.panelButtons}>
            <Link
              className={clsx('button button--lg', styles.btnManual)}
              to="/docs/getting-started/overview">
              Get Started
            </Link>
            <Link
              className={clsx('button button--lg', styles.btnGhost)}
              href="https://github.com/casys-kaist/LLMServingSim">
              GitHub
            </Link>
          </div>
        </div>
      </div>
      <div className={clsx(styles.panel, styles.panelAuto)}>
        <div className={styles.panelInner}>
          <span className={styles.panelEyebrow}>Coding-agent native · Automatic · Fast</span>
          <h2 className={styles.panelTitle}>Simthesizer</h2>
          <p className={styles.panelLead}>
            An LLM serving simulator structured for coding agents to
            extend: an agent can add new models, hardware and serving
            features quickly.
          </p>
          <div className={styles.panelChoose}>
            <span className={styles.panelChooseLabel}>Choose Simthesizer when</span>
            <p className={styles.panelChooseText}>
              you want to quickly explore your own ideas or requirements
              and simulate them right away.
            </p>
          </div>
          <div className={styles.panelButtons}>
            <Link
              className={clsx('button button--lg', styles.btnAuto)}
              href="https://simthesizer.ai/">
              Get Started
            </Link>
            <Link
              className={clsx('button button--lg', styles.btnGhost)}
              href="https://github.com/casys-kaist/Simthesizer">
              GitHub
            </Link>
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
      <NewsBanner />
      <ProjectSplit />
      <main>
        <HomepageFeatures />
        <Recognition />
      </main>
    </Layout>
  );
}
