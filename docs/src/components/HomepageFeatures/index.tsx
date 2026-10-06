import type {ReactNode, ComponentType, SVGProps} from 'react';
import clsx from 'clsx';
import {Gauge, Cpu, Network, Activity} from 'lucide-react';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Icon: ComponentType<SVGProps<SVGSVGElement> & {size?: number; strokeWidth?: number}>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Measured execution costs',
    Icon: Gauge,
    description: (
      <>
        Powered by a vLLM-based layerwise profiler, with reproducible
        TTFT, TPOT, and latency comparisons against recorded Llama
        and Qwen workloads.
      </>
    ),
  },
  {
    title: 'Heterogeneous HW',
    Icon: Cpu,
    description: (
      <>
        Mix GPU, CPU, CXL, and PIM tiers. Drop in any hardware target
        via the per-hardware CSV bundle format.
      </>
    ),
  },
  {
    title: 'Disaggregated serving',
    Icon: Network,
    description: (
      <>
        First-class TP / PP / EP / DP+EP support across multiple instances,
        with wave-synchronized expert collectives on multidimensional
        ASTRA-Sim topologies.
      </>
    ),
  },
  {
    title: 'Real workloads',
    Icon: Activity,
    description: (
      <>
        Drive the simulator with ShareGPT-style traces or closed-loop
        agentic sessions (sub-requests + tool calls) for SWE-bench-style
        scenarios.
      </>
    ),
  },
];

function Feature({title, Icon, description}: FeatureItem) {
  return (
    <div className={clsx('col col--3', styles.featureCol)}>
      <div className={styles.featureCard}>
        <div className={styles.featureIcon} aria-hidden="true">
          <Icon size={28} strokeWidth={1.75} />
        </div>
        <Heading as="h3" className={styles.featureTitle}>
          {title}
        </Heading>
        <p className={styles.featureDescription}>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
