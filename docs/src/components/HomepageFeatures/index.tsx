import type {ReactNode, ComponentType, SVGProps} from 'react';
import clsx from 'clsx';
import {Gauge, Database, Network, Cpu, Boxes, Activity} from 'lucide-react';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Icon: ComponentType<SVGProps<SVGSVGElement> & {size?: number; strokeWidth?: number}>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Measured execution profiles',
    Icon: Gauge,
    description: (
      <>
        Compute costs come from layer-level CUDA profiles measured on vLLM
        itself on the target GPU, not from fitting end-to-end benchmark
        results.
      </>
    ),
  },
  {
    title: 'Memory hierarchy and KV cache',
    Icon: Database,
    description: (
      <>
        vLLM's block pool, prefix caching and preemption, with KV blocks
        placed across GPU, CPU and CXL memory tiers and sizes derived from
        each checkpoint's configuration.
      </>
    ),
  },
  {
    title: 'ASTRA-Sim networking',
    Icon: Network,
    description: (
      <>
        TP, PP, EP and DP collectives on multidimensional ASTRA-Sim
        topologies, with link parameters calibrated from measured NCCL
        primitives where available.
      </>
    ),
  },
  {
    title: 'Heterogeneous and disaggregated',
    Icon: Cpu,
    description: (
      <>
        Multi-instance clusters with prefill/decode disaggregation, CPU and
        CXL KV-cache tiers, and PIM decode-attention offload.
      </>
    ),
  },
  {
    title: 'Modern model architectures',
    Icon: Boxes,
    description: (
      <>
        Catalogs for dense, MoE, MLA, sparse-attention and hybrid
        linear-attention families, plus MTP speculative decoding.
      </>
    ),
  },
  {
    title: 'Real and agentic workloads',
    Icon: Activity,
    description: (
      <>
        Replay ShareGPT-style request traces or closed-loop agentic
        sessions whose LLM calls wait on tool calls.
      </>
    ),
  },
];

function Feature({title, Icon, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4', styles.featureCol)}>
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
    <section className={styles.features} aria-labelledby="features-heading">
      <div className="container">
        <p className={styles.eyebrow}>Inside LLMServingSim 2.0</p>
        <h2 id="features-heading" className={styles.title}>
          Full-stack simulation, from kernels to cluster-scale serving
        </h2>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
