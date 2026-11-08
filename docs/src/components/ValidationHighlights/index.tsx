import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

type Deployment = {
  model: string;
  // Parallelism and GPU, shown as one line under the model.
  setup: string;
  // Largest absolute error across the fifteen TTFT / TPOT / latency
  // statistics (mean, median, P90, P95, P99).
  worst: string;
};

// The "worst of 15" column of the headline table in docs/validation.md,
// which reproduces the committed bench/examples/<hardware>/<model>/
// validation/summary.txt files. Refresh both together when those examples
// change; the page keeps the per-metric errors and vLLM versions.
const DEPLOYMENTS: Deployment[] = [
  {model: 'Llama-3.1-8B', setup: 'TP1 · RTX PRO 6000', worst: '0.7%'},
  {model: 'Qwen3-32B', setup: 'TP2 · RTX PRO 6000', worst: '0.5%'},
  {model: 'Qwen3-30B-A3B', setup: 'DP2 × EP2 MoE · RTX PRO 6000', worst: '1.2%'},
  {model: 'Llama-3.1-8B', setup: 'TP1 · RTX 4090', worst: '1.2%'},
];

export default function ValidationHighlights(): ReactNode {
  return (
    <section className={styles.validation} aria-labelledby="validation-heading">
      <div className="container">
        <p className={styles.eyebrow}>LLMServingSim 2.0 validation</p>
        <h2 id="validation-heading" className={styles.title}>
          High accuracy, with a max error of{' '}
          <span className={styles.titleAccent}>1.2%</span> against real vLLM
        </h2>

        <ul className={styles.grid}>
          {DEPLOYMENTS.map((d) => (
            <li key={`${d.model}-${d.setup}`} className={styles.card}>
              <div className={styles.worstRow}>
                <span className={styles.worst}>{d.worst}</span>
                <span className={styles.worstLabel}>max error</span>
              </div>
              <div className={styles.model}>{d.model}</div>
              <div className={styles.meta}>{d.setup}</div>
            </li>
          ))}
        </ul>

        <p className={styles.scope}>
          Largest TTFT, TPOT or latency error for each deployment. Other
          configurations need their own validation.{' '}
          <Link to="/docs/validation">View validation scope and results →</Link>
        </p>
      </div>
    </section>
  );
}
