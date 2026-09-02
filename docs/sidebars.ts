import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  userSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      link: {type: 'doc', id: 'getting-started/overview'},
      items: [
        {
          type: 'category',
          label: 'Installation',
          collapsed: false,
          link: {type: 'doc', id: 'getting-started/installation/index'},
          items: [
            'getting-started/installation/prerequisites',
            'getting-started/installation/simulator',
            'getting-started/installation/vllm',
          ],
        },
        'getting-started/quickstart',
        'getting-started/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      link: {type: 'doc', id: 'examples/index'},
      items: [
        'examples/cluster-config-explained',
        {
          type: 'category',
          label: 'Parallelism',
          items: [
            'examples/parallelism/tensor-parallel',
            'examples/parallelism/pipeline-parallel',
            'examples/parallelism/expert-parallel',
            'examples/parallelism/dp-ep-moe',
          ],
        },
        {
          type: 'category',
          label: 'Disaggregated serving',
          items: [
            'examples/disaggregated/multi-instance',
            'examples/disaggregated/prefill-decode-split',
            'examples/disaggregated/pim-attention-offload',
          ],
        },
        {
          type: 'category',
          label: 'Memory tiers',
          items: [
            'examples/memory-tiers/prefix-caching',
            'examples/memory-tiers/cxl-memory',
            'examples/memory-tiers/fp8-kv-cache',
          ],
        },
        {
          type: 'category',
          label: 'Model families',
          items: [
            'examples/model-families/hybrid-linear-attention',
            'examples/model-families/sparse-attention',
          ],
        },
        {
          type: 'category',
          label: 'Advanced',
          items: [
            'examples/advanced/power-modeling',
            'examples/advanced/speculative-decoding',
            'examples/advanced/sub-batch-interleaving',
          ],
        },
      ],
    },
    'validation',
    {
      type: 'category',
      label: 'Simulator',
      items: [
        'simulator/architecture',
        'simulator/request-lifecycle',
        {
          type: 'category',
          label: 'Scheduling',
          items: [
            'simulator/scheduling/continuous-batching',
            'simulator/scheduling/prefix-caching',
            'simulator/scheduling/kv-cache-and-memory',
          ],
        },
        'simulator/trace-generation',
        'simulator/parallelism-mechanics',
        'simulator/moe-expert-routing',
        {
          type: 'category',
          label: 'Specialized',
          items: [
            'simulator/specialized/pim-offload',
            'simulator/specialized/power-model',
          ],
        },
        'simulator/reading-output',
      ],
    },
    {
      type: 'category',
      label: 'Profiler',
      link: {type: 'doc', id: 'profiler/overview'},
      items: [
        'profiler/running',
        'profiler/output-bundle',
        'profiler/skew-alpha-fit',
        'profiler/adding-hardware',
        'profiler/adding-model-architecture',
      ],
    },
    {
      type: 'category',
      label: 'Workloads',
      link: {type: 'doc', id: 'workloads/overview'},
      items: [
        'workloads/jsonl-format',
        'workloads/sharegpt-generators',
        'workloads/agentic-sessions',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      link: {
        type: 'generated-index',
        description: 'CLI flags, config schemas, and file formats.',
      },
      items: [
        'reference/cli-flags',
        'reference/cluster-config',
        'reference/model-config',
        'reference/pim-config',
        'reference/trace-format',
        'reference/bench-cli',
      ],
    },
    'artifact-evaluation',
  ],
  contributorSidebar: [
    'contributor/welcome',
    'contributor/intro',
    'contributor/codebase-tour',
    'contributor/conventions',
    'contributor/validating-changes',
    'contributor/pr-workflow',
  ],
};

export default sidebars;
