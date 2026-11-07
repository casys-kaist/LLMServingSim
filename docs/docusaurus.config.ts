import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'LLMServingSim',
  tagline:
    'Toward unified simulation of heterogeneous and disaggregated LLM serving infrastructure',
  favicon: 'img/favicon.png',

  future: {
    v4: true,
  },

  url: 'https://llmservingsim.ai',
  baseUrl: '/',

  organizationName: 'casys-kaist',
  projectName: 'LLMServingSim',
  trailingSlash: false,

  headTags: [
    {
      tagName: 'link',
      attributes: {rel: 'preconnect', href: 'https://fonts.googleapis.com'},
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'preconnect',
        href: 'https://fonts.gstatic.com',
        crossorigin: 'anonymous',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'apple-touch-icon',
        sizes: '180x180',
        href: '/img/apple-touch-icon.png',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'icon',
        type: 'image/png',
        sizes: '256x256',
        href: '/img/favicon-256.png',
      },
    },
  ],

  stylesheets: [
    'https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800&family=Geist+Mono:wght@400;500;700&display=swap',
  ],

  onBrokenLinks: 'throw',

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      {
        // Build a static lunr-like index at build time and ship it to
        // the client. Swap out for Algolia DocSearch once approved.
        hashed: true,
        indexBlog: false,
        indexPages: true,
        docsRouteBasePath: '/docs',
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/casys-kaist/LLMServingSim/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.png',
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    // Force same Mermaid palette in light + dark so diagrams stay readable
    // either way. Custom 'base' theme tinted with our indigo brand.
    mermaid: {
      theme: {light: 'base', dark: 'base'},
      options: {
        themeVariables: {
          // Node defaults
          primaryColor: '#eef2ff',          // light indigo node fill
          primaryTextColor: '#1e1b4b',       // dark indigo text
          primaryBorderColor: '#4f46e5',    // indigo border
          lineColor: '#4f46e5',              // edges / arrows
          // Subgraph backgrounds
          secondaryColor: '#f5f3ff',         // soft violet
          tertiaryColor: '#fef3c7',          // optional accent
          // Notes / labels
          noteBkgColor: '#fef9c3',
          noteTextColor: '#422006',
          noteBorderColor: '#a16207',
          // Sequence-diagram actors
          actorBkg: '#eef2ff',
          actorBorder: '#4f46e5',
          actorTextColor: '#1e1b4b',
          actorLineColor: '#94a3b8',
          signalColor: '#4f46e5',
          signalTextColor: '#1e1b4b',
          // State diagrams
          labelBoxBkgColor: '#eef2ff',
          labelBoxBorderColor: '#4f46e5',
          labelTextColor: '#1e1b4b',
        },
      },
    },
    navbar: {
      title: 'LLMServingSim',
      logo: {
        alt: 'LLMServingSim',
        src: 'img/llmservingsim_compact_primary_transparent.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'userSidebar',
          position: 'left',
          label: 'For Users',
        },
        {
          type: 'docSidebar',
          sidebarId: 'contributorSidebar',
          position: 'left',
          label: 'For Contributors',
        },
        {
          to: '/changelog',
          label: 'Changelog',
          position: 'right',
        },
        {
          to: '/contact',
          label: 'Contact',
          position: 'right',
        },
        {
          href: 'https://github.com/casys-kaist/LLMServingSim',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Getting Started', to: '/docs/getting-started/overview'},
            {label: 'Simulator', to: '/docs/simulator/architecture'},
            {label: 'Profiler', to: '/docs/profiler/overview'},
            {label: 'Reference', to: '/docs/reference/cli-flags'},
          ],
        },
        {
          title: 'Project',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/casys-kaist/LLMServingSim',
            },
            {
              label: 'Issues',
              href: 'https://github.com/casys-kaist/LLMServingSim/issues',
            },
            {
              label: 'Changelog',
              to: '/changelog',
            },
            {
              label: 'Contact',
              to: '/contact',
            },
          ],
        },
        {
          title: 'Group',
          items: [
            {
              label: 'CASYS @ KAIST',
              href: 'https://casys.kaist.ac.kr/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} CASYS @ KAIST. Licensed under the MIT License.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'json', 'yaml'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
