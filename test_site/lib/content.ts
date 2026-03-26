// ─── Current product catalogue ────────────────────────────────────────────────

export type StockStatus = 'in_stock' | 'low_stock' | 'out_of_stock';

export interface Product {
  slug: string;
  name: string;
  price: number;
  currency: string;
  stock: StockStatus;
  tagline: string;
  description: string;
  launched: number; // year
}

// Ground truth: Blue Device = £84 | Premium Device = low_stock
export const CURRENT_PRODUCTS: Product[] = [
  {
    slug: 'blue-device',
    name: 'Blue Device',
    price: 84,
    currency: '£',
    stock: 'in_stock',
    tagline: 'Clear performance, everyday reliability.',
    description:
      'The Blue Device is our versatile everyday companion — lightweight, durable, and designed for all-day use. Ideal for home and professional environments alike.',
    launched: 2019,
  },
  {
    slug: 'premium-device',
    name: 'Premium Device',
    price: 149,
    currency: '£',
    stock: 'low_stock',
    tagline: 'Elevated experience, limited availability.',
    description:
      "Northstar's flagship model packs advanced sensors and premium materials into a sleek form factor. High demand means stock is limited — order soon.",
    launched: 2021,
  },
  {
    slug: 'essential-device',
    name: 'Essential Device',
    price: 59,
    currency: '£',
    stock: 'in_stock',
    tagline: 'Everything you need, nothing you don\'t.',
    description:
      'A straightforward, dependable device for users who value simplicity. The Essential Device covers the core functions without compromise.',
    launched: 2022,
  },
];

// ─── History & timeline ───────────────────────────────────────────────────────

export interface HistoryEvent {
  year: number;
  label: string;
  detail: string;
}

// Ground truth: 2018 = first portable device | 2021 = accessories expansion
export const HISTORY_EVENTS: HistoryEvent[] = [
  {
    year: 2015,
    label: 'Northstar Devices founded',
    detail:
      'Three engineers from Edinburgh set out to build precision measurement tools that were actually pleasant to use.',
  },
  {
    year: 2017,
    label: 'First commercial product released',
    detail:
      "The original Northstar unit shipped to early adopters across the UK, establishing the brand's reputation for reliability.",
  },
  {
    year: 2018,
    label: 'First portable device launched',
    detail:
      'Northstar introduced its first battery-powered portable model, opening up field use cases and a wider consumer audience.',
  },
  {
    year: 2020,
    label: 'Series A funding secured',
    detail:
      'A £4 million investment round enabled significant R&D expansion and the first international sales partnerships.',
  },
  {
    year: 2021,
    label: 'Expanded into accessories',
    detail:
      'Northstar broadened its catalogue with a dedicated accessories range: cases, cables, mounts, and calibration tools.',
  },
  {
    year: 2022,
    label: 'Essential Device launched',
    detail:
      'Responding to demand for an entry-level option, Northstar released the Essential Device at a lower price point.',
  },
  {
    year: 2023,
    label: '2023 product line refreshed',
    detail:
      'Annual refresh brought improved internals across the range and a revised accessory ecosystem.',
  },
];

export interface ProductLaunch {
  name: string;
  year: number;
  description: string;
}

export const PRODUCT_LAUNCHES: ProductLaunch[] = [
  {
    name: 'Original Northstar Unit',
    year: 2017,
    description:
      'The device that started it all. Bench-top form factor, USB connectivity, and a no-nonsense display.',
  },
  {
    name: 'Northstar Portable (first portable device)',
    year: 2018,
    description:
      "Cordless, compact, and field-ready. The Portable brought Northstar's accuracy to locations without mains power.",
  },
  {
    name: 'Blue Device',
    year: 2019,
    description:
      "Northstar's most successful product. Balanced specs, wide compatibility, and a distinctive colour coding system.",
  },
  {
    name: 'Premium Device',
    year: 2021,
    description:
      'Premium materials and extended sensor range. Launched alongside the accessories range.',
  },
  {
    name: 'Essential Device',
    year: 2022,
    description: 'Entry-level simplicity without sacrificing the core Northstar experience.',
  },
];

// ─── Support policies ─────────────────────────────────────────────────────────

export const SHIPPING_POLICY = {
  international: true,           // Ground truth: Yes
  domestic_free_threshold_gbp: 40,
  estimated_uk_days: '2–4',
  estimated_eu_days: '5–10',
  estimated_row_days: '7–21',
  carrier: 'Royal Mail & DPD',
  tracking: true,
  returns_days: 30,
};

export const WARRANTY_POLICY = {
  duration_years: 2,
  covers_accidental_damage: false, // Ground truth: No
  covers_manufacturing_defects: true,
  covers_wear_and_tear: false,
  repair_or_replace: true,
  claim_process: 'Contact support with proof of purchase and a description of the fault.',
  faqs: [
    {
      question: 'How long is the warranty?',
      answer:
        'All Northstar Devices products come with a standard 2-year limited warranty from the date of purchase.',
    },
    {
      question: 'What does the warranty cover?',
      answer:
        'The warranty covers manufacturing defects and component failures under normal use conditions.',
    },
    {
      question: 'Does the warranty cover accidental damage?',
      answer:
        'No, accidental damage is not covered. This includes drops, liquid ingress, and physical impact. We recommend a protective case for field use.',
    },
    {
      question: 'Does the warranty cover wear and tear?',
      answer:
        'No, normal wear and tear — such as worn buttons or faded markings — is not covered under warranty.',
    },
    {
      question: 'How do I make a warranty claim?',
      answer:
        'Contact our support team with your proof of purchase and a description of the fault. We will arrange collection and assessment within 5 working days.',
    },
  ],
};

// ─── Archive — 2023 product line ─────────────────────────────────────────────

export interface ArchivedProduct {
  name: string;
  price: number;
  currency: string;
  year: number;
  notes?: string;
}

// Ground truth: Blue Device 2023 price = £79
export const ARCHIVE_2023: ArchivedProduct[] = [
  {
    name: 'Blue Device',
    price: 79,
    currency: '£',
    year: 2023,
    notes: 'Pre-refresh pricing. Current price reflects updated hardware.',
  },
  {
    name: 'Premium Device',
    price: 139,
    currency: '£',
    year: 2023,
    notes: 'Pre-refresh pricing.',
  },
  {
    name: 'Essential Device',
    price: 55,
    currency: '£',
    year: 2023,
    notes: 'Pre-refresh pricing.',
  },
];

// ─── Archive — retired devices ────────────────────────────────────────────────

export interface RetiredDevice {
  name: string;
  launched: number;
  retired_year: number;
  reason: string;
}

// Ground truth: Classic Device retired before 2024
export const RETIRED_DEVICES: RetiredDevice[] = [
  {
    name: 'Classic Device',
    launched: 2017,
    retired_year: 2023,
    reason:
      'Discontinued at end of 2023 as the Blue Device superseded it across all use cases. Spare parts available until 2026.',
  },
  {
    name: 'Northstar Mini',
    launched: 2020,
    retired_year: 2022,
    reason:
      'Retired following the launch of the Essential Device, which offered comparable features at a more competitive price.',
  },
];
