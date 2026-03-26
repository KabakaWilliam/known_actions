import Link from 'next/link';
import { SHIPPING_POLICY } from '@/lib/content';

export const metadata = { title: 'Shipping — Northstar Devices' };

export default function ShippingPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">Home</Link>
        <span className="mx-2">›</span>
        <Link href="/support" className="hover:text-zinc-600">Support</Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">Shipping</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Shipping Policy
      </h1>
      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        Information about delivery options, timescales, and our returns process.
      </p>

      <div className="rounded-xl border border-zinc-200 bg-white divide-y divide-zinc-100">
        <div className="px-6 py-5 flex items-start justify-between gap-4">
          <div>
            <p className="font-medium text-zinc-900 text-sm">International shipping</p>
            <p className="text-sm text-zinc-500 mt-0.5">
              We ship to customers worldwide. Duties and taxes may apply depending on
              your country.
            </p>
          </div>
          <span className="shrink-0 text-sm font-semibold text-green-700 bg-green-50 px-3 py-1 rounded">
            {SHIPPING_POLICY.international ? 'Yes' : 'No'}
          </span>
        </div>

        <div className="px-6 py-5 flex items-start justify-between gap-4">
          <div>
            <p className="font-medium text-zinc-900 text-sm">Free domestic shipping threshold</p>
            <p className="text-sm text-zinc-500 mt-0.5">Orders over this value qualify for free UK delivery.</p>
          </div>
          <span className="shrink-0 text-sm font-semibold text-zinc-700">
            £{SHIPPING_POLICY.domestic_free_threshold_gbp}
          </span>
        </div>

        <div className="px-6 py-5 flex items-start justify-between gap-4">
          <div>
            <p className="font-medium text-zinc-900 text-sm">UK delivery estimate</p>
            <p className="text-sm text-zinc-500 mt-0.5">Typical working day range from dispatch.</p>
          </div>
          <span className="shrink-0 text-sm font-semibold text-zinc-700">
            {SHIPPING_POLICY.estimated_uk_days} days
          </span>
        </div>

        <div className="px-6 py-5 flex items-start justify-between gap-4">
          <div>
            <p className="font-medium text-zinc-900 text-sm">Europe delivery estimate</p>
            <p className="text-sm text-zinc-500 mt-0.5">Typical working day range for EU destinations.</p>
          </div>
          <span className="shrink-0 text-sm font-semibold text-zinc-700">
            {SHIPPING_POLICY.estimated_eu_days} days
          </span>
        </div>

        <div className="px-6 py-5 flex items-start justify-between gap-4">
          <div>
            <p className="font-medium text-zinc-900 text-sm">Rest of world delivery estimate</p>
            <p className="text-sm text-zinc-500 mt-0.5">Approximate delivery window for international orders.</p>
          </div>
          <span className="shrink-0 text-sm font-semibold text-zinc-700">
            {SHIPPING_POLICY.estimated_row_days} days
          </span>
        </div>

        <div className="px-6 py-5 flex items-start justify-between gap-4">
          <div>
            <p className="font-medium text-zinc-900 text-sm">Order tracking</p>
            <p className="text-sm text-zinc-500 mt-0.5">Tracking numbers are emailed at dispatch.</p>
          </div>
          <span className="shrink-0 text-sm font-semibold text-zinc-700">
            {SHIPPING_POLICY.tracking ? 'Included' : 'Not included'}
          </span>
        </div>

        <div className="px-6 py-5 flex items-start justify-between gap-4">
          <div>
            <p className="font-medium text-zinc-900 text-sm">Returns window</p>
            <p className="text-sm text-zinc-500 mt-0.5">
              Items must be returned in original packaging, unused. Contact support to
              initiate a return.
            </p>
          </div>
          <span className="shrink-0 text-sm font-semibold text-zinc-700">
            {SHIPPING_POLICY.returns_days} days
          </span>
        </div>

        <div className="px-6 py-5">
          <p className="font-medium text-zinc-900 text-sm">Carrier</p>
          <p className="text-sm text-zinc-500 mt-0.5">{SHIPPING_POLICY.carrier}</p>
        </div>
      </div>

      <p className="mt-6 text-sm text-zinc-500">
        For warranty queries, visit{' '}
        <Link href="/support/warranty" className="text-blue-600 hover:underline">
          Warranty
        </Link>
        .
      </p>
    </div>
  );
}
