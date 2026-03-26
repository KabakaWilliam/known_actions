import Link from 'next/link';
import { ARCHIVE_2023 } from '@/lib/content';

export const metadata = { title: '2023 Product Line — Northstar Devices Archive' };

export default function Archive2023Page() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">Home</Link>
        <span className="mx-2">›</span>
        <Link href="/archive" className="hover:text-zinc-600">Archive</Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">2023 Line</span>
      </div>

      <div className="flex items-center gap-3 mt-4 mb-3">
        <h1 className="text-3xl font-bold text-zinc-900 tracking-tight">
          2023 Product Line
        </h1>
        <span className="text-xs font-medium bg-zinc-100 text-zinc-500 px-2 py-1 rounded">
          Archived
        </span>
      </div>

      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        Pricing and product details from the 2023 Northstar Devices range. These prices
        were superseded following the annual refresh. For current pricing, see{' '}
        <Link href="/facts/pricing" className="text-blue-600 hover:underline">
          current pricing
        </Link>
        .
      </p>

      <div className="rounded-xl border border-zinc-200 bg-white overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-200 bg-zinc-50">
              <th className="text-left px-5 py-3 font-medium text-zinc-500">Product</th>
              <th className="text-left px-5 py-3 font-medium text-zinc-500">2023 Price</th>
              <th className="text-left px-5 py-3 font-medium text-zinc-500">Notes</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100">
            {ARCHIVE_2023.map((p) => (
              <tr key={p.name}>
                <td className="px-5 py-4 font-medium text-zinc-900">{p.name}</td>
                <td className="px-5 py-4 text-zinc-700 font-semibold tabular-nums">
                  {p.currency}{p.price}
                </td>
                <td className="px-5 py-4 text-zinc-400 text-xs">{p.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-5 text-xs text-zinc-400">
        Prices shown are the 2023 list prices and do not reflect current retail pricing.
        For retired products, see{' '}
        <Link href="/archive/retired-devices" className="hover:text-zinc-600 underline">
          Retired Devices
        </Link>
        .
      </p>
    </div>
  );
}
