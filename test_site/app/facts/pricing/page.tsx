import Link from 'next/link';
import { CURRENT_PRODUCTS } from '@/lib/content';

export const metadata = { title: 'Pricing — Northstar Devices' };

const STOCK_LABEL: Record<string, string> = {
  in_stock: 'In stock',
  low_stock: 'Low stock',
  out_of_stock: 'Out of stock',
};

const STOCK_CLASS: Record<string, string> = {
  in_stock: 'text-green-700 bg-green-50',
  low_stock: 'text-amber-700 bg-amber-50',
  out_of_stock: 'text-red-700 bg-red-50',
};

export default function PricingPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">Home</Link>
        <span className="mx-2">›</span>
        <Link href="/facts" className="hover:text-zinc-600">Facts</Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">Pricing</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Current Pricing
      </h1>
      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        All prices shown are current retail prices inclusive of VAT, in pounds sterling.
        For archived pricing, see the{' '}
        <Link href="/archive/2023-line" className="text-blue-600 hover:underline">
          2023 archive
        </Link>
        .
      </p>

      <div className="rounded-xl border border-zinc-200 bg-white overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-200 bg-zinc-50">
              <th className="text-left px-5 py-3 font-medium text-zinc-500">Product</th>
              <th className="text-left px-5 py-3 font-medium text-zinc-500">Price</th>
              <th className="text-left px-5 py-3 font-medium text-zinc-500">Availability</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100">
            {CURRENT_PRODUCTS.map((p) => (
              <tr key={p.slug}>
                <td className="px-5 py-4 font-medium text-zinc-900">{p.name}</td>
                <td className="px-5 py-4 text-zinc-800 font-semibold tabular-nums">
                  {p.currency}{p.price}
                </td>
                <td className="px-5 py-4">
                  <span
                    className={`inline-block text-xs font-medium px-2 py-0.5 rounded ${STOCK_CLASS[p.stock]}`}
                  >
                    {STOCK_LABEL[p.stock]}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-5 text-xs text-zinc-400">
        Prices are updated periodically. For historical pricing, visit the{' '}
        <Link href="/archive" className="hover:text-zinc-600 underline">Archive</Link>.
      </p>
    </div>
  );
}
