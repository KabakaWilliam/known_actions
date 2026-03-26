import Link from 'next/link';
import { CURRENT_PRODUCTS } from '@/lib/content';

export const metadata = { title: 'Devices — Northstar Devices' };

const STOCK_LABEL: Record<string, string> = {
  in_stock: 'In stock',
  low_stock: 'Low stock',
  out_of_stock: 'Out of stock',
};

const STOCK_CLASS: Record<string, string> = {
  in_stock: 'text-green-700 bg-green-50 border-green-200',
  low_stock: 'text-amber-700 bg-amber-50 border-amber-200',
  out_of_stock: 'text-red-700 bg-red-50 border-red-200',
};

export default function DevicesPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">Home</Link>
        <span className="mx-2">›</span>
        <Link href="/facts" className="hover:text-zinc-600">Facts</Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">Devices</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Current Devices
      </h1>
      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        The active Northstar product range. For pricing, visit{' '}
        <Link href="/facts/pricing" className="text-blue-600 hover:underline">
          Pricing
        </Link>
        .
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
        {CURRENT_PRODUCTS.map((p) => (
          <div
            key={p.slug}
            className="rounded-xl border border-zinc-200 bg-white p-5 flex flex-col gap-3"
          >
            <div className="flex items-start justify-between gap-2">
              <h2 className="font-semibold text-zinc-900 text-[15px]">{p.name}</h2>
              <span
                className={`shrink-0 text-xs font-medium border px-2 py-0.5 rounded ${STOCK_CLASS[p.stock]}`}
              >
                {STOCK_LABEL[p.stock]}
              </span>
            </div>
            <p className="text-xs text-zinc-400">Launched {p.launched}</p>
            <p className="text-sm text-zinc-500 leading-relaxed">{p.description}</p>
            <p className="text-xs italic text-zinc-400 mt-auto">{p.tagline}</p>
          </div>
        ))}
      </div>

      <div className="mt-8 text-sm text-zinc-500">
        See{' '}
        <Link href="/facts/pricing" className="text-blue-600 hover:underline">
          current pricing →
        </Link>
      </div>
    </div>
  );
}
